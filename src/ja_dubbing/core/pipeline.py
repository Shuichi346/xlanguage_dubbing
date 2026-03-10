#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
メイン処理フロー。
1つの動画を英語→日本語吹き替え動画に変換する。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ja_dubbing.asr import get_asr_engine
from ja_dubbing.config import (
    KEEP_TEMP,
    MIN_SEGMENT_SEC,
    OUTPUT_SUFFIX,
    SPACY_CHUNK_GAP_SEC,
    SPACY_CHUNK_MAX_CHARS,
    SPACY_CHUNK_MAX_SEC,
    SPACY_UNIT_MAX_SENTENCES,
    SPACY_UNIT_MERGE_MAX_CHARS,
    SPACY_UNIT_MERGE_MAX_GAP_SEC,
)
from ja_dubbing.core.models import TtsMeta
from ja_dubbing.core.progress import ProgressStore
from ja_dubbing.core.retime import build_retime_parts
from ja_dubbing.audio.ffmpeg import (
    concat_audio_to_flac,
    concat_ts_files,
    create_silence_flac,
    encode_english_audio_chunk_flac,
    encode_video_chunk_ts,
    ffprobe_duration_sec,
    ffprobe_has_audio,
    mux_retimed_video_with_tracks,
    remux_ts_to_mp4,
)
from ja_dubbing.audio.segment_io import (
    load_segments_json,
    save_segments_json_atomic,
    save_srt_atomic,
)
from ja_dubbing.asr.whisper import extract_wav_for_whisper
from ja_dubbing.segments.merge import merge_segments
from ja_dubbing.segments.sentence import merge_sentence_units
from ja_dubbing.segments.spacy_split import (
    chunk_segments_for_spacy,
    split_segments_by_spacy_sentences,
)
from ja_dubbing.translation.plamo import (
    PlamoTranslateClient,
    translate_segments_resumable,
)
from ja_dubbing.tts.miotts import (
    generate_segment_tts,
    load_tts_meta,
    save_tts_meta_atomic,
)
from ja_dubbing.tts.reference import SpeakerReferenceCache
from ja_dubbing.utils import (
    PipelineError,
    ensure_dir,
    force_memory_cleanup,
    print_step,
    sanitize_text_for_tts,
)


def process_one_video(
    video_path: Path,
    client: PlamoTranslateClient,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """1つの動画を処理する。"""
    print_step(f"=== 開始: {video_path} ===")

    out_path = video_path.with_name(video_path.stem + OUTPUT_SUFFIX)
    if out_path.exists():
        print_step(f"スキップ（出力済み）: {out_path}")
        return

    work_dir = ref_cache.cache_dir.parent
    ensure_dir(work_dir)

    progress = ProgressStore(work_dir / "progress.json", video_path)
    progress.load()
    progress.save()

    wav_whisper = work_dir / "audio_whisper_16k.wav"

    seg_json_en = work_dir / "segments_en.json"
    seg_json_enja = work_dir / "segments_en_ja.json"
    srt_en_path = work_dir / "subtitles_en.srt"

    seg_audio_dir = work_dir / "seg_audio"
    ensure_dir(seg_audio_dir)

    asr_engine = get_asr_engine()
    print_step(f"ASR エンジン: {asr_engine}")

    # ===== 0. 動画尺 =====
    print_step("0. 動画尺を取得(ffprobe)")
    if (
        progress.step("probe_done")
        and isinstance(
            progress.data.get("artifacts", {}).get("video_duration_sec"),
            (int, float),
        )
    ):
        video_dur = float(progress.data["artifacts"]["video_duration_sec"])
        print_step(f"   動画尺(再開): {video_dur:.3f} sec")
    else:
        video_dur = ffprobe_duration_sec(video_path)
        progress.set_artifact("video_duration_sec", video_dur)
        progress.set_step("probe_done", True)
        progress.save()
        print_step(f"   動画尺: {video_dur:.3f} sec")

    # ===== 1. 音声抽出 =====
    if not wav_whisper.exists():
        print_step("1. ffmpegで音声抽出（16kHz mono wav）")
        extract_wav_for_whisper(video_path, wav_whisper)
    else:
        print_step("1. 既存の音声抽出 wav を利用")

    # ===== 2-5. 文字起こし〜セグメント加工 =====
    if progress.step("asr_done") and seg_json_en.exists():
        print_step("2-5. ASR+話者分離: 既に完了（segments_en.json から再開）")
        segments_en = load_segments_json(seg_json_en)

        if not srt_en_path.exists():
            save_srt_atomic(segments_en, srt_en_path)

        diarization = None
    else:
        if asr_engine == "vibevoice":
            from ja_dubbing.asr.vibevoice import (
                release_vibevoice_model,
                vibevoice_transcribe,
            )

            print_step("2. VibeVoice-ASR で文字起こし（話者分離内蔵）")
            segments_raw, diarization = vibevoice_transcribe(wav_whisper)
            if not segments_raw:
                raise PipelineError("VibeVoice-ASR セグメントが空です。")
            print_step(f"   セグメント数: {len(segments_raw)}")

            release_vibevoice_model()
            force_memory_cleanup()

            save_srt_atomic(segments_raw, srt_en_path)
            progress.set_artifact("asr_srt_en", str(srt_en_path))
            progress.save()

            print_step("3. 話者分離: VibeVoice-ASR 内蔵のため省略")
            print_step("4. 話者ID割り当て: VibeVoice-ASR 内蔵のため省略")

            segments_with_speaker = segments_raw
        else:
            from ja_dubbing.asr.whisper import (
                release_whisper_model,
                whisper_transcribe,
            )
            from ja_dubbing.diarization.alignment import assign_speakers
            from ja_dubbing.diarization.speaker import (
                release_pipeline,
                run_diarization,
            )

            print_step("2. whisper.cpp CLIで文字起こし")
            segments_raw = whisper_transcribe(wav_whisper)
            if not segments_raw:
                raise PipelineError("Whisper セグメントが空です。")
            print_step(f"   Whisper rawセグメント数: {len(segments_raw)}")

            release_whisper_model()

            save_srt_atomic(segments_raw, srt_en_path)
            progress.set_artifact("asr_srt_en", str(srt_en_path))
            progress.save()

            # 3. 話者分離
            print_step("3. pyannote.audio で話者分離")
            diarization = run_diarization(wav_whisper)

            progress.set_step("diarization_done", True)
            progress.save()

            # 4. 話者ID割り当て
            print_step("4. Whisperセグメントに話者ID割り当て")
            segments_with_speaker = assign_speakers(segments_raw, diarization)
            speaker_ids = set(s.speaker_id for s in segments_with_speaker)
            print_step(f"   話者数: {len(speaker_ids)}, ID: {sorted(speaker_ids)}")

            # pyannote パイプラインとtorchメモリを解放
            release_pipeline()
            force_memory_cleanup()

        # 5. セグメント加工（結合 → spaCy文分割 → 翻訳ユニット結合）
        print_step(
            f"5. セグメント加工: {len(segments_with_speaker)} -> 結合中..."
        )
        segments_merged = merge_segments(segments_with_speaker)
        print_step(f"   結合後セグメント数: {len(segments_merged)}")

        print_step("   spaCy用にチャンク化 → 文単位に再分割 → 2文結合")
        chunks = chunk_segments_for_spacy(
            segments_merged,
            max_sec=SPACY_CHUNK_MAX_SEC,
            max_chars=SPACY_CHUNK_MAX_CHARS,
            max_gap_sec=SPACY_CHUNK_GAP_SEC,
        )
        print_step(
            f"   チャンク数: {len(chunks)}（最大{SPACY_CHUNK_MAX_SEC:.1f}s）"
        )

        segments_sent = split_segments_by_spacy_sentences(chunks)
        print_step(f"   spaCy文分割後セグメント数: {len(segments_sent)}")

        segments_en = merge_sentence_units(
            segments_sent,
            max_sentences=SPACY_UNIT_MAX_SENTENCES,
            merge_max_chars=SPACY_UNIT_MERGE_MAX_CHARS,
            max_gap_sec=SPACY_UNIT_MERGE_MAX_GAP_SEC,
        )
        print_step(f"   2文結合後セグメント数: {len(segments_en)}")

        save_segments_json_atomic(segments_en, seg_json_en)
        progress.set_artifact("segments_en_json", str(seg_json_en))
        progress.set_step("asr_done", True)
        progress.save()

    if not segments_en:
        raise PipelineError("segments_en が空です。")

    # ===== 6. 話者リファレンス音声生成 =====
    if diarization is not None:
        print_step("6. 話者ごとの代表リファレンス音声を抽出")
        ref_cache.build_references(video_path, diarization)
    else:
        _reload_cached_references(ref_cache, segments_en)

    # ===== 6.5. セグメント単位リファレンス音声生成 =====
    print_step("6.5. セグメント単位のリファレンス音声を抽出（感情・テンポ対応）")
    ref_cache.build_segment_references(video_path, segments_en)

    # ===== 7. 翻訳 =====
    print_step("7. plamo-translate-cli (PLaMo2 Translate MLX) で翻訳（再開対応）")
    if progress.step("translate_done") and seg_json_enja.exists():
        print_step("   翻訳: 既に完了（segments_en_ja.json から再開）")
        segments_enja = load_segments_json(seg_json_enja)
        if len(segments_enja) != len(segments_en):
            print_step(
                "   注意: セグメント数が一致しないため翻訳をやり直します。"
            )
            segments_enja = translate_segments_resumable(
                client, segments_en, seg_json_enja, progress
            )
    else:
        segments_enja = translate_segments_resumable(
            client, segments_en, seg_json_enja, progress
        )

    # ===== 8. TTS（話者クローン） =====
    print_step("8. MioTTS で話者クローン日本語音声生成（セグメント単位リファレンス優先）")

    tts_meta_path = work_dir / "tts_meta.json"
    tts_meta = load_tts_meta(tts_meta_path)

    total = len(segments_enja)
    tts_done = 0
    tts_failed = 0

    for segno, seg in enumerate(segments_enja, start=1):
        if seg.duration < MIN_SEGMENT_SEC:
            tts_done += 1
            continue

        text = sanitize_text_for_tts(seg.text_ja)
        if not text:
            tts_done += 1
            continue

        out_audio_stub = seg_audio_dir / f"seg_{segno:05d}"
        out_flac = out_audio_stub.with_suffix(".flac")

        if segno in tts_meta and Path(tts_meta[segno].flac_path).exists():
            tts_done += 1
            if segno % 50 == 0:
                progress.set_step(
                    "tts", {"done_count": tts_done, "total": total}
                )
                progress.save()
            continue

        if out_flac.exists():
            dur = ffprobe_duration_sec(out_flac)
            if dur > 0:
                tts_meta[segno] = TtsMeta(
                    segno=segno,
                    flac_path=str(out_flac),
                    duration_sec=float(dur),
                )
                save_tts_meta_atomic(tts_meta_path, tts_meta)
                tts_done += 1
                progress.set_step(
                    "tts", {"done_count": tts_done, "total": total}
                )
                progress.save()
                continue

        has_seg_ref = ref_cache.get_segment_reference_path(segno) is not None
        ref_type = "セグメント単位" if has_seg_ref else "話者代表"
        print_step(
            f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
            f"speaker={seg.speaker_id} ref={ref_type}"
        )

        try:
            meta0 = generate_segment_tts(
                seg, out_audio_stub, ref_cache, segno=segno
            )
        except Exception as exc:
            print_step(
                f"  TTS失敗（スキップ）: seg {segno}/{total}: {exc}"
            )
            meta0 = None
            tts_failed += 1

        if meta0 is None:
            tts_done += 1
            continue

        tts_meta[segno] = TtsMeta(
            segno=segno,
            flac_path=meta0.flac_path,
            duration_sec=meta0.duration_sec,
        )
        save_tts_meta_atomic(tts_meta_path, tts_meta)

        tts_done += 1
        progress.set_step("tts", {"done_count": tts_done, "total": total})
        progress.set_artifact("tts_meta_json", str(tts_meta_path))
        progress.save()

    if tts_failed > 0:
        print_step(f"  TTS失敗セグメント数: {tts_failed}/{total}")

    progress.set_step("tts", {"done_count": total, "total": total})
    progress.set_artifact("tts_meta_json", str(tts_meta_path))
    progress.save()

    # TTS完了後にメモリクリーンアップ
    force_memory_cleanup()

    # ===== 9. リタイム =====
    print_step("9. TTSに合わせて元動画の速度を区間ごとに変更（リタイム）")

    has_audio = ffprobe_has_audio(video_path)

    parts, new_total_dur = build_retime_parts(segments_enja, tts_meta, video_dur)
    progress.set_artifact("retimed_total_duration_sec", new_total_dur)
    progress.save()

    print_step(
        f"   リタイム後の総尺: {new_total_dur:.3f} sec（元: {video_dur:.3f} sec）"
    )
    print_step(f"   パート数: {len(parts)}")

    retime_dir = work_dir / "retime"
    video_chunk_dir = retime_dir / "video_chunks"
    ja_chunk_dir = retime_dir / "ja_chunks"
    en_chunk_dir = retime_dir / "en_chunks"
    ensure_dir(video_chunk_dir)
    ensure_dir(ja_chunk_dir)
    ensure_dir(en_chunk_dir)

    # 9-1. 映像チャンク作成
    print_step("   9-1. 映像チャンクを生成（setptsで速度変更）")
    video_chunks: list[Path] = []
    for i, part in enumerate(parts, start=1):
        out_ts = video_chunk_dir / f"v_{i:05d}.ts"
        video_chunks.append(out_ts)
        if out_ts.exists():
            continue
        encode_video_chunk_ts(
            video_path,
            out_ts,
            start=part.orig_start,
            end=part.orig_end,
            speed=part.speed,
        )

    video_ts = retime_dir / "video_retimed.ts"
    video_list = retime_dir / "video_concat.txt"
    if not video_ts.exists():
        print_step("   9-2. 映像チャンクを結合（TS concat）")
        concat_ts_files(video_chunks, video_ts, video_list)

    video_retimed_mp4 = retime_dir / "video_retimed.mp4"
    if not video_retimed_mp4.exists():
        print_step("   9-3. 映像TSをMP4へリマックス")
        remux_ts_to_mp4(video_ts, video_retimed_mp4)

    progress.set_artifact("retimed_video_mp4", str(video_retimed_mp4))
    progress.save()

    # 9-4. 英語音声トラック
    english_retimed_flac: Path | None = None
    if has_audio:
        print_step(
            "   9-4. 英語音声を同じ倍率でリタイム（atempo）して結合"
        )
        en_chunks: list[Path] = []
        for i, part in enumerate(parts, start=1):
            out_flac = en_chunk_dir / f"en_{i:05d}.flac"
            en_chunks.append(out_flac)
            if out_flac.exists():
                continue
            encode_english_audio_chunk_flac(
                video_path,
                out_flac,
                start=part.orig_start,
                end=part.orig_end,
                speed=part.speed,
            )

        english_retimed_flac = retime_dir / "english_retimed.flac"
        en_list = retime_dir / "english_concat.txt"
        if not english_retimed_flac.exists():
            concat_audio_to_flac(en_chunks, english_retimed_flac, en_list)

        progress.set_artifact(
            "english_retimed_flac", str(english_retimed_flac)
        )
        progress.save()
    else:
        print_step(
            "   9-4. 元動画に音声が無いため、英語トラックは作りません。"
        )

    # 9-5. 日本語音声トラック
    print_step("   9-5. 日本語トラックを生成（無音 + TTS を順に結合）")
    ja_items: list[Path] = []
    for i, part in enumerate(parts, start=1):
        if part.kind in ("gap", "tail"):
            sil = ja_chunk_dir / f"ja_{i:05d}_sil.flac"
            ja_items.append(sil)
            if not sil.exists():
                create_silence_flac(sil, part.out_duration)
            continue

        meta = tts_meta.get(part.segno)
        if meta and Path(meta.flac_path).exists():
            ja_items.append(Path(meta.flac_path))
        else:
            sil = ja_chunk_dir / f"ja_{i:05d}_missing.flac"
            ja_items.append(sil)
            if not sil.exists():
                create_silence_flac(sil, part.out_duration)

    japanese_full_flac = retime_dir / "japanese_full.flac"
    ja_list = retime_dir / "japanese_concat.txt"
    if not japanese_full_flac.exists():
        concat_audio_to_flac(ja_items, japanese_full_flac, ja_list)

    progress.set_artifact("japanese_full_flac", str(japanese_full_flac))
    progress.set_step("retime_done", True)
    progress.save()

    # ===== 10. mux =====
    if progress.step("mux_done") and out_path.exists():
        print_step(f"=== 完了（mux済み）: {out_path} ===")
    else:
        print_step(
            "10. リタイム済み映像 + 日本語音声（+英語薄く）を合成して出力"
        )
        mux_retimed_video_with_tracks(
            video_retimed_mp4,
            japanese_full_flac,
            out_path,
            english_flac=english_retimed_flac,
        )
        progress.set_step("mux_done", True)
        progress.save()
        print_step(f"=== 完了: {out_path} ===")

    if not KEEP_TEMP:
        print_step(f"一時フォルダ削除: {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)

    # 動画1本の処理完了後にメモリを解放する
    force_memory_cleanup()
    print_step("メモリクリーンアップ実行済み")


def _reload_cached_references(
    ref_cache: SpeakerReferenceCache,
    segments: list,
) -> None:
    """再開時にキャッシュ済みリファレンスを検出してロードする。"""
    speaker_ids = set(s.speaker_id for s in segments if s.speaker_id)
    ref_cache.reload_speaker_references(speaker_ids)
    ref_cache.reload_segment_references(len(segments))
