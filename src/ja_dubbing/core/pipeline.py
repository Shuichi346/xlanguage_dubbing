#!/usr/bin/env python3
"""
メイン処理フロー。
1つの動画を英語→日本語吹き替え動画に変換する。
TTS エンジンとして MioTTS（話者クローン対応）、Kokoro（高速・クローン非対応）、
GPT-SoVITS（V2ProPlus ゼロショットボイスクローン）、
T5Gemma-TTS（ゼロショットボイスクローン + 再生時間制御）を選択可能。
翻訳エンジンは CAT-Translate-7b (GGUF, llama-cpp-python) を使用する。
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from ja_dubbing.asr import get_asr_engine
from ja_dubbing.asr.whisper import extract_wav_for_whisper
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
    TTS_ENGINE,
)
from ja_dubbing.core.models import TtsMeta
from ja_dubbing.core.progress import ProgressStore
from ja_dubbing.core.retime import build_retime_parts
from ja_dubbing.segments.merge import merge_segments
from ja_dubbing.segments.sentence import merge_sentence_units
from ja_dubbing.segments.spacy_split import (
    chunk_segments_for_spacy,
    split_segments_by_spacy_sentences,
)
from ja_dubbing.translation.cat_translate import (
    CatTranslateClient,
    translate_segments_resumable,
)
from ja_dubbing.tts.miotts import (
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


def _get_tts_engine() -> str:
    """現在選択されている TTS エンジン名を返す。"""
    return TTS_ENGINE.strip().lower()


def _needs_speaker_diarization(tts_engine: str) -> bool:
    """話者分離が必要かどうかを判定する。"""
    # Kokoro はクローン非対応なので話者分離不要
    # MioTTS / GPT-SoVITS / T5Gemma-TTS は話者クローンのため話者分離が必要
    return tts_engine != "kokoro"


def process_one_video(
    video_path: Path,
    client: CatTranslateClient,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """1つの動画を処理する。"""
    print_step(f"=== 開始: {video_path} ===")

    tts_engine = _get_tts_engine()
    need_diarization = _needs_speaker_diarization(tts_engine)

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
    print_step(f"TTS エンジン: {tts_engine}")
    if not need_diarization:
        print_step("話者分離: 不要（Kokoro TTS はクローン非対応のため）")

    # ===== 0. 動画尺 =====
    print_step("0. 動画尺を取得(ffprobe)")
    if (
        progress.step("probe_done")
        and isinstance(
            progress.data.get("artifacts", {}).get("video_duration_sec"),
            int | float,
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
        print_step("2-5. ASR+セグメント加工: 既に完了（segments_en.json から再開）")
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

            if need_diarization:
                print_step("3. 話者分離: VibeVoice-ASR 内蔵のため省略")
                print_step("4. 話者ID割り当て: VibeVoice-ASR 内蔵のため省略")
            else:
                print_step("3. 話者分離: Kokoro TTS のため省略")
                print_step("4. 話者ID割り当て: Kokoro TTS のため省略")

            segments_with_speaker = segments_raw
        else:
            from ja_dubbing.asr.whisper import (
                release_whisper_model,
                whisper_transcribe,
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

            if need_diarization:
                from ja_dubbing.diarization.alignment import assign_speakers
                from ja_dubbing.diarization.speaker import (
                    release_pipeline,
                    run_diarization,
                )

                print_step("3. pyannote.audio で話者分離")
                diarization = run_diarization(wav_whisper)

                progress.set_step("diarization_done", True)
                progress.save()

                print_step("4. Whisperセグメントに話者ID割り当て")
                segments_with_speaker = assign_speakers(segments_raw, diarization)
                speaker_ids = {s.speaker_id for s in segments_with_speaker}
                print_step(f"   話者数: {len(speaker_ids)}, ID: {sorted(speaker_ids)}")

                release_pipeline()
                force_memory_cleanup()
            else:
                print_step("3. 話者分離: Kokoro TTS のため省略（pyannote 不使用）")
                print_step("4. 話者ID割り当て: Kokoro TTS のため省略（統一話者ID）")

                from ja_dubbing.core.models import Segment

                segments_with_speaker = [
                    Segment(
                        idx=s.idx,
                        start=s.start,
                        end=s.end,
                        text_en=s.text_en,
                        text_ja=s.text_ja,
                        speaker_id="KOKORO",
                    )
                    for s in segments_raw
                ]
                diarization = None

                progress.set_step("diarization_done", True)
                progress.save()

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
    if tts_engine == "miotts":
        if diarization is not None:
            print_step("6. 話者ごとの代表リファレンス音声を抽出（MioTTS）")
            ref_cache.build_references(video_path, diarization)
        else:
            _reload_cached_references(ref_cache, segments_en)

        print_step("6.5. セグメント単位のリファレンス音声を抽出（感情・テンポ対応）")
        ref_cache.build_segment_references(video_path, segments_en)

    elif tts_engine == "gptsovits":
        if diarization is not None:
            print_step(
                "6. GPT-SoVITS 用の話者代表リファレンス音声を抽出"
                "（3〜10秒、声質のみ）"
            )
            ref_cache.build_gptsovits_references(
                video_path, diarization
            )
        else:
            _reload_cached_references(ref_cache, segments_en)

        print_step(
            "6.5. セグメント単位リファレンス: 不要"
            "（GPT-SoVITS は声質のみ抽出のため話者代表を使い回す）"
        )

    elif tts_engine == "t5gemma":
        if diarization is not None:
            print_step(
                "6. T5Gemma-TTS 用の話者代表リファレンス音声を抽出"
                "（3〜15秒、ボイスクローン用）"
            )
            ref_cache.build_t5gemma_references(
                video_path, diarization, segments_en
            )
        else:
            _reload_cached_references(ref_cache, segments_en)

        print_step(
            "6.5. T5Gemma-TTS セグメント単位リファレンス音声を抽出"
            "（ASR 再文字起こしで音声・テキスト一致を保証）"
        )
        ref_cache.build_t5gemma_segment_references(video_path, segments_en)

    else:
        print_step("6. リファレンス音声抽出: Kokoro TTS（クローン非対応）のため省略")

    # ===== 7. 翻訳 =====
    print_step("7. CAT-Translate-7b (GGUF, llama-cpp-python) で翻訳（再開対応）")
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

    # ===== 8. TTS =====
    if tts_engine == "kokoro":
        _run_tts_kokoro(segments_enja, seg_audio_dir, work_dir, progress)
    elif tts_engine == "gptsovits":
        _run_tts_gptsovits(
            segments_enja, seg_audio_dir, work_dir, progress, ref_cache
        )
    elif tts_engine == "t5gemma":
        _run_tts_t5gemma(
            segments_enja, seg_audio_dir, work_dir, progress, ref_cache
        )
    else:
        _run_tts_miotts(segments_enja, seg_audio_dir, work_dir, progress, ref_cache)

    force_memory_cleanup()

    tts_meta_path = work_dir / "tts_meta.json"
    tts_meta = load_tts_meta(tts_meta_path)

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

    force_memory_cleanup()
    print_step("メモリクリーンアップ実行済み")


def _should_skip_tts_segment(seg) -> bool:
    """TTS対象外のセグメントかどうかを返す。"""
    if seg.duration < MIN_SEGMENT_SEC:
        return True
    return not sanitize_text_for_tts(seg.text_ja)


def _save_tts_meta_entry(
    tts_meta: dict[int, TtsMeta],
    tts_meta_path: Path,
    segno: int,
    flac_path: str,
    duration_sec: float,
) -> None:
    """TTSメタ情報を1件保存する。"""
    tts_meta[segno] = TtsMeta(
        segno=segno,
        flac_path=flac_path,
        duration_sec=float(duration_sec),
    )
    save_tts_meta_atomic(tts_meta_path, tts_meta)


def _update_tts_progress(
    progress,
    done_count: int,
    total: int,
    tts_meta_path: Path,
    save_artifact: bool = False,
) -> None:
    """TTS進捗を保存する。"""
    progress.set_step("tts", {"done_count": done_count, "total": total})
    if save_artifact:
        progress.set_artifact("tts_meta_json", str(tts_meta_path))
    progress.save()


def _reuse_existing_tts_output(
    segno: int,
    out_flac: Path,
    tts_meta: dict[int, TtsMeta],
    tts_meta_path: Path,
) -> str | None:
    """既存TTS出力を再利用できる場合は理由を返す。"""
    if segno in tts_meta and Path(tts_meta[segno].flac_path).exists():
        return "meta"

    if out_flac.exists():
        dur = ffprobe_duration_sec(out_flac)
        if dur > 0:
            _save_tts_meta_entry(
                tts_meta,
                tts_meta_path,
                segno=segno,
                flac_path=str(out_flac),
                duration_sec=dur,
            )
            return "flac"

    return None


def _run_tts_loop(
    segments_enja: list,
    seg_audio_dir: Path,
    work_dir: Path,
    progress,
    segment_label_builder: Callable[[int, int, object], str],
    generator: Callable[[object, Path, int], TtsMeta | None],
) -> None:
    """各TTSエンジン共通のセグメント処理ループ。"""
    tts_meta_path = work_dir / "tts_meta.json"
    tts_meta = load_tts_meta(tts_meta_path)

    total = len(segments_enja)
    tts_done = 0
    tts_failed = 0

    for segno, seg in enumerate(segments_enja, start=1):
        if _should_skip_tts_segment(seg):
            tts_done += 1
            continue

        out_audio_stub = seg_audio_dir / f"seg_{segno:05d}"
        out_flac = out_audio_stub.with_suffix(".flac")

        reused = _reuse_existing_tts_output(
            segno=segno,
            out_flac=out_flac,
            tts_meta=tts_meta,
            tts_meta_path=tts_meta_path,
        )
        if reused is not None:
            tts_done += 1
            if reused == "meta":
                if segno % 50 == 0:
                    _update_tts_progress(progress, tts_done, total, tts_meta_path)
            else:
                _update_tts_progress(progress, tts_done, total, tts_meta_path)
            continue

        print_step(segment_label_builder(segno, total, seg))

        try:
            meta0 = generator(seg, out_audio_stub, segno)
        except Exception as exc:
            print_step(f"  TTS失敗（スキップ）: seg {segno}/{total}: {exc}")
            meta0 = None
            tts_failed += 1

        if meta0 is None:
            tts_done += 1
            continue

        _save_tts_meta_entry(
            tts_meta,
            tts_meta_path,
            segno=segno,
            flac_path=meta0.flac_path,
            duration_sec=meta0.duration_sec,
        )

        tts_done += 1
        _update_tts_progress(
            progress,
            done_count=tts_done,
            total=total,
            tts_meta_path=tts_meta_path,
            save_artifact=True,
        )

    if tts_failed > 0:
        print_step(f"  TTS失敗セグメント数: {tts_failed}/{total}")

    _update_tts_progress(
        progress,
        done_count=total,
        total=total,
        tts_meta_path=tts_meta_path,
        save_artifact=True,
    )


def _run_tts_kokoro(
    segments_enja: list,
    seg_audio_dir: Path,
    work_dir: Path,
    progress,
) -> None:
    """Kokoro TTS でセグメントの日本語音声を生成する。"""
    from ja_dubbing.tts.kokoro_tts import generate_segment_tts_kokoro

    print_step("8. Kokoro TTS で日本語音声生成（高速・クローン非対応）")

    def _build_segment_label(segno: int, total: int, seg: object) -> str:
        return f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} (Kokoro)"

    def _generate(seg: object, out_audio_stub: Path, segno: int) -> TtsMeta | None:
        return generate_segment_tts_kokoro(seg, out_audio_stub, segno=segno)

    _run_tts_loop(
        segments_enja=segments_enja,
        seg_audio_dir=seg_audio_dir,
        work_dir=work_dir,
        progress=progress,
        segment_label_builder=_build_segment_label,
        generator=_generate,
    )


def _run_tts_gptsovits(
    segments_enja: list,
    seg_audio_dir: Path,
    work_dir: Path,
    progress,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """GPT-SoVITS でゼロショットボイスクローン日本語音声を生成する。"""
    from ja_dubbing.tts.gptsovits import generate_segment_tts_gptsovits

    print_step(
        "8. GPT-SoVITS V2ProPlus でゼロショットボイスクローン日本語音声生成"
        "（話者代表リファレンス使用）"
    )

    def _build_segment_label(segno: int, total: int, seg: object) -> str:
        return (
            f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
            f"speaker={seg.speaker_id} (GPT-SoVITS)"
        )

    def _generate(seg: object, out_audio_stub: Path, segno: int) -> TtsMeta | None:
        return generate_segment_tts_gptsovits(
            seg, out_audio_stub, ref_cache, segno=segno
        )

    _run_tts_loop(
        segments_enja=segments_enja,
        seg_audio_dir=seg_audio_dir,
        work_dir=work_dir,
        progress=progress,
        segment_label_builder=_build_segment_label,
        generator=_generate,
    )


def _run_tts_t5gemma(
    segments_enja: list,
    seg_audio_dir: Path,
    work_dir: Path,
    progress,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """T5Gemma-TTS でボイスクローン日本語音声を生成する。"""
    from ja_dubbing.tts.t5gemma_tts import (
        generate_segment_tts_t5gemma,
        release_t5gemma_model,
    )

    print_step(
        "8. T5Gemma-TTS でボイスクローン日本語音声生成"
        "（セグメント単位リファレンス優先、再生時間制御あり）"
    )

    try:
        def _build_segment_label(segno: int, total: int, seg: object) -> str:
            has_seg_ref = (
                ref_cache.get_t5gemma_segment_reference_path(segno) is not None
            )
            ref_type = "セグメント単位" if has_seg_ref else "話者代表"
            return (
                f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
                f"speaker={seg.speaker_id} ref={ref_type} (T5Gemma)"
            )

        def _generate(
            seg: object,
            out_audio_stub: Path,
            segno: int,
        ) -> TtsMeta | None:
            return generate_segment_tts_t5gemma(
                seg, out_audio_stub, ref_cache, segno=segno
            )

        _run_tts_loop(
            segments_enja=segments_enja,
            seg_audio_dir=seg_audio_dir,
            work_dir=work_dir,
            progress=progress,
            segment_label_builder=_build_segment_label,
            generator=_generate,
        )
    finally:
        release_t5gemma_model()


def _run_tts_miotts(
    segments_enja: list,
    seg_audio_dir: Path,
    work_dir: Path,
    progress,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """MioTTS で話者クローン日本語音声を生成する。"""
    from ja_dubbing.tts.miotts import generate_segment_tts

    print_step("8. MioTTS で話者クローン日本語音声生成（セグメント単位リファレンス優先）")

    def _build_segment_label(segno: int, total: int, seg: object) -> str:
        has_seg_ref = ref_cache.get_segment_reference_path(segno) is not None
        ref_type = "セグメント単位" if has_seg_ref else "話者代表"
        return (
            f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
            f"speaker={seg.speaker_id} ref={ref_type}"
        )

    def _generate(seg: object, out_audio_stub: Path, segno: int) -> TtsMeta | None:
        return generate_segment_tts(seg, out_audio_stub, ref_cache, segno=segno)

    _run_tts_loop(
        segments_enja=segments_enja,
        seg_audio_dir=seg_audio_dir,
        work_dir=work_dir,
        progress=progress,
        segment_label_builder=_build_segment_label,
        generator=_generate,
    )


def _reload_cached_references(
    ref_cache: SpeakerReferenceCache,
    segments: list,
) -> None:
    """再開時にキャッシュ済みリファレンスを検出してロードする。"""
    speaker_ids = {s.speaker_id for s in segments if s.speaker_id}
    ref_cache.reload_speaker_references(speaker_ids)
    ref_cache.reload_segment_references(len(segments))
    ref_cache.reload_t5gemma_segment_references(len(segments))
