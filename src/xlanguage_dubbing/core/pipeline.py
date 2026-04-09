#!/usr/bin/env python3
"""
メイン処理フロー。
1つの動画を多言語吹き替え動画に変換する。
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from xlanguage_dubbing.asr import get_asr_engine
from xlanguage_dubbing.asr.whisper import extract_wav_for_whisper
from xlanguage_dubbing.audio.ffmpeg import (
    concat_audio_to_flac,
    concat_ts_files,
    create_silence_flac,
    encode_original_audio_chunk_flac,
    encode_video_chunk_ts,
    ffprobe_duration_sec,
    ffprobe_has_audio,
    mux_retimed_video_with_tracks,
    remux_ts_to_mp4,
)
from xlanguage_dubbing.audio.segment_io import (
    load_segments_json,
    save_segments_json_atomic,
    save_srt_atomic,
)
from xlanguage_dubbing.config import (
    INPUT_LANG,
    KEEP_TEMP,
    MIN_SEGMENT_SEC,
    OUTPUT_LANG,
    OUTPUT_SUFFIX,
    SPACY_CHUNK_GAP_SEC,
    SPACY_CHUNK_MAX_CHARS,
    SPACY_CHUNK_MAX_SEC,
    SPACY_UNIT_MAX_SENTENCES,
    SPACY_UNIT_MERGE_MAX_CHARS,
    SPACY_UNIT_MERGE_MAX_GAP_SEC,
)
from xlanguage_dubbing.core.models import TtsMeta
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.core.retime import build_retime_parts
from xlanguage_dubbing.lang_utils import (
    detect_segments_language,
    get_lang_name,
    select_translation_engine,
)
from xlanguage_dubbing.segments.merge import merge_segments
from xlanguage_dubbing.segments.sentence import merge_sentence_units
from xlanguage_dubbing.segments.spacy_split import (
    chunk_segments_for_spacy,
    split_segments_by_spacy_sentences,
)
from xlanguage_dubbing.translation.cat_translate import (
    CatTranslateClient,
    release_all_translation_models,
    translate_segments_resumable,
)
from xlanguage_dubbing.omnivoice_tts import (
    load_tts_meta,
    save_tts_meta_atomic,
)
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache
from xlanguage_dubbing.utils import (
    PipelineError,
    ensure_dir,
    force_memory_cleanup,
    print_step,
    sanitize_text_for_tts,
)


def process_one_video(
    video_path: Path,
    client: CatTranslateClient,
    ref_cache: SpeakerReferenceCache,
) -> None:
    """1つの動画を処理する。"""
    print_step(f"=== 開始: {video_path} ===")
    print_step(f"  入力言語: {INPUT_LANG} → 出力言語: {OUTPUT_LANG}")

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

    seg_json_src = work_dir / "segments_src.json"
    seg_json_translated = work_dir / "segments_translated.json"
    srt_src_path = work_dir / "subtitles_src.srt"

    seg_audio_dir = work_dir / "seg_audio"
    ensure_dir(seg_audio_dir)

    asr_engine = get_asr_engine()
    print_step(f"ASR エンジン: {asr_engine}")
    print_step("TTS エンジン: OmniVoice")

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
    detected_lang = ""

    if progress.step("asr_done") and seg_json_src.exists():
        print_step("2-5. ASR+セグメント加工: 既に完了（再開）")
        segments_src = load_segments_json(seg_json_src)

        # 検出言語を再取得する
        detected_lang = progress.data.get("artifacts", {}).get(
            "detected_lang", ""
        )
        if not detected_lang:
            detected_lang = detect_segments_language(segments_src, INPUT_LANG)
            progress.set_artifact("detected_lang", detected_lang)
            progress.save()

        if not srt_src_path.exists():
            save_srt_atomic(segments_src, srt_src_path)

        diarization = None
    else:
        if asr_engine == "vibevoice":
            from xlanguage_dubbing.asr.vibevoice import (
                release_vibevoice_model,
                vibevoice_transcribe,
            )

            print_step("2. VibeVoice-ASR で文字起こし（コードスイッチング対応）")
            segments_raw, diarization = vibevoice_transcribe(wav_whisper)
            if not segments_raw:
                raise PipelineError("VibeVoice-ASR セグメントが空です。")
            print_step(f"   セグメント数: {len(segments_raw)}")

            release_vibevoice_model()
            force_memory_cleanup()

            detected_lang = detect_segments_language(segments_raw, INPUT_LANG)
            progress.set_artifact("detected_lang", detected_lang)

            save_srt_atomic(segments_raw, srt_src_path)
            progress.set_artifact("asr_srt", str(srt_src_path))
            progress.save()

            print_step("3. 話者分離: VibeVoice-ASR 内蔵のため省略")
            print_step("4. 話者ID割り当て: VibeVoice-ASR 内蔵のため省略")

            segments_with_speaker = segments_raw
        else:
            from xlanguage_dubbing.asr.whisper import (
                release_whisper_model,
                whisper_transcribe,
            )

            print_step("2. whisper.cpp CLIで文字起こし")
            segments_raw, whisper_detected_lang = whisper_transcribe(wav_whisper)
            if not segments_raw:
                raise PipelineError("Whisper セグメントが空です。")
            print_step(f"   Whisper rawセグメント数: {len(segments_raw)}")

            if whisper_detected_lang:
                detected_lang = whisper_detected_lang
            else:
                detected_lang = detect_segments_language(
                    segments_raw, INPUT_LANG
                )
            progress.set_artifact("detected_lang", detected_lang)

            release_whisper_model()

            save_srt_atomic(segments_raw, srt_src_path)
            progress.set_artifact("asr_srt", str(srt_src_path))
            progress.save()

            from xlanguage_dubbing.diarization.alignment import assign_speakers
            from xlanguage_dubbing.diarization.speaker import (
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
            print_step(
                f"   話者数: {len(speaker_ids)}, ID: {sorted(speaker_ids)}"
            )

            release_pipeline()
            force_memory_cleanup()

        # 5. セグメント加工
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
        print_step(f"   チャンク数: {len(chunks)}")

        segments_sent = split_segments_by_spacy_sentences(chunks)
        print_step(f"   spaCy文分割後セグメント数: {len(segments_sent)}")

        segments_src = merge_sentence_units(
            segments_sent,
            max_sentences=SPACY_UNIT_MAX_SENTENCES,
            merge_max_chars=SPACY_UNIT_MERGE_MAX_CHARS,
            max_gap_sec=SPACY_UNIT_MERGE_MAX_GAP_SEC,
        )
        print_step(f"   2文結合後セグメント数: {len(segments_src)}")

        save_segments_json_atomic(segments_src, seg_json_src)
        progress.set_artifact("segments_src_json", str(seg_json_src))
        progress.set_step("asr_done", True)
        progress.save()

    if not segments_src:
        raise PipelineError("segments_src が空です。")

    print_step(
        f"  検出言語: {detected_lang} ({get_lang_name(detected_lang)})"
    )
    engine_name = select_translation_engine(detected_lang, OUTPUT_LANG)
    print_step(
        f"  翻訳エンジン: "
        f"{'CAT-Translate' if engine_name == 'cat_translate' else 'TranslateGemma'} "
        f"({detected_lang} → {OUTPUT_LANG})"
    )

    # ===== 6. 話者リファレンス音声生成 =====
    if diarization is not None:
        print_step("6. OmniVoice 用の話者代表リファレンス音声を抽出")
        ref_cache.build_omnivoice_references(
            video_path, diarization, segments_src
        )
    else:
        _reload_cached_references(ref_cache, segments_src)

    print_step("6.5. OmniVoice セグメント単位リファレンス音声を切り出し")
    ref_cache.build_omnivoice_segment_references(video_path, segments_src)

    # ===== 7. 翻訳 =====
    print_step("7. 翻訳（再開対応）")
    if progress.step("translate_done") and seg_json_translated.exists():
        print_step("   翻訳: 既に完了（再開）")
        segments_translated = load_segments_json(seg_json_translated)
        if len(segments_translated) != len(segments_src):
            print_step("   注意: セグメント数不一致 → 翻訳やり直し")
            segments_translated = translate_segments_resumable(
                client, segments_src, seg_json_translated, progress,
                detected_lang=detected_lang,
            )
    else:
        segments_translated = translate_segments_resumable(
            client, segments_src, seg_json_translated, progress,
            detected_lang=detected_lang,
        )

    # 翻訳完了後、TTS 前に翻訳モデルを解放してメモリを確保する
    print_step("7.5. 翻訳モデルを解放（TTS 用メモリ確保）")
    release_all_translation_models()
    force_memory_cleanup()

    # ===== 8. TTS =====
    _run_tts_omnivoice(
        segments_translated, seg_audio_dir, work_dir, progress, ref_cache
    )

    force_memory_cleanup()

    tts_meta_path = work_dir / "tts_meta.json"
    tts_meta = load_tts_meta(tts_meta_path)

    # ===== 9. リタイム =====
    print_step("9. TTSに合わせて元動画の速度を区間ごとに変更")

    has_audio = ffprobe_has_audio(video_path)

    parts, new_total_dur = build_retime_parts(
        segments_translated, tts_meta, video_dur
    )
    progress.set_artifact("retimed_total_duration_sec", new_total_dur)
    progress.save()

    print_step(
        f"   リタイム後の総尺: {new_total_dur:.3f} sec（元: {video_dur:.3f} sec）"
    )
    print_step(f"   パート数: {len(parts)}")

    retime_dir = work_dir / "retime"
    video_chunk_dir = retime_dir / "video_chunks"
    dubbed_chunk_dir = retime_dir / "dubbed_chunks"
    orig_chunk_dir = retime_dir / "orig_chunks"
    ensure_dir(video_chunk_dir)
    ensure_dir(dubbed_chunk_dir)
    ensure_dir(orig_chunk_dir)

    print_step("   9-1. 映像チャンクを生成")
    video_chunks: list[Path] = []
    for i, part in enumerate(parts, start=1):
        out_ts = video_chunk_dir / f"v_{i:05d}.ts"
        video_chunks.append(out_ts)
        if out_ts.exists():
            continue
        encode_video_chunk_ts(
            video_path, out_ts,
            start=part.orig_start, end=part.orig_end, speed=part.speed,
        )

    video_ts = retime_dir / "video_retimed.ts"
    video_list = retime_dir / "video_concat.txt"
    if not video_ts.exists():
        print_step("   9-2. 映像チャンクを結合")
        concat_ts_files(video_chunks, video_ts, video_list)

    video_retimed_mp4 = retime_dir / "video_retimed.mp4"
    if not video_retimed_mp4.exists():
        print_step("   9-3. 映像TSをMP4へリマックス")
        remux_ts_to_mp4(video_ts, video_retimed_mp4)

    progress.set_artifact("retimed_video_mp4", str(video_retimed_mp4))
    progress.save()

    original_retimed_flac: Path | None = None
    if has_audio:
        print_step("   9-4. 元音声を同じ倍率でリタイムして結合")
        orig_chunks: list[Path] = []
        for i, part in enumerate(parts, start=1):
            out_flac = orig_chunk_dir / f"orig_{i:05d}.flac"
            orig_chunks.append(out_flac)
            if out_flac.exists():
                continue
            encode_original_audio_chunk_flac(
                video_path, out_flac,
                start=part.orig_start, end=part.orig_end, speed=part.speed,
            )

        original_retimed_flac = retime_dir / "original_retimed.flac"
        orig_list = retime_dir / "original_concat.txt"
        if not original_retimed_flac.exists():
            concat_audio_to_flac(
                orig_chunks, original_retimed_flac, orig_list
            )

        progress.set_artifact(
            "original_retimed_flac", str(original_retimed_flac)
        )
        progress.save()
    else:
        print_step("   9-4. 元動画に音声が無いため、元音声トラックは作りません。")

    print_step("   9-5. 吹き替えトラックを生成（無音 + TTS を順に結合）")
    dubbed_items: list[Path] = []
    for i, part in enumerate(parts, start=1):
        if part.kind in ("gap", "tail"):
            sil = dubbed_chunk_dir / f"dub_{i:05d}_sil.flac"
            dubbed_items.append(sil)
            if not sil.exists():
                create_silence_flac(sil, part.out_duration)
            continue

        meta = tts_meta.get(part.segno)
        if meta and Path(meta.flac_path).exists():
            dubbed_items.append(Path(meta.flac_path))
        else:
            sil = dubbed_chunk_dir / f"dub_{i:05d}_missing.flac"
            dubbed_items.append(sil)
            if not sil.exists():
                create_silence_flac(sil, part.out_duration)

    dubbed_full_flac = retime_dir / "dubbed_full.flac"
    dubbed_list = retime_dir / "dubbed_concat.txt"
    if not dubbed_full_flac.exists():
        concat_audio_to_flac(dubbed_items, dubbed_full_flac, dubbed_list)

    progress.set_artifact("dubbed_full_flac", str(dubbed_full_flac))
    progress.set_step("retime_done", True)
    progress.save()

    # ===== 10. mux =====
    if progress.step("mux_done") and out_path.exists():
        print_step(f"=== 完了（mux済み）: {out_path} ===")
    else:
        print_step("10. リタイム済み映像 + 吹き替え音声（+元音声薄く）を合成")
        mux_retimed_video_with_tracks(
            video_retimed_mp4,
            dubbed_full_flac,
            out_path,
            original_flac=original_retimed_flac,
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
    return not sanitize_text_for_tts(seg.text_tgt)


def _save_tts_meta_entry(tts_meta, tts_meta_path, segno, flac_path, duration_sec):
    tts_meta[segno] = TtsMeta(
        segno=segno, flac_path=flac_path, duration_sec=float(duration_sec),
    )
    save_tts_meta_atomic(tts_meta_path, tts_meta)


def _update_tts_progress(progress, done_count, total, tts_meta_path, save_artifact=False):
    progress.set_step("tts", {"done_count": done_count, "total": total})
    if save_artifact:
        progress.set_artifact("tts_meta_json", str(tts_meta_path))
    progress.save()


def _reuse_existing_tts_output(segno, out_flac, tts_meta, tts_meta_path):
    if segno in tts_meta and Path(tts_meta[segno].flac_path).exists():
        return "meta"

    if out_flac.exists():
        dur = ffprobe_duration_sec(out_flac)
        if dur > 0:
            _save_tts_meta_entry(
                tts_meta, tts_meta_path,
                segno=segno, flac_path=str(out_flac), duration_sec=dur,
            )
            return "flac"

    return None


def _run_tts_loop(
    segments_translated, seg_audio_dir, work_dir, progress,
    segment_label_builder, generator,
):
    tts_meta_path = work_dir / "tts_meta.json"
    tts_meta = load_tts_meta(tts_meta_path)

    total = len(segments_translated)
    tts_done = 0
    tts_failed = 0

    for segno, seg in enumerate(segments_translated, start=1):
        if _should_skip_tts_segment(seg):
            tts_done += 1
            continue

        out_audio_stub = seg_audio_dir / f"seg_{segno:05d}"
        out_flac = out_audio_stub.with_suffix(".flac")

        reused = _reuse_existing_tts_output(
            segno=segno, out_flac=out_flac,
            tts_meta=tts_meta, tts_meta_path=tts_meta_path,
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
            tts_meta, tts_meta_path,
            segno=segno, flac_path=meta0.flac_path, duration_sec=meta0.duration_sec,
        )

        tts_done += 1
        _update_tts_progress(
            progress, done_count=tts_done, total=total,
            tts_meta_path=tts_meta_path, save_artifact=True,
        )

    if tts_failed > 0:
        print_step(f"  TTS失敗セグメント数: {tts_failed}/{total}")

    _update_tts_progress(
        progress, done_count=total, total=total,
        tts_meta_path=tts_meta_path, save_artifact=True,
    )


def _run_tts_omnivoice(
    segments_translated, seg_audio_dir, work_dir, progress, ref_cache,
):
    from xlanguage_dubbing.omnivoice_tts import (
        generate_segment_tts_omnivoice,
        release_omnivoice_model,
    )

    print_step("8. OmniVoice でボイスクローン音声生成")

    try:
        def _build_segment_label(segno, total, seg):
            has_seg_ref = (
                ref_cache.get_omnivoice_segment_reference_path(segno) is not None
            )
            ref_type = "セグメント単位" if has_seg_ref else "話者代表"
            return (
                f"  TTS seg {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
                f"speaker={seg.speaker_id} ref={ref_type} (OmniVoice)"
            )

        def _generate(seg, out_audio_stub, segno):
            return generate_segment_tts_omnivoice(
                seg, out_audio_stub, ref_cache, segno=segno
            )

        _run_tts_loop(
            segments_translated=segments_translated,
            seg_audio_dir=seg_audio_dir,
            work_dir=work_dir,
            progress=progress,
            segment_label_builder=_build_segment_label,
            generator=_generate,
        )
    finally:
        release_omnivoice_model()
        force_memory_cleanup()


def _reload_cached_references(ref_cache, segments):
    speaker_ids = {s.speaker_id for s in segments if s.speaker_id}
    ref_cache.reload_speaker_references(speaker_ids)
    ref_cache.reload_omnivoice_segment_references(len(segments))
