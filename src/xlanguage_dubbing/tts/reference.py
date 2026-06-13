#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
話者別リファレンス音声の抽出・キャッシュ管理。
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Set

from xlanguage_dubbing.audio.ffmpeg import extract_audio_segment
from xlanguage_dubbing.config import (
    OMNIVOICE_REFERENCE_MAX_SEC,
    OMNIVOICE_REFERENCE_MIN_SEC,
    OMNIVOICE_REFERENCE_TARGET_SEC,
    TTS_ENGINE,
)
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.utils import (
    atomic_write_json,
    ensure_dir,
    load_json_if_exists,
    normalize_spaces,
    print_step,
)


class SpeakerReferenceCache:
    """話者ごとのリファレンス音声をキャッシュ管理する。"""

    def __init__(self, cache_dir: Path, tts_engine: str | None = None) -> None:
        self._cache_dir = cache_dir
        self._tts_engine = _normalize_reference_engine(tts_engine or TTS_ENGINE)
        self._omnivoice_refs: Dict[str, Path] = {}
        self._omnivoice_prompt_texts: Dict[str, str] = {}
        self._omnivoice_segment_refs: Dict[int, Path] = {}
        self._omnivoice_segment_prompt_texts: Dict[int, str] = {}
        ensure_dir(cache_dir)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def segment_reference_dir(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_segment_refs"

    def get_omnivoice_reference_path(self, speaker_id: str) -> Optional[Path]:
        path = self._omnivoice_refs.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_prompt_text(self, speaker_id: str) -> str:
        return self._omnivoice_prompt_texts.get(speaker_id, "")

    def get_omnivoice_segment_reference_path(self, segno: int) -> Optional[Path]:
        path = self._omnivoice_segment_refs.get(segno)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_segment_prompt_text(self, segno: int) -> str:
        return self._omnivoice_segment_prompt_texts.get(segno, "")

    def reload_speaker_references(self, speaker_ids: Set[str]) -> None:
        for speaker_id in speaker_ids:
            ov_wav = self._cache_dir / f"ovref_{speaker_id}.wav"
            if ov_wav.exists():
                self._omnivoice_refs[speaker_id] = ov_wav
        self._load_omnivoice_prompt_meta()
        self._load_omnivoice_segment_meta()

    def build_omnivoice_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
        segments: List[Segment],
    ) -> None:
        from xlanguage_dubbing.asr import transcribe_reference_audio

        speakers: Dict[str, List[DiarizationSegment]] = {}
        for dia in diarization:
            speakers.setdefault(dia.speaker, []).append(dia)

        prompt_meta: Dict[str, Dict[str, str | float]] = {}

        for speaker_id, dia_segments in speakers.items():
            out_wav = self._cache_dir / f"ovref_{speaker_id}.wav"
            if out_wav.exists():
                self._omnivoice_refs[speaker_id] = out_wav
                if speaker_id in self._omnivoice_prompt_texts:
                    continue

            best_seg = _select_best_reference_segment(
                dia_segments,
                min_sec=OMNIVOICE_REFERENCE_MIN_SEC,
                max_sec=OMNIVOICE_REFERENCE_MAX_SEC,
                target_sec=OMNIVOICE_REFERENCE_TARGET_SEC,
            )
            if best_seg is None:
                print_step(f"  警告: リファレンスなし: {speaker_id}")
                continue

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=best_seg.start, end=best_seg.end,
                    sample_rate=44100, channels=1,
                )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._omnivoice_refs[speaker_id] = out_wav
                ref_dur = best_seg.end - best_seg.start

                prompt_text = transcribe_reference_audio(out_wav, language="")
                if not prompt_text:
                    prompt_text = _collect_reference_prompt_text(
                        best_seg, segments
                    )

                self._omnivoice_prompt_texts[speaker_id] = prompt_text
                prompt_meta[speaker_id] = {
                    "prompt_text": prompt_text,
                    "ref_start": best_seg.start,
                    "ref_end": best_seg.end,
                }
                preview = prompt_text[:80] if prompt_text else "(empty)"
                print_step(
                    f"  {_reference_engine_label(self._tts_engine)} "
                    f"リファレンス生成: {speaker_id} "
                    f"({ref_dur:.1f}s) text='{preview}'"
                )
            else:
                out_wav.unlink(missing_ok=True)

        self._save_omnivoice_prompt_meta(prompt_meta)

    def build_omnivoice_segment_references(
        self,
        video_path: Path,
        segments: List[Segment],
    ) -> None:
        seg_ref_dir = self.segment_reference_dir
        ensure_dir(seg_ref_dir)

        segment_meta: Dict[str, Dict[str, str | float]] = {}

        for segno, seg in enumerate(segments, start=1):
            out_wav = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"

            if (
                out_wav.exists()
                and segno in self._omnivoice_segment_prompt_texts
            ):
                self._omnivoice_segment_refs[segno] = out_wav
                continue

            dur = seg.end - seg.start
            if dur < 0.3:
                continue

            effective_end = seg.start + min(dur, OMNIVOICE_REFERENCE_MAX_SEC)

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=seg.start, end=effective_end,
                    sample_rate=44100, channels=1,
                )

            if not out_wav.exists() or out_wav.stat().st_size <= 100:
                out_wav.unlink(missing_ok=True)
                continue

            self._omnivoice_segment_refs[segno] = out_wav

            prompt_text = normalize_spaces(seg.text_src)
            self._omnivoice_segment_prompt_texts[segno] = prompt_text

            segment_meta[str(segno)] = {
                "prompt_text": prompt_text,
                "ref_start": seg.start,
                "ref_end": effective_end,
            }

        self._save_omnivoice_segment_meta(segment_meta)

        generated = len(self._omnivoice_segment_refs)
        print_step(
            f"  {_reference_engine_label(self._tts_engine)} "
            "セグメント単位リファレンス生成: "
            f"{generated}/{len(segments)} 件"
        )

    def reload_omnivoice_segment_references(self, total_segments: int) -> None:
        seg_ref_dir = self.segment_reference_dir
        if not seg_ref_dir.exists():
            return
        for segno in range(1, total_segments + 1):
            wav_path = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"
            if wav_path.exists():
                self._omnivoice_segment_refs[segno] = wav_path
        self._load_omnivoice_segment_meta()

    def _save_omnivoice_prompt_meta(self, meta):
        meta_path = self._prompt_meta_path()
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_prompt_meta(self):
        meta_path = self._prompt_meta_path()
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for speaker_id, info in obj.items():
            if not isinstance(info, dict):
                continue
            self._omnivoice_prompt_texts[speaker_id] = str(
                info.get("prompt_text", "") or ""
            )

    def _prompt_meta_path(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_prompt_meta.json"

    def _save_omnivoice_segment_meta(self, meta):
        meta_path = self._segment_meta_path()
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_segment_meta(self):
        meta_path = self._segment_meta_path()
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for segno_str, info in obj.items():
            if not isinstance(info, dict):
                continue
            try:
                segno = int(segno_str)
            except (ValueError, TypeError):
                continue
            self._omnivoice_segment_prompt_texts[segno] = str(
                info.get("prompt_text", "") or ""
            )

    def _segment_meta_path(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_segment_meta.json"

    def clear(self):
        self._omnivoice_refs.clear()
        self._omnivoice_prompt_texts.clear()
        self._omnivoice_segment_refs.clear()
        self._omnivoice_segment_prompt_texts.clear()
        gc.collect()


def _select_best_reference_segment(dia_segments, min_sec, max_sec, target_sec):
    candidates = []
    for seg in dia_segments:
        dur = seg.end - seg.start
        if dur < min_sec:
            continue
        candidates.append(seg)

    if not candidates:
        if dia_segments:
            longest = max(dia_segments, key=lambda s: s.end - s.start)
            if longest.end - longest.start >= 1.0:
                return longest
        return None

    def score(seg):
        dur = seg.end - seg.start
        if dur > max_sec:
            dur = max_sec
        return abs(dur - target_sec)

    best = min(candidates, key=score)

    dur = best.end - best.start
    if dur > max_sec:
        return DiarizationSegment(
            start=best.start,
            end=best.start + max_sec,
            speaker=best.speaker,
        )
    return best


def _collect_reference_prompt_text(reference_seg, segments):
    overlapped = []
    for seg in segments:
        overlap = min(reference_seg.end, seg.end) - max(
            reference_seg.start, seg.start
        )
        if overlap > 0:
            overlapped.append((overlap, seg))

    if overlapped:
        overlapped.sort(key=lambda item: item[1].start)
        texts = [normalize_spaces(seg.text_src) for _, seg in overlapped]
        return normalize_spaces(" ".join(t for t in texts if t))

    if not segments:
        return ""

    ref_center = (reference_seg.start + reference_seg.end) / 2.0
    nearest = min(
        segments,
        key=lambda seg: abs(((seg.start + seg.end) / 2.0) - ref_center),
    )
    return normalize_spaces(nearest.text_src)


def _normalize_reference_engine(tts_engine: str) -> str:
    normalized = tts_engine.strip().lower()
    if normalized == "voxcpm2":
        return "voxcpm2"
    if normalized in {"irodori", "irodori-tts", "irodori_tts"}:
        return "irodori"
    return "omnivoice"


def _reference_engine_label(tts_engine: str) -> str:
    if tts_engine == "voxcpm2":
        return "VoxCPM2"
    if tts_engine == "irodori":
        return "Irodori"
    return "OmniVoice"
