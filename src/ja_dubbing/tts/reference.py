#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
話者別リファレンス音声の抽出・キャッシュ管理。
OmniVoice 用の話者代表リファレンス（3〜15秒）とセグメント単位リファレンスを管理する。
Kokoro TTS はボイスクローン非対応のためリファレンス不要。
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydub import AudioSegment

from ja_dubbing.audio.ffmpeg import extract_audio_segment
from ja_dubbing.config import (
    OMNIVOICE_REFERENCE_MAX_SEC,
    OMNIVOICE_REFERENCE_MIN_SEC,
    OMNIVOICE_REFERENCE_TARGET_SEC,
)
from ja_dubbing.core.models import DiarizationSegment, Segment
from ja_dubbing.utils import (
    atomic_write_json,
    ensure_dir,
    load_json_if_exists,
    normalize_spaces,
    print_step,
)


class SpeakerReferenceCache:
    """話者ごとのリファレンス音声をキャッシュ管理する。"""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        # OmniVoice 用: 話者代表リファレンス
        self._omnivoice_refs: Dict[str, Path] = {}
        # OmniVoice 用: 参照音声の書き起こしテキスト
        self._omnivoice_prompt_texts: Dict[str, str] = {}
        # OmniVoice 用: セグメント単位リファレンス音声
        self._omnivoice_segment_refs: Dict[int, Path] = {}
        # OmniVoice 用: セグメント単位リファレンスの書き起こしテキスト
        self._omnivoice_segment_prompt_texts: Dict[int, str] = {}
        ensure_dir(cache_dir)

    @property
    def cache_dir(self) -> Path:
        """キャッシュディレクトリのパスを返す。"""
        return self._cache_dir

    # ========================================
    # OmniVoice 用: 話者代表リファレンス
    # ========================================

    def get_omnivoice_reference_path(self, speaker_id: str) -> Optional[Path]:
        """OmniVoice 用の代表リファレンス音声の絶対パスを返す。"""
        path = self._omnivoice_refs.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_prompt_text(self, speaker_id: str) -> str:
        """OmniVoice 参照音声の英語テキストを返す。"""
        return self._omnivoice_prompt_texts.get(speaker_id, "")

    def get_omnivoice_segment_reference_path(self, segno: int) -> Optional[Path]:
        """OmniVoice セグメント単位リファレンス音声のパスを返す。"""
        path = self._omnivoice_segment_refs.get(segno)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_segment_prompt_text(self, segno: int) -> str:
        """OmniVoice セグメント単位リファレンスの書き起こしテキストを返す。"""
        return self._omnivoice_segment_prompt_texts.get(segno, "")

    def reload_speaker_references(self, speaker_ids: Set[str]) -> None:
        """再開時にキャッシュ済み話者代表リファレンスを検出してロードする。"""
        for speaker_id in speaker_ids:
            ov_wav = self._cache_dir / f"ovref_{speaker_id}.wav"
            if ov_wav.exists():
                self._omnivoice_refs[speaker_id] = ov_wav
        # OmniVoice プロンプトメタデータを読み込む
        self._load_omnivoice_prompt_meta()
        # OmniVoice セグメント単位メタデータを読み込む
        self._load_omnivoice_segment_meta()

    def build_omnivoice_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
        segments: List[Segment],
    ) -> None:
        """OmniVoice 用の話者代表リファレンス音声を生成する。"""
        from ja_dubbing.asr import transcribe_reference_audio

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
                print_step(
                    f"  警告: OmniVoice 用リファレンスなし: {speaker_id}"
                )
                continue

            if not out_wav.exists():
                extract_audio_segment(
                    video_path,
                    out_wav,
                    start=best_seg.start,
                    end=best_seg.end,
                    sample_rate=44100,
                    channels=1,
                )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._omnivoice_refs[speaker_id] = out_wav
                ref_dur = best_seg.end - best_seg.start

                # 参照音声を ASR で直接文字起こしする
                prompt_text = transcribe_reference_audio(
                    out_wav, language="en",
                )
                if not prompt_text:
                    # ASR 失敗時はセグメントテキストからフォールバック
                    prompt_text = _collect_reference_prompt_text(
                        best_seg, segments
                    )
                    if prompt_text:
                        print_step(
                            f"    ASR 失敗: {speaker_id} → "
                            "セグメントテキストからフォールバック"
                        )

                self._omnivoice_prompt_texts[speaker_id] = prompt_text
                prompt_meta[speaker_id] = {
                    "prompt_text": prompt_text,
                    "ref_start": best_seg.start,
                    "ref_end": best_seg.end,
                }
                preview = prompt_text[:80] if prompt_text else "(empty)"
                print_step(
                    f"  OmniVoice リファレンス生成: {speaker_id} "
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
        """
        OmniVoice 用のセグメント単位リファレンス音声を生成する。

        各セグメントの元英語音声を切り出し、ASR エンジンで直接文字起こしして
        参照テキストを生成する。これにより参照音声とテキストの一致を保証する。
        """
        from ja_dubbing.asr import transcribe_reference_audio

        seg_ref_dir = self._cache_dir / "omnivoice_segment_refs"
        ensure_dir(seg_ref_dir)

        segment_meta: Dict[str, Dict[str, str | float]] = {}

        for segno, seg in enumerate(segments, start=1):
            out_wav = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"

            # 既にキャッシュがあり、メタデータも読み込み済みならスキップ
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

            # 切り出した参照音声を ASR で直接文字起こしする
            prompt_text = transcribe_reference_audio(
                out_wav, language="en",
            )
            if not prompt_text:
                # ASR 失敗時は元のセグメントの英語テキストをフォールバック
                prompt_text = normalize_spaces(seg.text_en)

            self._omnivoice_segment_prompt_texts[segno] = prompt_text

            segment_meta[str(segno)] = {
                "prompt_text": prompt_text,
                "ref_start": seg.start,
                "ref_end": effective_end,
            }

        # メタデータを保存する（再開用）
        self._save_omnivoice_segment_meta(segment_meta)

        generated = len(self._omnivoice_segment_refs)
        print_step(
            f"  OmniVoice セグメント単位リファレンス生成: "
            f"{generated}/{len(segments)} 件"
        )

    def reload_omnivoice_segment_references(self, total_segments: int) -> None:
        """再開時にキャッシュ済み OmniVoice セグメント単位リファレンスを検出する。"""
        seg_ref_dir = self._cache_dir / "omnivoice_segment_refs"
        if not seg_ref_dir.exists():
            return
        for segno in range(1, total_segments + 1):
            wav_path = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"
            if wav_path.exists():
                self._omnivoice_segment_refs[segno] = wav_path
        self._load_omnivoice_segment_meta()

    def _save_omnivoice_prompt_meta(
        self, meta: Dict[str, Dict[str, str | float]]
    ) -> None:
        """OmniVoice プロンプトメタデータを保存する。"""
        meta_path = self._cache_dir / "omnivoice_prompt_meta.json"
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_prompt_meta(self) -> None:
        """OmniVoice プロンプトメタデータを読み込む。"""
        meta_path = self._cache_dir / "omnivoice_prompt_meta.json"
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for speaker_id, info in obj.items():
            if not isinstance(info, dict):
                continue
            self._omnivoice_prompt_texts[speaker_id] = str(
                info.get("prompt_text", "") or ""
            )

    def _save_omnivoice_segment_meta(
        self, meta: Dict[str, Dict[str, str | float]]
    ) -> None:
        """OmniVoice セグメント単位メタデータを保存する。"""
        meta_path = self._cache_dir / "omnivoice_segment_meta.json"
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_segment_meta(self) -> None:
        """OmniVoice セグメント単位メタデータを読み込む。"""
        meta_path = self._cache_dir / "omnivoice_segment_meta.json"
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

    def clear(self) -> None:
        """キャッシュをクリアしてメモリを解放する。"""
        self._omnivoice_refs.clear()
        self._omnivoice_prompt_texts.clear()
        self._omnivoice_segment_refs.clear()
        self._omnivoice_segment_prompt_texts.clear()
        gc.collect()


def _select_best_reference_segment(
    dia_segments: List[DiarizationSegment],
    min_sec: float,
    max_sec: float,
    target_sec: float,
) -> Optional[DiarizationSegment]:
    """目標秒数に最も近いリファレンス区間を選ぶ。"""
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

    def score(seg: DiarizationSegment) -> float:
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


def _collect_reference_prompt_text(
    reference_seg: DiarizationSegment,
    segments: List[Segment],
) -> str:
    """参照音声区間と重なる英語文字起こしを結合する。"""
    overlapped: List[tuple[float, Segment]] = []
    for seg in segments:
        overlap = min(reference_seg.end, seg.end) - max(
            reference_seg.start, seg.start
        )
        if overlap > 0:
            overlapped.append((overlap, seg))

    if overlapped:
        overlapped.sort(key=lambda item: item[1].start)
        texts = [normalize_spaces(seg.text_en) for _, seg in overlapped]
        return normalize_spaces(" ".join(t for t in texts if t))

    if not segments:
        return ""

    ref_center = (reference_seg.start + reference_seg.end) / 2.0
    nearest = min(
        segments,
        key=lambda seg: abs(((seg.start + seg.end) / 2.0) - ref_center),
    )
    return normalize_spaces(nearest.text_en)
