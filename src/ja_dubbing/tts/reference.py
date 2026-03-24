#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
話者別リファレンス音声の抽出・キャッシュ管理。
話者代表リファレンスとセグメント単位リファレンスの両方をサポートする。

GPT-SoVITS 用の代表リファレンス（3〜10秒）も管理する。
GPT-SoVITS ではセグメント単位リファレンスは不要（声質のみ抽出されるため）。
T5Gemma-TTS 用の代表リファレンス（3〜15秒）とセグメント単位リファレンスを管理する。
T5Gemma-TTS のセグメント単位リファレンスでは、参照音声を ASR で再文字起こしし、
音声とテキストの一致を保証する。
"""

from __future__ import annotations

import base64
import gc
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydub import AudioSegment

from ja_dubbing.audio.ffmpeg import extract_audio_segment
from ja_dubbing.config import (
    GPTSOVITS_PROMPT_LANG,
    GPTSOVITS_REFERENCE_MAX_SEC,
    GPTSOVITS_REFERENCE_MIN_SEC,
    GPTSOVITS_REFERENCE_TARGET_SEC,
    MIOTTS_REFERENCE_MAX_SEC,
    T5GEMMA_REFERENCE_MAX_SEC,
    T5GEMMA_REFERENCE_MIN_SEC,
    T5GEMMA_REFERENCE_TARGET_SEC,
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
        self._refs: Dict[str, Path] = {}
        self._segment_refs: Dict[int, Path] = {}
        # GPT-SoVITS 用: 話者代表リファレンス（3〜10秒）
        self._gptsovits_refs: Dict[str, Path] = {}
        # GPT-SoVITS 用: 参照音声の書き起こしテキストと言語
        self._gptsovits_prompt_texts: Dict[str, str] = {}
        self._gptsovits_prompt_langs: Dict[str, str] = {}
        # T5Gemma-TTS 用: 話者代表リファレンス
        self._t5gemma_refs: Dict[str, Path] = {}
        # T5Gemma-TTS 用: 参照音声の書き起こしテキスト（英語文字起こし再利用）
        self._t5gemma_prompt_texts: Dict[str, str] = {}
        # T5Gemma-TTS 用: セグメント単位リファレンス音声
        self._t5gemma_segment_refs: Dict[int, Path] = {}
        # T5Gemma-TTS 用: セグメント単位リファレンスの書き起こしテキスト
        self._t5gemma_segment_prompt_texts: Dict[int, str] = {}
        ensure_dir(cache_dir)

    @property
    def cache_dir(self) -> Path:
        """キャッシュディレクトリのパスを返す。"""
        return self._cache_dir

    # ========================================
    # MioTTS 用: 話者代表リファレンス
    # ========================================

    def get_reference_base64(self, speaker_id: str) -> Optional[str]:
        """キャッシュ済み話者代表リファレンスをbase64で返す。"""
        path = self._refs.get(speaker_id)
        if path and path.exists():
            return base64.b64encode(path.read_bytes()).decode("ascii")
        return None

    def get_segment_reference_base64(self, segno: int) -> Optional[str]:
        """セグメント単位リファレンスをbase64で返す。"""
        path = self._segment_refs.get(segno)
        if path and path.exists():
            return base64.b64encode(path.read_bytes()).decode("ascii")
        return None

    def get_segment_reference_path(self, segno: int) -> Optional[Path]:
        """セグメント単位リファレンスのパスを返す。"""
        return self._segment_refs.get(segno)

    def reload_speaker_references(self, speaker_ids: Set[str]) -> None:
        """再開時にキャッシュ済み話者代表リファレンスを検出してロードする。"""
        for speaker_id in speaker_ids:
            wav_path = self._cache_dir / f"ref_{speaker_id}.wav"
            if wav_path.exists():
                self._refs[speaker_id] = wav_path
            # GPT-SoVITS 用リファレンスも検出
            gs_wav = self._cache_dir / f"gsref_{speaker_id}.wav"
            if gs_wav.exists():
                self._gptsovits_refs[speaker_id] = gs_wav
            # T5Gemma-TTS 用リファレンスも検出
            t5_wav = self._cache_dir / f"t5ref_{speaker_id}.wav"
            if t5_wav.exists():
                self._t5gemma_refs[speaker_id] = t5_wav
        # GPT-SoVITS プロンプトメタデータを読み込む
        self._load_gptsovits_prompt_meta()
        # T5Gemma-TTS プロンプトメタデータを読み込む
        self._load_t5gemma_prompt_meta()
        # T5Gemma-TTS セグメント単位メタデータを読み込む
        self._load_t5gemma_segment_meta()

    def build_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
    ) -> None:
        """全話者の代表リファレンス音声を生成する（MioTTS 用: 最大20秒）。"""
        speakers: Dict[str, List[DiarizationSegment]] = {}
        for dia in diarization:
            speakers.setdefault(dia.speaker, []).append(dia)

        for speaker_id, dia_segments in speakers.items():
            out_wav = self._cache_dir / f"ref_{speaker_id}.wav"
            if out_wav.exists():
                self._refs[speaker_id] = out_wav
                continue

            # 発話区間を長い順にソートし、合計が上限に収まるまで抽出
            segments_sorted = sorted(
                dia_segments, key=lambda s: s.end - s.start, reverse=True
            )
            collected: List[AudioSegment] = []
            total_sec = 0.0

            for seg in segments_sorted:
                dur = seg.end - seg.start
                if dur < 0.5:
                    continue
                if total_sec + dur > MIOTTS_REFERENCE_MAX_SEC:
                    remain = MIOTTS_REFERENCE_MAX_SEC - total_sec
                    if remain < 0.5:
                        break
                    dur = remain

                tmp_wav = self._cache_dir / f"tmp_{speaker_id}_{seg.start:.3f}.wav"
                extract_audio_segment(
                    video_path, tmp_wav,
                    start=seg.start, end=seg.start + dur,
                    sample_rate=44100, channels=1,
                )
                try:
                    audio = AudioSegment.from_wav(str(tmp_wav))
                    collected.append(audio)
                    total_sec += dur
                finally:
                    tmp_wav.unlink(missing_ok=True)

                if total_sec >= MIOTTS_REFERENCE_MAX_SEC:
                    break

            if collected:
                combined = collected[0]
                for a in collected[1:]:
                    combined += a
                combined.export(str(out_wav), format="wav")
                self._refs[speaker_id] = out_wav
                print_step(
                    f"  リファレンス生成: {speaker_id} ({total_sec:.1f}s)"
                )
                del combined
            else:
                print_step(
                    f"  警告: {speaker_id} のリファレンス音声を生成できません"
                )

            del collected
            gc.collect()

    def build_segment_references(
        self,
        video_path: Path,
        segments: List[Segment],
    ) -> None:
        """各セグメントの元英語音声をリファレンスとして抽出する（MioTTS 用）。"""
        seg_ref_dir = self._cache_dir / "segment_refs"
        ensure_dir(seg_ref_dir)

        for segno, seg in enumerate(segments, start=1):
            out_wav = seg_ref_dir / f"seg_ref_{segno:05d}.wav"

            if out_wav.exists():
                self._segment_refs[segno] = out_wav
                continue

            dur = seg.end - seg.start
            if dur < 0.3:
                continue

            effective_end = seg.start + min(dur, MIOTTS_REFERENCE_MAX_SEC)

            extract_audio_segment(
                video_path, out_wav,
                start=seg.start, end=effective_end,
                sample_rate=44100, channels=1,
            )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._segment_refs[segno] = out_wav
            else:
                out_wav.unlink(missing_ok=True)

        generated = len(self._segment_refs)
        print_step(
            f"  セグメント単位リファレンス生成: {generated}/{len(segments)} 件"
        )

    def reload_segment_references(self, total_segments: int) -> None:
        """再開時にキャッシュ済みセグメント単位リファレンスを検出してロードする。"""
        seg_ref_dir = self._cache_dir / "segment_refs"
        if not seg_ref_dir.exists():
            return
        for segno in range(1, total_segments + 1):
            wav_path = seg_ref_dir / f"seg_ref_{segno:05d}.wav"
            if wav_path.exists():
                self._segment_refs[segno] = wav_path

    # ========================================
    # GPT-SoVITS 用: 話者代表リファレンス（3〜10秒）
    # ========================================

    def get_gptsovits_reference_path(self, speaker_id: str) -> Optional[Path]:
        """GPT-SoVITS 用の代表リファレンス音声の絶対パスを返す。"""
        path = self._gptsovits_refs.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_gptsovits_prompt_text(self, speaker_id: str) -> str:
        """GPT-SoVITS 参照音声の書き起こしテキストを返す。"""
        return self._gptsovits_prompt_texts.get(speaker_id, "")

    def get_gptsovits_prompt_lang(self, speaker_id: str) -> str:
        """GPT-SoVITS 参照音声の言語を返す。"""
        return self._gptsovits_prompt_langs.get(speaker_id, GPTSOVITS_PROMPT_LANG)

    def build_gptsovits_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
    ) -> None:
        """
        GPT-SoVITS 用の話者代表リファレンス音声を生成する。

        参照音声を切り出した後、ASR エンジンで直接文字起こしして
        prompt_text を生成する。これにより参照音声とテキストの
        不一致を防ぐ。
        """
        from ja_dubbing.asr import transcribe_reference_audio

        speakers: Dict[str, List[DiarizationSegment]] = {}
        for dia in diarization:
            speakers.setdefault(dia.speaker, []).append(dia)

        prompt_meta: Dict[str, Dict[str, str]] = {}

        for speaker_id, dia_segments in speakers.items():
            out_wav = self._cache_dir / f"gsref_{speaker_id}.wav"
            if out_wav.exists():
                self._gptsovits_refs[speaker_id] = out_wav
                # メタデータが既にロード済みならスキップ
                if speaker_id in self._gptsovits_prompt_texts:
                    continue

            # 目標長（5秒）に最も近い発話区間を選ぶ
            best_seg = _select_best_reference_segment(
                dia_segments,
                min_sec=GPTSOVITS_REFERENCE_MIN_SEC,
                max_sec=GPTSOVITS_REFERENCE_MAX_SEC,
                target_sec=GPTSOVITS_REFERENCE_TARGET_SEC,
            )
            if best_seg is None:
                print_step(
                    f"  警告: GPT-SoVITS 用リファレンスなし: {speaker_id}"
                )
                continue

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=best_seg.start, end=best_seg.end,
                    sample_rate=44100, channels=1,
                )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._gptsovits_refs[speaker_id] = out_wav
                ref_dur = best_seg.end - best_seg.start

                # 参照音声を ASR エンジンで直接文字起こしする
                prompt_text = transcribe_reference_audio(
                    out_wav, language=GPTSOVITS_PROMPT_LANG,
                )
                if not prompt_text:
                    print_step(
                        f"    警告: {speaker_id} の参照音声の文字起こしが空です"
                    )

                self._gptsovits_prompt_texts[speaker_id] = prompt_text
                self._gptsovits_prompt_langs[speaker_id] = GPTSOVITS_PROMPT_LANG

                prompt_meta[speaker_id] = {
                    "prompt_text": prompt_text,
                    "prompt_lang": GPTSOVITS_PROMPT_LANG,
                    "ref_start": best_seg.start,
                    "ref_end": best_seg.end,
                }

                print_step(
                    f"  GPT-SoVITS リファレンス生成: {speaker_id} "
                    f"({ref_dur:.1f}s) text='{prompt_text[:80]}'"
                )
            else:
                out_wav.unlink(missing_ok=True)

        # プロンプトメタデータを保存する（再開用）
        self._save_gptsovits_prompt_meta(prompt_meta)

    # ========================================
    # T5Gemma-TTS 用: 話者代表リファレンス（3〜15秒）
    # ========================================

    def get_t5gemma_reference_path(self, speaker_id: str) -> Optional[Path]:
        """T5Gemma-TTS 用の代表リファレンス音声の絶対パスを返す。"""
        path = self._t5gemma_refs.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_t5gemma_prompt_text(self, speaker_id: str) -> str:
        """T5Gemma-TTS 参照音声の英語テキストを返す。"""
        return self._t5gemma_prompt_texts.get(speaker_id, "")

    def get_t5gemma_segment_reference_path(self, segno: int) -> Optional[Path]:
        """T5Gemma-TTS セグメント単位リファレンス音声のパスを返す。"""
        path = self._t5gemma_segment_refs.get(segno)
        if path and path.exists():
            return path.resolve()
        return None

    def get_t5gemma_segment_prompt_text(self, segno: int) -> str:
        """T5Gemma-TTS セグメント単位リファレンスの書き起こしテキストを返す。"""
        return self._t5gemma_segment_prompt_texts.get(segno, "")

    def build_t5gemma_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
        segments: List[Segment],
    ) -> None:
        """T5Gemma-TTS 用の話者代表リファレンス音声を生成する。"""
        from ja_dubbing.asr import transcribe_reference_audio

        speakers: Dict[str, List[DiarizationSegment]] = {}
        for dia in diarization:
            speakers.setdefault(dia.speaker, []).append(dia)

        prompt_meta: Dict[str, Dict[str, str | float]] = {}

        for speaker_id, dia_segments in speakers.items():
            out_wav = self._cache_dir / f"t5ref_{speaker_id}.wav"
            if out_wav.exists():
                self._t5gemma_refs[speaker_id] = out_wav
                if speaker_id in self._t5gemma_prompt_texts:
                    continue

            best_seg = _select_best_reference_segment(
                dia_segments,
                min_sec=T5GEMMA_REFERENCE_MIN_SEC,
                max_sec=T5GEMMA_REFERENCE_MAX_SEC,
                target_sec=T5GEMMA_REFERENCE_TARGET_SEC,
            )
            if best_seg is None:
                print_step(
                    f"  警告: T5Gemma-TTS 用リファレンスなし: {speaker_id}"
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
                self._t5gemma_refs[speaker_id] = out_wav
                ref_dur = best_seg.end - best_seg.start

                # 参照音声を ASR で直接文字起こしする（GPT-SoVITS と同様）
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

                self._t5gemma_prompt_texts[speaker_id] = prompt_text
                prompt_meta[speaker_id] = {
                    "prompt_text": prompt_text,
                    "ref_start": best_seg.start,
                    "ref_end": best_seg.end,
                }
                preview = prompt_text[:80] if prompt_text else "(empty)"
                print_step(
                    f"  T5Gemma-TTS リファレンス生成: {speaker_id} "
                    f"({ref_dur:.1f}s) text='{preview}'"
                )
            else:
                out_wav.unlink(missing_ok=True)

        self._save_t5gemma_prompt_meta(prompt_meta)

    def build_t5gemma_segment_references(
        self,
        video_path: Path,
        segments: List[Segment],
    ) -> None:
        """
        T5Gemma-TTS 用のセグメント単位リファレンス音声を生成する。

        各セグメントの元英語音声を切り出し、ASR エンジンで直接文字起こしして
        参照テキストを生成する。これにより参照音声とテキストの一致を保証する。
        """
        from ja_dubbing.asr import transcribe_reference_audio

        seg_ref_dir = self._cache_dir / "t5gemma_segment_refs"
        ensure_dir(seg_ref_dir)

        segment_meta: Dict[str, Dict[str, str | float]] = {}

        for segno, seg in enumerate(segments, start=1):
            out_wav = seg_ref_dir / f"t5seg_ref_{segno:05d}.wav"

            # 既にキャッシュがあり、メタデータも読み込み済みならスキップ
            if out_wav.exists() and segno in self._t5gemma_segment_prompt_texts:
                self._t5gemma_segment_refs[segno] = out_wav
                continue

            dur = seg.end - seg.start
            if dur < 0.3:
                continue

            # T5Gemma の参照音声上限でクリップ
            effective_end = seg.start + min(dur, T5GEMMA_REFERENCE_MAX_SEC)

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=seg.start, end=effective_end,
                    sample_rate=44100, channels=1,
                )

            if not out_wav.exists() or out_wav.stat().st_size <= 100:
                out_wav.unlink(missing_ok=True)
                continue

            self._t5gemma_segment_refs[segno] = out_wav

            # 切り出した参照音声を ASR で直接文字起こしする
            prompt_text = transcribe_reference_audio(
                out_wav, language="en",
            )
            if not prompt_text:
                # ASR 失敗時は元のセグメントの英語テキストをフォールバック
                prompt_text = normalize_spaces(seg.text_en)

            self._t5gemma_segment_prompt_texts[segno] = prompt_text

            segment_meta[str(segno)] = {
                "prompt_text": prompt_text,
                "ref_start": seg.start,
                "ref_end": effective_end,
            }

        # メタデータを保存する（再開用）
        self._save_t5gemma_segment_meta(segment_meta)

        generated = len(self._t5gemma_segment_refs)
        print_step(
            f"  T5Gemma セグメント単位リファレンス生成: "
            f"{generated}/{len(segments)} 件"
        )

    def reload_t5gemma_segment_references(self, total_segments: int) -> None:
        """再開時にキャッシュ済み T5Gemma セグメント単位リファレンスを検出してロードする。"""
        seg_ref_dir = self._cache_dir / "t5gemma_segment_refs"
        if not seg_ref_dir.exists():
            return
        for segno in range(1, total_segments + 1):
            wav_path = seg_ref_dir / f"t5seg_ref_{segno:05d}.wav"
            if wav_path.exists():
                self._t5gemma_segment_refs[segno] = wav_path
        # メタデータも読み込む
        self._load_t5gemma_segment_meta()

    def _save_gptsovits_prompt_meta(
        self, meta: Dict[str, Dict[str, str]]
    ) -> None:
        """GPT-SoVITS プロンプトメタデータを保存する。"""
        meta_path = self._cache_dir / "gptsovits_prompt_meta.json"
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_gptsovits_prompt_meta(self) -> None:
        """GPT-SoVITS プロンプトメタデータを読み込む。"""
        meta_path = self._cache_dir / "gptsovits_prompt_meta.json"
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for speaker_id, info in obj.items():
            if not isinstance(info, dict):
                continue
            self._gptsovits_prompt_texts[speaker_id] = info.get(
                "prompt_text", ""
            )
            self._gptsovits_prompt_langs[speaker_id] = info.get(
                "prompt_lang", GPTSOVITS_PROMPT_LANG
            )

    def _save_t5gemma_prompt_meta(
        self, meta: Dict[str, Dict[str, str | float]]
    ) -> None:
        """T5Gemma-TTS プロンプトメタデータを保存する。"""
        meta_path = self._cache_dir / "t5gemma_prompt_meta.json"
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_t5gemma_prompt_meta(self) -> None:
        """T5Gemma-TTS プロンプトメタデータを読み込む。"""
        meta_path = self._cache_dir / "t5gemma_prompt_meta.json"
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for speaker_id, info in obj.items():
            if not isinstance(info, dict):
                continue
            self._t5gemma_prompt_texts[speaker_id] = str(
                info.get("prompt_text", "") or ""
            )

    def _save_t5gemma_segment_meta(
        self, meta: Dict[str, Dict[str, str | float]]
    ) -> None:
        """T5Gemma-TTS セグメント単位メタデータを保存する。"""
        meta_path = self._cache_dir / "t5gemma_segment_meta.json"
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_t5gemma_segment_meta(self) -> None:
        """T5Gemma-TTS セグメント単位メタデータを読み込む。"""
        meta_path = self._cache_dir / "t5gemma_segment_meta.json"
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
            self._t5gemma_segment_prompt_texts[segno] = str(
                info.get("prompt_text", "") or ""
            )

    def clear(self) -> None:
        """キャッシュをクリアしてメモリを解放する。"""
        self._refs.clear()
        self._segment_refs.clear()
        self._gptsovits_refs.clear()
        self._gptsovits_prompt_texts.clear()
        self._gptsovits_prompt_langs.clear()
        self._t5gemma_refs.clear()
        self._t5gemma_prompt_texts.clear()
        self._t5gemma_segment_refs.clear()
        self._t5gemma_segment_prompt_texts.clear()
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
        overlap = min(reference_seg.end, seg.end) - max(reference_seg.start, seg.start)
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
