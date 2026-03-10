#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
話者別リファレンス音声の抽出・キャッシュ管理。
話者代表リファレンスとセグメント単位リファレンスの両方をサポートする。
"""

from __future__ import annotations

import base64
import gc
from pathlib import Path
from typing import Dict, List, Optional, Set

from pydub import AudioSegment

from ja_dubbing.audio.ffmpeg import extract_audio_segment
from ja_dubbing.config import MIOTTS_REFERENCE_MAX_SEC
from ja_dubbing.core.models import DiarizationSegment, Segment
from ja_dubbing.utils import ensure_dir, print_step


class SpeakerReferenceCache:
    """話者ごとのリファレンス音声をキャッシュ管理する。"""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._refs: Dict[str, Path] = {}
        self._segment_refs: Dict[int, Path] = {}
        ensure_dir(cache_dir)

    @property
    def cache_dir(self) -> Path:
        """キャッシュディレクトリのパスを返す。"""
        return self._cache_dir

    def get_reference_path(self, speaker_id: str) -> Optional[Path]:
        """キャッシュ済み話者代表リファレンスのパスを返す。"""
        return self._refs.get(speaker_id)

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

    def build_references(
        self,
        video_path: Path,
        diarization: List[DiarizationSegment],
    ) -> None:
        """全話者の代表リファレンス音声を生成する。"""
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
                # pydub AudioSegment のメモリを解放する
                del combined
            else:
                print_step(
                    f"  警告: {speaker_id} のリファレンス音声を生成できません"
                )

            # 各話者の処理後にcollectedリストを明示的に解放する
            del collected
            gc.collect()

    def build_segment_references(
        self,
        video_path: Path,
        segments: List[Segment],
    ) -> None:
        """各セグメントの元英語音声をリファレンスとして抽出する。"""
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

            # MioTTS のリファレンス上限は20秒だが、設定値で制限
            effective_end = seg.start + min(dur, MIOTTS_REFERENCE_MAX_SEC)

            extract_audio_segment(
                video_path, out_wav,
                start=seg.start, end=effective_end,
                sample_rate=44100, channels=1,
            )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._segment_refs[segno] = out_wav
            else:
                # 抽出に失敗した場合はファイルを除去
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

    def clear(self) -> None:
        """キャッシュをクリアしてメモリを解放する。"""
        self._refs.clear()
        self._segment_refs.clear()
        gc.collect()
