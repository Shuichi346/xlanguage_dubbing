#!/usr/bin/env python3
"""
データ構造の定義。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiarizationSegment:
    """話者分離結果の1区間。"""

    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class Segment:
    """音声セグメントを表すデータクラス。

    text_src: 元言語のテキスト（旧 text_en）
    text_tgt: 翻訳後のテキスト（旧 text_ja）
    detected_lang: 検出された元言語の ISO 639-1 コード
    """

    idx: int
    start: float
    end: float
    text_src: str
    text_tgt: str = ""
    speaker_id: str = ""
    detected_lang: str = ""

    # 後方互換性のためのプロパティ
    @property
    def text_en(self) -> str:
        """text_src のエイリアス（後方互換性）。"""
        return self.text_src

    @property
    def text_ja(self) -> str:
        """text_tgt のエイリアス（後方互換性）。"""
        return self.text_tgt

    @property
    def duration(self) -> float:
        """セグメントの長さを秒で返す。"""
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TtsMeta:
    """TTS生成結果のメタ情報（再開用）。"""

    segno: int  # 1-based
    flac_path: str
    duration_sec: float


@dataclass(frozen=True)
class RetimePart:
    """元動画の区間を、新しいタイムライン上で何秒にするかを表す。"""

    kind: str  # "gap" or "speech" or "tail"
    orig_start: float
    orig_end: float
    out_duration: float
    speed: float
    segno: int = 0  # speech の場合のみ 1-based
