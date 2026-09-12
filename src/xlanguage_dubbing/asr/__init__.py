#!/usr/bin/env python3
"""
ASR エンジンの統一インターフェース。
"""

from __future__ import annotations

from pathlib import Path

from xlanguage_dubbing.config import ASR_ENGINE


def get_asr_engine() -> str:
    """現在選択されている ASR エンジン名を返す。"""
    return ASR_ENGINE.strip().lower()


def transcribe_reference_audio(wav_path: Path, language: str = "") -> str:
    """参照音声ファイルを文字起こしする統一エントリポイント。"""
    engine = get_asr_engine()

    if engine == "vibevoice":
        from xlanguage_dubbing.asr.vibevoice import transcribe_short_audio_vibevoice
        return transcribe_short_audio_vibevoice(wav_path)

    from xlanguage_dubbing.asr.whisper import transcribe_short_audio
    return transcribe_short_audio(wav_path, language=language)
