#!/usr/bin/env python3
"""
ASR エンジンの統一インターフェース。
"""

from __future__ import annotations

from pathlib import Path

from xlanguage_dubbing.config import ASR_ENGINE
from xlanguage_dubbing.utils import PipelineError


def get_asr_engine() -> str:
    """現在選択されている ASR エンジン名を返す。"""
    engine = ASR_ENGINE.strip().lower()
    if engine not in {"vibevoice", "whisper", "nemo-speech"}:
        raise PipelineError(
            f"未知の ASR_ENGINE です: {engine}（vibevoice / whisper / nemo-speech を指定）"
        )
    return engine


def transcribe_reference_audio(wav_path: Path, language: str = "") -> str:
    """参照音声ファイルを文字起こしする統一エントリポイント。"""
    engine = get_asr_engine()

    if engine == "vibevoice":
        from xlanguage_dubbing.asr.vibevoice import transcribe_short_audio_vibevoice
        return transcribe_short_audio_vibevoice(wav_path)

    if engine == "nemo-speech":
        from xlanguage_dubbing.asr.nemo_speech import transcribe_short_audio_nemo
        return transcribe_short_audio_nemo(wav_path, language=language)

    from xlanguage_dubbing.asr.whisper import transcribe_short_audio
    return transcribe_short_audio(wav_path, language=language)
