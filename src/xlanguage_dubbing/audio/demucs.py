#!/usr/bin/env python3
"""
Audio source separation with Demucs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from xlanguage_dubbing.config import DEMUCS_DEVICE, DEMUCS_MODEL
from xlanguage_dubbing.utils import (
    PipelineError,
    ensure_dir,
    print_step,
    run_cmd,
    which_or_raise,
)


def extract_audio_for_demucs(media_in: Path, out_wav: Path) -> None:
    """Extract input media audio as a 44.1kHz stereo WAV for Demucs."""
    which_or_raise("ffmpeg")
    ensure_dir(out_wav.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(media_in),
        "-vn",
        "-ac", "2",
        "-ar", "44100",
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run_cmd(cmd)


def separate_voice_and_background(
    media_in: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Separate the input audio into vocals and background stems."""
    if importlib.util.find_spec("demucs") is None:
        raise PipelineError(
            "demucs がインストールされていません。\n"
            "  uv pip install demucs を実行してください。"
        )

    demucs_dir = work_dir / "demucs"
    input_wav = demucs_dir / "demucs_input.wav"
    stem_dir = demucs_dir / DEMUCS_MODEL / input_wav.stem
    vocals_wav = stem_dir / "vocals.wav"
    background_wav = stem_dir / "no_vocals.wav"

    if vocals_wav.exists() and background_wav.exists():
        print_step("1. Demucs音声分離: 既存の分離済み音声を利用")
        return vocals_wav, background_wav

    if not input_wav.exists():
        print_step("1. 元音声をDemucs入力用WAVに抽出")
        extract_audio_for_demucs(media_in, input_wav)

    print_step(
        "1. Demucsで音声分離: "
        f"model={DEMUCS_MODEL}, device={DEMUCS_DEVICE}, two-stems=vocals"
    )
    cmd = [
        sys.executable,
        "-m", "demucs",
        "-n", DEMUCS_MODEL,
        "--two-stems", "vocals",
        "-d", DEMUCS_DEVICE,
        "-o", str(demucs_dir),
        str(input_wav),
    ]
    run_cmd(cmd)

    if not vocals_wav.exists() or not background_wav.exists():
        raise PipelineError(
            "Demucs の出力ファイルが見つかりません。\n"
            f"  vocals: {vocals_wav}\n"
            f"  background: {background_wav}"
        )

    return vocals_wav, background_wav
