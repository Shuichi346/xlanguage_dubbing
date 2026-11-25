#!/usr/bin/env python3
"""
Audio source separation with Demucs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from xlanguage_dubbing.audio.ffmpeg import validate_audio_file
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
    with TemporaryDirectory(prefix=".extract-", dir=out_wav.parent) as directory:
        staged = Path(directory) / out_wav.name
        run_cmd([*cmd[:-1], str(staged)])
        if not validate_audio_file(staged, sample_rate=44100, channels=2):
            raise PipelineError("Demucs入力音声の検証に失敗しました。")
        staged.replace(out_wav)


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

    if not validate_audio_file(input_wav, sample_rate=44100, channels=2):
        print_step("1. 元音声をDemucs入力用WAVに抽出")
        extract_audio_for_demucs(media_in, input_wav)
    expected_samples = sf.info(input_wav).frames
    if all(validate_audio_file(path, expected_samples=expected_samples,
                               sample_rate=44100, channels=2)
           for path in (vocals_wav, background_wav)):
        print_step("1. Demucs音声分離: 既存の分離済み音声を利用")
        return vocals_wav, background_wav

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
    with TemporaryDirectory(prefix=".separate-", dir=demucs_dir) as directory:
        cmd[cmd.index("-o") + 1] = directory
        run_cmd(cmd)
        staged_dir = Path(directory) / DEMUCS_MODEL / input_wav.stem
        staged_stems = [staged_dir / path.name for path in (vocals_wav, background_wav)]
        if not all(validate_audio_file(path, expected_samples=expected_samples,
                                       sample_rate=44100, channels=2) for path in staged_stems):
            raise PipelineError("Demucs の出力音声が不完全、または入力と長さが一致しません。")
        ensure_dir(stem_dir)
        for staged, destination in zip(staged_stems, (vocals_wav, background_wav), strict=True):
            staged.replace(destination)

    return vocals_wav, background_wav
