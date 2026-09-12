#!/usr/bin/env python3
"""
pyannote.audioによる話者分離処理。
"""

from __future__ import annotations

import gc
from pathlib import Path

from xlanguage_dubbing.config import HF_AUTH_TOKEN, PYANNOTE_MODEL
from xlanguage_dubbing.core.models import DiarizationSegment
from xlanguage_dubbing.utils import PipelineError, print_step

_PIPELINE = None


def _get_pipeline():
    """pyannoteパイプラインを遅延ロードする。"""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise PipelineError(
            "pyannote-audio がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    if not HF_AUTH_TOKEN:
        raise PipelineError(
            "HF_AUTH_TOKEN が未設定です。.env に HuggingFace トークンを設定してください。"
        )

    _PIPELINE = Pipeline.from_pretrained(
        PYANNOTE_MODEL,
        token=HF_AUTH_TOKEN,
    )
    return _PIPELINE


def _load_audio_waveform(wav_path: Path):
    """音声ファイルを waveform テンソルとして読み込む。"""
    import torch

    try:
        from torchcodec.decoders import AudioDecoder

        decoder = AudioDecoder(str(wav_path))
        result = decoder.get_all_samples()
        waveform = result.data
        sample_rate = result.sample_rate
        if waveform.dtype != torch.float32:
            waveform = waveform.to(torch.float32)
            if waveform.abs().max() > 1.0:
                waveform = waveform / 32768.0
        return waveform, int(sample_rate)
    except Exception:
        pass

    try:
        import torchaudio
        waveform, sample_rate = torchaudio.load(str(wav_path))
        return waveform, int(sample_rate)
    except Exception:
        pass

    try:
        import numpy as np
        import soundfile as sf

        data, sample_rate = sf.read(str(wav_path), dtype="float32")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        else:
            data = data.T
        waveform = torch.from_numpy(data)
        return waveform, int(sample_rate)
    except Exception as exc:
        raise PipelineError(
            f"音声ファイルを読み込めません: {wav_path}"
        ) from exc


def _extract_annotation(raw_output):
    """pyannote.audio の出力から Annotation を取得する。"""
    if hasattr(raw_output, "speaker_diarization"):
        return raw_output.speaker_diarization

    if hasattr(raw_output, "itertracks"):
        return raw_output

    raise PipelineError(
        f"pyannote.audio の出力形式が不明です: {type(raw_output).__name__}"
    )


def run_diarization(wav_path: Path) -> list[DiarizationSegment]:
    """話者分離を実行する。"""
    pipeline = _get_pipeline()

    waveform, sample_rate = _load_audio_waveform(wav_path)
    print_step(f"  話者分離: waveform shape={waveform.shape}, sr={sample_rate}")

    raw_output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    del waveform
    gc.collect()

    annotation = _extract_annotation(raw_output)

    results: list[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append(DiarizationSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
        ))

    del raw_output, annotation
    gc.collect()

    print_step(
        f"  話者分離完了: {len(results)} 区間, "
        f"話者数={len({r.speaker for r in results})}"
    )
    return results


def release_pipeline() -> None:
    """パイプラインを解放する。"""
    global _PIPELINE

    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None

    gc.collect()

    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass

    gc.collect()
    print_step("  pyannote パイプラインと torch MPS キャッシュを解放しました")
