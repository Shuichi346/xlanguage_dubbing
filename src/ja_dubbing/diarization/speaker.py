#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pyannote.audioによる話者分離処理。
macOS torchcodec互換回避済み。
pyannote.audio 4.x の DiarizeOutput API に対応。
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import List

from ja_dubbing.config import HF_AUTH_TOKEN, PYANNOTE_MODEL
from ja_dubbing.core.models import DiarizationSegment
from ja_dubbing.utils import PipelineError, print_step

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
            "  uv pip install pyannote-audio\n"
        ) from exc

    if not HF_AUTH_TOKEN:
        raise PipelineError(
            "HF_AUTH_TOKEN が未設定です。.env に HuggingFace トークンを設定してください。"
        )

    # pyannote.audioのバージョンによりパラメータ名が異なる場合を考慮
    try:
        _PIPELINE = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            token=HF_AUTH_TOKEN,
        )
    except TypeError:
        _PIPELINE = Pipeline.from_pretrained(
            PYANNOTE_MODEL,
            token=HF_AUTH_TOKEN,
        )
    return _PIPELINE


def _extract_annotation(raw_output):
    """
    pyannote.audio の出力から Annotation オブジェクトを取得する。
    4.x: DiarizeOutput → output.speaker_diarization で Annotation を取得
    3.x: 直接 Annotation が返る
    """
    # 4.x: DiarizeOutput ラッパー
    if hasattr(raw_output, "speaker_diarization"):
        return raw_output.speaker_diarization

    # 3.x: 直接 Annotation
    if hasattr(raw_output, "itertracks"):
        return raw_output

    raise PipelineError(
        f"pyannote.audio の出力形式が不明です: {type(raw_output).__name__}\n"
        "pyannote.audio 3.x または 4.x に対応しています。"
    )


def run_diarization(wav_path: Path) -> List[DiarizationSegment]:
    """話者分離を実行する。macOS torchcodec互換回避済み。"""
    import torchaudio

    pipeline = _get_pipeline()

    # macOS (brew ffmpeg) での torchcodec 不整合を回避
    waveform, sample_rate = torchaudio.load(str(wav_path))
    print_step(f"  話者分離: waveform shape={waveform.shape}, sr={sample_rate}")

    raw_output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # waveform テンソルを即座に解放する
    del waveform
    gc.collect()

    # 4.x / 3.x 両対応で Annotation を取得
    annotation = _extract_annotation(raw_output)

    results: List[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        results.append(DiarizationSegment(
            start=float(turn.start),
            end=float(turn.end),
            speaker=str(speaker),
        ))

    # raw_output（内部テンソル含む）を解放する
    del raw_output, annotation
    gc.collect()

    print_step(
        f"  話者分離完了: {len(results)} 区間, "
        f"話者数={len(set(r.speaker for r in results))}"
    )
    return results


def release_pipeline() -> None:
    """メモリ節約のためパイプラインとtorchキャッシュを解放する。"""
    global _PIPELINE

    if _PIPELINE is not None:
        del _PIPELINE
        _PIPELINE = None

    gc.collect()

    # PyTorch MPS メモリキャッシュを明示的に解放する
    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass

    gc.collect()
    print_step("  pyannote パイプラインと torch MPS キャッシュを解放しました")
