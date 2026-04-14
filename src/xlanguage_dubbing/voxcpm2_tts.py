#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoxCPM2 による音声合成処理。
30言語対応の Ultimate Cloning モード（参照音声＋参照テキスト）で
最高品質のボイスクローン TTS を実現する。
48kHz ネイティブ出力。
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Optional

import torch

from xlanguage_dubbing.audio.ffmpeg import ffprobe_duration_sec
from xlanguage_dubbing.config import (
    MIN_SEGMENT_SEC,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
    VOXCPM2_CFG_VALUE,
    VOXCPM2_DURATION_SCALE,
    VOXCPM2_DURATION_TOLERANCE,
    VOXCPM2_INFERENCE_TIMESTEPS,
    VOXCPM2_MODEL,
    VOXCPM2_QUALITY_RETRIES,
    VOXCPM2_SAMPLE_RATE,
)
from xlanguage_dubbing.core.models import Segment, TtsMeta
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache
from xlanguage_dubbing.utils import (
    PipelineError,
    ensure_dir,
    print_step,
    run_cmd,
    sanitize_text_for_tts,
    which_or_raise,
)

_VOXCPM2_MODEL = None


class TTSQualityError(Exception):
    """TTS 品質バリデーション失敗を示す例外。"""


def _get_voxcpm2_model():
    """VoxCPM2 モデルを遅延ロードする。

    voxcpm パッケージのバージョンにより from_pretrained() が受け付ける
    引数が異なるため、device/optimize 付きで試行し、TypeError になった場合は
    引数なしで再試行する。
    """
    global _VOXCPM2_MODEL
    if _VOXCPM2_MODEL is not None:
        return _VOXCPM2_MODEL

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        from voxcpm import VoxCPM
    except ImportError as exc:
        raise PipelineError(
            "voxcpm がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    print_step(f"  VoxCPM2 モデル初期化中: {VOXCPM2_MODEL}")

    # v2.0.2 以降は device / optimize / load_denoiser を受け付ける。
    # v2.0.0 では device が __init__ に存在しない場合がある。
    try:
        _VOXCPM2_MODEL = VoxCPM.from_pretrained(
            VOXCPM2_MODEL,
            device="auto",
            optimize=False,
            load_denoiser=False,
        )
    except TypeError as exc:
        # 古いバージョンとの互換性: 認識されない引数を除去して再試行する
        print_step(
            f"  VoxCPM2 初期化リトライ（互換モード）: {exc}"
        )
        try:
            _VOXCPM2_MODEL = VoxCPM.from_pretrained(
                VOXCPM2_MODEL,
                load_denoiser=False,
            )
        except TypeError:
            # load_denoiser も未サポートの場合は引数なしで試す
            _VOXCPM2_MODEL = VoxCPM.from_pretrained(VOXCPM2_MODEL)

    print_step("  VoxCPM2 モデル初期化完了")
    return _VOXCPM2_MODEL


def release_voxcpm2_model() -> None:
    """VoxCPM2 モデルを解放する。"""
    global _VOXCPM2_MODEL

    if _VOXCPM2_MODEL is not None:
        del _VOXCPM2_MODEL
        _VOXCPM2_MODEL = None

    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print_step("  VoxCPM2 モデルを解放しました")


def voxcpm2_synthesize(
    text: str,
    out_wav: Path,
    ref_audio_path: Optional[Path] = None,
    ref_text: str = "",
) -> None:
    """VoxCPM2 の Ultimate Cloning モードで音声を合成する。

    Ultimate Cloning: prompt_wav_path + prompt_text + reference_wav_path
    で最高品質のボイスクローンを実現する。
    同じ参照音声を prompt_wav_path と reference_wav_path の両方に渡すことで
    音色の類似度を最大化する（公式ドキュメント推奨）。
    """
    ensure_dir(out_wav.parent)
    model = _get_voxcpm2_model()

    gen_kwargs = {
        "text": text,
        "cfg_value": VOXCPM2_CFG_VALUE,
        "inference_timesteps": VOXCPM2_INFERENCE_TIMESTEPS,
    }

    if ref_audio_path is not None and ref_audio_path.exists():
        ref_path_str = str(ref_audio_path)
        # Ultimate Cloning: prompt_wav_path + prompt_text + reference_wav_path
        gen_kwargs["prompt_wav_path"] = ref_path_str
        gen_kwargs["reference_wav_path"] = ref_path_str
        if ref_text.strip():
            gen_kwargs["prompt_text"] = ref_text.strip()

    try:
        wav = model.generate(**gen_kwargs)
    except Exception as exc:
        raise PipelineError(f"VoxCPM2 生成エラー: {exc}") from exc

    if wav is None or (hasattr(wav, "__len__") and len(wav) == 0):
        raise PipelineError("VoxCPM2: 生成音声が空です。")

    try:
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError(
            "soundfile がインストールされていません。\n"
            "  uv sync を実行してください。"
        ) from exc

    import numpy as np

    if isinstance(wav, torch.Tensor):
        wav_np = wav.squeeze().detach().cpu().float().numpy()
    elif isinstance(wav, np.ndarray):
        wav_np = wav.squeeze()
    else:
        wav_np = np.array(wav).squeeze()

    # サンプルレートをモデルから動的に取得する（フォールバック: 設定値）
    sample_rate = VOXCPM2_SAMPLE_RATE
    if hasattr(model, "tts_model") and hasattr(model.tts_model, "sample_rate"):
        sample_rate = int(model.tts_model.sample_rate)

    sf.write(str(out_wav), wav_np, sample_rate)


def _convert_to_flac(in_wav: Path, out_flac: Path) -> None:
    """WAV をプロジェクト標準の FLAC に変換する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_flac.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_wav),
        "-ac", str(TTS_CHANNELS),
        "-ar", str(TTS_SAMPLE_RATE),
        "-c:a", "flac",
        str(out_flac),
    ]
    run_cmd(cmd)


def _validate_voxcpm2_quality(
    audio_duration_sec: float,
    target_duration_sec: float,
    text: str,
) -> None:
    """生成音声の品質を検査する。"""
    if not text.strip():
        raise TTSQualityError("合成対象テキストが空です。")
    if audio_duration_sec <= 0:
        raise TTSQualityError("生成音声の長さが 0 秒以下です。")
    if target_duration_sec <= 0:
        return
    ratio = abs(audio_duration_sec - target_duration_sec) / target_duration_sec
    if ratio > VOXCPM2_DURATION_TOLERANCE:
        raise TTSQualityError(
            f"生成音声の長さ({audio_duration_sec:.1f}秒)が "
            f"目標({target_duration_sec:.1f}秒)から "
            f"{ratio * 100:.0f}%乖離（許容: {VOXCPM2_DURATION_TOLERANCE * 100:.0f}%）"
        )


def _synthesize_with_quality_retry(
    text: str,
    tmp_wav: Path,
    ref_audio_path: Optional[Path],
    ref_text: str,
    target_duration: float,
) -> None:
    """品質チェック付きで VoxCPM2 を実行する。"""
    attempts = VOXCPM2_QUALITY_RETRIES + 1

    for attempt in range(1, attempts + 1):
        voxcpm2_synthesize(
            text=text,
            out_wav=tmp_wav,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
        )

        if not tmp_wav.exists() or tmp_wav.stat().st_size <= 100:
            if attempt < attempts:
                print_step(
                    f"    品質リトライ {attempt}/{attempts}: "
                    "生成ファイルが空 → 再生成"
                )
                tmp_wav.unlink(missing_ok=True)
                continue
            raise PipelineError("VoxCPM2: 生成ファイルが空です。")

        try:
            duration = ffprobe_duration_sec(tmp_wav)
            _validate_voxcpm2_quality(duration, target_duration, text)
            return
        except TTSQualityError as exc:
            if attempt < attempts:
                wait_sec = float(attempt)
                print_step(
                    f"    品質リトライ {attempt}/{attempts}: "
                    f"{exc} → {wait_sec:.0f}秒後に再生成"
                )
                tmp_wav.unlink(missing_ok=True)
                time.sleep(wait_sec)
            else:
                print_step(
                    f"    品質リトライ枯渇 {attempt}/{attempts}: "
                    f"{exc} → そのまま使用"
                )
                return


def generate_segment_tts_voxcpm2(
    seg: Segment,
    out_audio_stub: Path,
    ref_cache: SpeakerReferenceCache,
    segno: int = 0,
) -> Optional[TtsMeta]:
    """VoxCPM2 でセグメントのボイスクローン音声を生成する。"""
    if seg.duration < MIN_SEGMENT_SEC:
        return None

    text = sanitize_text_for_tts(seg.text_tgt)
    if not text:
        return None

    out_flac = out_audio_stub.with_suffix(".flac")
    if out_flac.exists():
        duration = ffprobe_duration_sec(out_flac)
        if duration > 0:
            return TtsMeta(
                segno=segno,
                flac_path=str(out_flac),
                duration_sec=float(duration),
            )

    # セグメント単位リファレンスを優先、なければ話者代表リファレンス
    seg_ref_path = ref_cache.get_omnivoice_segment_reference_path(segno)
    if seg_ref_path is not None:
        reference_speech = seg_ref_path
        reference_text = ref_cache.get_omnivoice_segment_prompt_text(segno)
    else:
        reference_speech = ref_cache.get_omnivoice_reference_path(seg.speaker_id)
        reference_text = ref_cache.get_omnivoice_prompt_text(seg.speaker_id)

    if reference_speech is None:
        print_step(
            f"    警告: 話者 {seg.speaker_id} の VoxCPM2 リファレンスがありません。"
        )
        return None

    target_duration = max(seg.duration * VOXCPM2_DURATION_SCALE, 0.5)

    tmp_wav = out_audio_stub.with_suffix(".voxcpm2.wav")

    try:
        _synthesize_with_quality_retry(
            text=text,
            tmp_wav=tmp_wav,
            ref_audio_path=reference_speech,
            ref_text=reference_text,
            target_duration=target_duration,
        )
        _convert_to_flac(tmp_wav, out_flac)
        duration = ffprobe_duration_sec(out_flac)
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass

    if duration <= 0:
        return None

    return TtsMeta(
        segno=segno,
        flac_path=str(out_flac),
        duration_sec=float(duration),
    )
