#!/usr/bin/env python3
"""
OmniVoice による音声合成処理。
600+言語対応のゼロショットボイスクローン TTS。
OUTPUT_LANG に合わせた言語で音声を生成する。
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import numpy as np
import torch

from xlanguage_dubbing.audio.ffmpeg import ffprobe_duration_sec
from xlanguage_dubbing.config import (
    MIN_SEGMENT_SEC,
    OMNIVOICE_DTYPE,
    OMNIVOICE_DURATION_SCALE,
    OMNIVOICE_DURATION_TOLERANCE,
    OMNIVOICE_GUIDANCE_SCALE,
    OMNIVOICE_MODEL,
    OMNIVOICE_NUM_STEP,
    OMNIVOICE_QUALITY_RETRIES,
    OMNIVOICE_SAMPLE_RATE,
    OMNIVOICE_SPEED,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
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

_OMNIVOICE_MODEL = None


class TTSQualityError(Exception):
    """TTS 品質バリデーション失敗を示す例外。"""


def _get_device() -> str:
    """推論デバイスを返す。"""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_dtype() -> torch.dtype:
    """設定に基づいた dtype を返す。"""
    dtype_str = OMNIVOICE_DTYPE.strip().lower()
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def _get_omnivoice_model():
    """OmniVoice モデルを遅延ロードする。"""
    global _OMNIVOICE_MODEL
    if _OMNIVOICE_MODEL is not None:
        return _OMNIVOICE_MODEL

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise PipelineError(
            "omnivoice がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    device = _get_device()
    dtype = _get_dtype()

    print_step(f"  OmniVoice モデル初期化中: {OMNIVOICE_MODEL} ({device}, {dtype})")

    _OMNIVOICE_MODEL = OmniVoice.from_pretrained(
        OMNIVOICE_MODEL,
        device_map=device,
        dtype=dtype,
    )

    print_step("  OmniVoice モデル初期化完了")
    return _OMNIVOICE_MODEL


def release_omnivoice_model() -> None:
    """OmniVoice モデルを解放する。"""
    global _OMNIVOICE_MODEL

    if _OMNIVOICE_MODEL is not None:
        del _OMNIVOICE_MODEL
        _OMNIVOICE_MODEL = None

    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print_step("  OmniVoice モデルを解放しました")


def _to_numpy(waveform) -> np.ndarray:
    """torch.Tensor / numpy.ndarray / その他を 1D numpy float32 に変換する。"""
    if isinstance(waveform, torch.Tensor):
        return waveform.squeeze().detach().cpu().float().numpy()
    if isinstance(waveform, np.ndarray):
        return waveform.squeeze().astype(np.float32)
    return np.array(waveform, dtype=np.float32).squeeze()


def omnivoice_synthesize(
    text: str,
    out_wav: Path,
    ref_audio_path: Path | None = None,
    ref_text: str = "",
    target_duration: float | None = None,
) -> None:
    """OmniVoice で音声を合成する。"""
    ensure_dir(out_wav.parent)
    model = _get_omnivoice_model()

    gen_kwargs = {
        "text": text,
        "num_step": OMNIVOICE_NUM_STEP,
        "guidance_scale": OMNIVOICE_GUIDANCE_SCALE,
    }

    if ref_audio_path is not None and ref_audio_path.exists():
        gen_kwargs["ref_audio"] = str(ref_audio_path)
        if ref_text.strip():
            gen_kwargs["ref_text"] = ref_text.strip()

    if target_duration is not None and target_duration > 0:
        gen_kwargs["duration"] = target_duration
    elif OMNIVOICE_SPEED != 1.0:
        gen_kwargs["speed"] = OMNIVOICE_SPEED

    try:
        audio_list = model.generate(**gen_kwargs)
    except Exception as exc:
        raise PipelineError(f"OmniVoice 生成エラー: {exc}") from exc

    if not audio_list or len(audio_list) == 0:
        raise PipelineError("OmniVoice: 生成音声が空です。")

    if len(audio_list) == 1:
        wav_np = _to_numpy(audio_list[0])
    else:
        wav_np = np.concatenate(
            [_to_numpy(a) for a in audio_list], axis=-1
        )

    try:
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError(
            "soundfile がインストールされていません。\n"
            "  uv sync を実行してください。"
        ) from exc

    # サンプルレートをモデルから動的に取得する（フォールバック: 設定値）
    sample_rate = OMNIVOICE_SAMPLE_RATE
    if hasattr(model, "sampling_rate") and model.sampling_rate:
        sample_rate = int(model.sampling_rate)

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


def _validate_omnivoice_quality(
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
    if ratio > OMNIVOICE_DURATION_TOLERANCE:
        raise TTSQualityError(
            f"生成音声の長さ({audio_duration_sec:.1f}秒)が "
            f"目標({target_duration_sec:.1f}秒)から "
            f"{ratio * 100:.0f}%乖離（許容: {OMNIVOICE_DURATION_TOLERANCE * 100:.0f}%）"
        )


def _synthesize_with_quality_retry(
    text: str,
    tmp_wav: Path,
    ref_audio_path: Path | None,
    ref_text: str,
    target_duration: float,
) -> None:
    """品質チェック付きで OmniVoice を実行する。"""
    attempts = OMNIVOICE_QUALITY_RETRIES + 1

    for attempt in range(1, attempts + 1):
        omnivoice_synthesize(
            text=text,
            out_wav=tmp_wav,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            target_duration=target_duration,
        )

        if not tmp_wav.exists() or tmp_wav.stat().st_size <= 100:
            if attempt < attempts:
                print_step(
                    f"    品質リトライ {attempt}/{attempts}: "
                    "生成ファイルが空 → 再生成"
                )
                tmp_wav.unlink(missing_ok=True)
                continue
            raise PipelineError("OmniVoice: 生成ファイルが空です。")

        try:
            duration = ffprobe_duration_sec(tmp_wav)
            _validate_omnivoice_quality(duration, target_duration, text)
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


def generate_segment_tts_omnivoice(
    seg: Segment,
    out_audio_stub: Path,
    ref_cache: SpeakerReferenceCache,
    segno: int = 0,
) -> TtsMeta | None:
    """OmniVoice でセグメントのボイスクローン音声を生成する。"""
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

    seg_ref_path = ref_cache.get_omnivoice_segment_reference_path(segno)
    if seg_ref_path is not None:
        reference_speech = seg_ref_path
        reference_text = ref_cache.get_omnivoice_segment_prompt_text(segno)
    else:
        reference_speech = ref_cache.get_omnivoice_reference_path(seg.speaker_id)
        reference_text = ref_cache.get_omnivoice_prompt_text(seg.speaker_id)

    if reference_speech is None:
        print_step(
            f"    警告: 話者 {seg.speaker_id} の OmniVoice リファレンスがありません。"
        )
        return None

    target_duration = max(seg.duration * OMNIVOICE_DURATION_SCALE, 0.5)

    tmp_wav = out_audio_stub.with_suffix(".omnivoice.wav")

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


def load_tts_meta(path: Path) -> dict[int, TtsMeta]:
    """TTSメタ情報を読み込む。"""
    from xlanguage_dubbing.utils import load_json_if_exists

    obj = load_json_if_exists(path)
    if not isinstance(obj, list):
        return {}
    out: dict[int, TtsMeta] = {}
    for row in obj:
        if not isinstance(row, dict):
            continue
        segno = int(row.get("segno", 0) or 0)
        flac_path = str(row.get("flac_path", "") or "")
        dur = float(row.get("duration_sec", 0.0) or 0.0)
        if segno <= 0 or not flac_path or dur <= 0:
            continue
        out[segno] = TtsMeta(segno=segno, flac_path=flac_path, duration_sec=dur)
    return out


def save_tts_meta_atomic(path: Path, meta: dict[int, TtsMeta]) -> None:
    """TTSメタ情報をアトミックに保存する。"""
    from xlanguage_dubbing.utils import atomic_write_json

    rows: list[dict] = []
    for segno in sorted(meta.keys()):
        m = meta[segno]
        rows.append(
            {
                "segno": m.segno,
                "flac_path": m.flac_path,
                "duration_sec": m.duration_sec,
            }
        )
    atomic_write_json(path, rows)
