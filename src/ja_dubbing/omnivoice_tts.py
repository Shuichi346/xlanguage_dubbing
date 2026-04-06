#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniVoice による音声合成処理。
600+言語対応のゼロショットボイスクローン TTS。
参照音声の声色を再現しつつ、日本語テキストを読み上げる。
duration パラメータにより再生時間制御が可能。
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Optional

import torch

from ja_dubbing.audio.ffmpeg import ffprobe_duration_sec
from ja_dubbing.config import (
    MIN_SEGMENT_SEC,
    OMNIVOICE_DURATION_SCALE,
    OMNIVOICE_DURATION_TOLERANCE,
    OMNIVOICE_DTYPE,
    OMNIVOICE_GUIDANCE_SCALE,
    OMNIVOICE_MODEL,
    OMNIVOICE_NUM_STEP,
    OMNIVOICE_QUALITY_RETRIES,
    OMNIVOICE_SAMPLE_RATE,
    OMNIVOICE_SPEED,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
)
from ja_dubbing.core.models import Segment, TtsMeta
from ja_dubbing.tts.reference import SpeakerReferenceCache
from ja_dubbing.utils import (
    PipelineError,
    ensure_dir,
    print_step,
    run_cmd,
    sanitize_text_for_tts,
    which_or_raise,
)

# モデルの遅延ロード用グローバルキャッシュ
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
    # MPS では float32 が最も安定（bfloat16/float16 は音声品質劣化の報告あり）
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
            "以下を実行してください:\n"
            "  uv sync\n"
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
    """OmniVoice モデルを解放してメモリを回復する。"""
    global _OMNIVOICE_MODEL

    if _OMNIVOICE_MODEL is not None:
        del _OMNIVOICE_MODEL
        _OMNIVOICE_MODEL = None

    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    print_step("  OmniVoice モデルを解放しました")


def omnivoice_synthesize(
    text_ja: str,
    out_wav: Path,
    ref_audio_path: Optional[Path] = None,
    ref_text: str = "",
    target_duration: Optional[float] = None,
) -> None:
    """OmniVoice で音声を合成する。"""
    ensure_dir(out_wav.parent)
    model = _get_omnivoice_model()

    gen_kwargs = {
        "text": text_ja,
        "num_step": OMNIVOICE_NUM_STEP,
        "guidance_scale": OMNIVOICE_GUIDANCE_SCALE,
    }

    # ボイスクローン: 参照音声が指定されている場合
    if ref_audio_path is not None and ref_audio_path.exists():
        gen_kwargs["ref_audio"] = str(ref_audio_path)
        # ref_text が空なら OmniVoice 内蔵の Whisper が自動文字起こしする
        if ref_text.strip():
            gen_kwargs["ref_text"] = ref_text.strip()

    # 再生時間制御: duration が指定されている場合は duration を優先
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

    # audio_list は List[torch.Tensor] で各テンソルは (1, T) @ 24kHz
    # 全チャンクを結合
    if len(audio_list) == 1:
        waveform = audio_list[0]
    else:
        waveform = torch.cat(audio_list, dim=-1)

    # WAV として保存（24kHz）
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError(
            "soundfile がインストールされていません。\n"
            "  uv sync を実行してください。"
        ) from exc

    wav_np = waveform.squeeze().detach().cpu().float().numpy()
    sf.write(str(out_wav), wav_np, OMNIVOICE_SAMPLE_RATE)


def _convert_to_flac(in_wav: Path, out_flac: Path) -> None:
    """OmniVoice が出力した WAV（24kHz）をプロジェクト標準の FLAC に変換する。"""
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
    """生成音声の長さが目標から大きく外れていないか検査する。"""
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
    ref_audio_path: Optional[Path],
    ref_text: str,
    target_duration: float,
) -> None:
    """品質チェック付きで OmniVoice を実行する。"""
    attempts = OMNIVOICE_QUALITY_RETRIES + 1

    for attempt in range(1, attempts + 1):
        omnivoice_synthesize(
            text_ja=text,
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
) -> Optional[TtsMeta]:
    """OmniVoice でセグメントのボイスクローン音声を生成する。

    セグメント単位リファレンスを優先し、なければ話者代表リファレンスにフォールバックする。
    """
    if seg.duration < MIN_SEGMENT_SEC:
        return None

    text = sanitize_text_for_tts(seg.text_ja)
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

    # セグメント単位リファレンスを優先、なければ話者代表にフォールバック
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


# =========================================================
# TTS メタ I/O（miotts.py から移動）
# =========================================================


def load_tts_meta(path: Path) -> dict[int, TtsMeta]:
    """TTSメタ情報を読み込む。"""
    from ja_dubbing.utils import load_json_if_exists

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
    from ja_dubbing.utils import atomic_write_json

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
