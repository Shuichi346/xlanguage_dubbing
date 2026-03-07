#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VibeVoice-ASR（Microsoft 製、mlx-audio 経由）による音声認識処理。
文字起こし・話者分離・タイムスタンプを1パスで出力する。

メモリ最適化: エンコーダをチャンク単位で実行し、
Mac Mini 24GB ユニファイドメモリでもメモリスパイクを回避する。
"""

from __future__ import annotations

import gc
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ja_dubbing.config import (
    VIBEVOICE_CHUNK_MAX_SECONDS,
    VIBEVOICE_CHUNK_MIN_SECONDS,
    VIBEVOICE_CHUNK_REFERENCE_AVAILABLE_GB,
    VIBEVOICE_CHUNK_REFERENCE_SECONDS,
    VIBEVOICE_CHUNK_SAFETY_MARGIN,
    VIBEVOICE_CONTEXT,
    VIBEVOICE_MAX_TOKENS,
    VIBEVOICE_MEMORY_LIMIT_RATIO,
    VIBEVOICE_MODEL,
    VIBEVOICE_PREFILL_STEP_SIZE,
)
from ja_dubbing.core.models import DiarizationSegment, Segment
from ja_dubbing.utils import PipelineError, print_step

# モデルの遅延ロード用グローバルキャッシュ
_VIBEVOICE_MODEL = None

# 非音声タグのパターン（[Music], [Noise], [Environmental Sounds], [Speech] など）
_NON_SPEECH_PATTERN = re.compile(r"^\[.*\]$")

# VibeVoice エンコーダの定数
_SAMPLE_RATE = 24000
_HOP_LENGTH = 3200


# ================================================================
# モデル管理
# ================================================================


def _load_model_func():
    """mlx-audio のモデルロード関数を取得する。"""
    try:
        from mlx_audio.stt.utils import load_model
        return load_model
    except ImportError:
        pass

    try:
        from mlx_audio.stt.utils import load as load_model
        return load_model
    except ImportError as exc:
        raise PipelineError(
            "mlx-audio[stt] がインストールされていません。\n"
            "VibeVoice-ASR を使うには以下を実行してください:\n"
            "  uv pip install 'mlx-audio[stt]>=0.3.0'\n"
        ) from exc


def _get_vibevoice_model():
    """VibeVoice-ASR モデルを遅延ロードする（メモリ上限設定付き）。"""
    global _VIBEVOICE_MODEL
    if _VIBEVOICE_MODEL is not None:
        return _VIBEVOICE_MODEL

    import mlx.core as mx

    # GPU 最大推奨サイズの指定割合をメモリ上限に設定（OS 分を確保）
    device_info = mx.metal.device_info()
    max_recommended = device_info["max_recommended_working_set_size"]
    memory_limit = int(max_recommended * VIBEVOICE_MEMORY_LIMIT_RATIO)
    mx.metal.set_memory_limit(memory_limit)
    print_step(
        f"  GPU最大推奨メモリ: {max_recommended / (1024**3):.1f} GB"
    )
    print_step(f"  メモリ上限設定: {memory_limit / (1024**3):.1f} GB")

    mx.reset_peak_memory()

    load_model = _load_model_func()

    print_step(f"  VibeVoice-ASR モデルをロード中: {VIBEVOICE_MODEL}")
    _VIBEVOICE_MODEL = load_model(VIBEVOICE_MODEL)
    mx.eval(_VIBEVOICE_MODEL.parameters())
    model_memory_gb = mx.metal.get_active_memory() / (1024 ** 3)
    print_step(f"  VibeVoice-ASR モデルのロード完了")
    print_step(f"  モデルロード後メモリ: {model_memory_gb:.2f} GB")
    return _VIBEVOICE_MODEL


def _is_non_speech(text: str) -> bool:
    """テキストが非音声タグかどうかを判定する。"""
    return bool(_NON_SPEECH_PATTERN.match(text.strip()))


# ================================================================
# メモリベースのチャンクサイズ自動決定
# ================================================================


def _get_available_memory_gb() -> float:
    """Metal GPU の空きメモリを GB 単位で返す。"""
    import mlx.core as mx

    device_info = mx.metal.device_info()
    max_recommended = device_info["max_recommended_working_set_size"]
    active = mx.metal.get_active_memory()
    available = max_recommended - active
    return available / (1024 ** 3)


def _calculate_encoder_chunk_seconds(
    available_gb: Optional[float] = None,
) -> int:
    """空きメモリに基づいてエンコーダのチャンクサイズ（秒）を決定する。"""
    if available_gb is None:
        available_gb = _get_available_memory_gb()

    ratio = available_gb / VIBEVOICE_CHUNK_REFERENCE_AVAILABLE_GB
    raw_seconds = (
        VIBEVOICE_CHUNK_REFERENCE_SECONDS * ratio * VIBEVOICE_CHUNK_SAFETY_MARGIN
    )

    hop_seconds = _HOP_LENGTH / _SAMPLE_RATE
    aligned_seconds = int(raw_seconds / hop_seconds) * hop_seconds
    aligned_seconds = int(aligned_seconds)

    chunk_seconds = max(
        VIBEVOICE_CHUNK_MIN_SECONDS,
        min(VIBEVOICE_CHUNK_MAX_SECONDS, aligned_seconds),
    )
    return chunk_seconds


# ================================================================
# チャンクエンコード（メモリ最適化の核心）
# ================================================================


def _chunked_encode_speech(model, audio_tensor, chunk_seconds: int) -> "mx.array":
    """
    音声エンコーダをチャンク単位で実行し、メモリスパイクを防ぐ。
    各チャンクのエンコード後にキャッシュクリアする。
    """
    import mlx.core as mx

    total_samples = audio_tensor.shape[1]
    chunk_samples = chunk_seconds * _SAMPLE_RATE
    chunk_samples = (chunk_samples // _HOP_LENGTH) * _HOP_LENGTH
    num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

    print_step(f"    チャンクエンコード: {chunk_seconds}秒 × {num_chunks}チャンク")

    acoustic_features_list = []
    semantic_features_list = []

    for i in range(num_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, total_samples)
        chunk = audio_tensor[:, start:end]
        chunk_dur = (end - start) / _SAMPLE_RATE

        print_step(
            f"    チャンク {i + 1}/{num_chunks} "
            f"({start / _SAMPLE_RATE:.0f}s - {end / _SAMPLE_RATE:.0f}s, "
            f"{chunk_dur:.0f}秒)"
        )

        # batch + channel 次元を追加 [B, T] → [B, 1, T]
        if chunk.ndim == 2:
            chunk_3d = chunk[:, None, :]
        else:
            chunk_3d = chunk

        # Acoustic エンコード
        acoustic_tokens = model.acoustic_tokenizer.encode(chunk_3d)
        mx.eval(acoustic_tokens)
        acoustic_feat = model.acoustic_connector(acoustic_tokens)
        mx.eval(acoustic_feat)
        del acoustic_tokens
        mx.metal.clear_cache()
        acoustic_features_list.append(acoustic_feat)

        # Semantic エンコード
        semantic_tokens = model.semantic_tokenizer.encode(chunk_3d)
        mx.eval(semantic_tokens)
        semantic_feat = model.semantic_connector(semantic_tokens)
        mx.eval(semantic_feat)
        del semantic_tokens, chunk_3d
        mx.metal.clear_cache()
        semantic_features_list.append(semantic_feat)

        print_step(
            f"      メモリ: {mx.metal.get_active_memory() / 1e9:.2f} GB "
            f"(ピーク: {mx.metal.get_peak_memory() / 1e9:.2f} GB)"
        )
        gc.collect()

    # 全チャンクの特徴量を時間軸方向に結合
    print_step("    特徴量結合中...")
    all_acoustic = mx.concatenate(acoustic_features_list, axis=1)
    mx.eval(all_acoustic)
    del acoustic_features_list

    all_semantic = mx.concatenate(semantic_features_list, axis=1)
    mx.eval(all_semantic)
    del semantic_features_list

    combined = all_acoustic + all_semantic
    mx.eval(combined)
    del all_acoustic, all_semantic
    mx.metal.clear_cache()
    gc.collect()

    print_step(
        f"    結合済み特徴量: {combined.shape}, "
        f"メモリ: {mx.metal.get_active_memory() / 1e9:.2f} GB"
    )
    return combined


# ================================================================
# テキスト生成（事前計算済み特徴量を使用）
# ================================================================


def _generate_with_precomputed_features(
    model,
    speech_features,
    audio_duration: float,
    context: Optional[str] = None,
) -> dict:
    """事前計算済みの speech_features を使って LLM でテキスト生成する。"""
    import mlx.core as mx

    try:
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
    except ImportError as exc:
        raise PipelineError(
            "mlx-lm がインストールされていません。\n"
            "  uv pip install 'mlx-audio[stt]>=0.3.0'\n"
        ) from exc

    print_step(
        f"    テキスト生成: 音声長={audio_duration:.1f}秒, "
        f"特徴量トークン数={speech_features.shape[1]}"
    )

    # プロンプト構築
    input_ids, acoustic_input_mask = model._build_prompt_tokens(
        speech_features, audio_duration, context
    )
    print_step(f"    入力トークン数: {input_ids.shape[1]}")

    sampler = make_sampler(0.0, 1.0, 0.0, min_tokens_to_keep=1, top_k=0)
    logits_processors = make_logits_processors(
        repetition_penalty=1.0,
        repetition_context_size=100,
    )

    generated_tokens = []
    start_time = time.time()

    for token, _ in model.stream_generate(
        input_ids=input_ids,
        speech_features=speech_features,
        acoustic_input_mask=acoustic_input_mask,
        max_tokens=VIBEVOICE_MAX_TOKENS,
        sampler=sampler,
        logits_processors=logits_processors,
        prefill_step_size=VIBEVOICE_PREFILL_STEP_SIZE,
        verbose=False,
    ):
        generated_tokens.append(token)

    end_time = time.time()
    mx.metal.clear_cache()

    text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    segments = model.parse_transcription(text)

    total_time = end_time - start_time
    gen_tps = len(generated_tokens) / total_time if total_time > 0 else 0

    print_step(
        f"    生成完了: {len(generated_tokens)}トークン, "
        f"{gen_tps:.1f} tok/s, {total_time:.1f}秒"
    )
    print_step(
        f"    ピークメモリ: {mx.metal.get_peak_memory() / 1e9:.2f} GB"
    )

    return {
        "text": text.strip(),
        "segments": segments,
    }


# ================================================================
# 短い音声向けの通常の generate 呼び出し
# ================================================================


def _generate_standard(model, wav_path: Path) -> dict:
    """短い音声ファイルを通常の model.generate() で処理する。"""
    gen_kwargs = {
        "max_tokens": VIBEVOICE_MAX_TOKENS,
        "temperature": 0.0,
    }
    if VIBEVOICE_CONTEXT.strip():
        gen_kwargs["context"] = VIBEVOICE_CONTEXT.strip()

    result = model.generate(
        audio=str(wav_path),
        verbose=False,
        **gen_kwargs,
    )
    return result


# ================================================================
# メイン関数
# ================================================================


def vibevoice_transcribe(
    wav_path: Path,
) -> Tuple[List[Segment], List[DiarizationSegment]]:
    """
    VibeVoice-ASR で文字起こしと話者分離を同時に実行する。

    音声が長い場合はチャンクエンコード方式でメモリ最適化を行う。
    短い音声の場合は通常の generate() を使用する。
    """
    model = _get_vibevoice_model()

    print_step(f"  VibeVoice-ASR 実行中: {wav_path.name}")
    print_step(f"    max_tokens={VIBEVOICE_MAX_TOKENS}")
    if VIBEVOICE_CONTEXT.strip():
        print_step(f"    context={VIBEVOICE_CONTEXT.strip()}")

    # 音声の長さを推定してチャンクエンコードの要否を判定する
    import mlx.core as mx

    available_gb = _get_available_memory_gb()
    chunk_seconds = _calculate_encoder_chunk_seconds(available_gb)
    print_step(f"    空きメモリ: {available_gb:.2f} GB")
    print_step(
        f"    推奨チャンクサイズ: {chunk_seconds}秒 ({chunk_seconds / 60:.1f}分)"
    )

    # 音声を前処理してサンプル数から長さを判定
    audio_tensor = model._preprocess_audio(str(wav_path))
    audio_duration = audio_tensor.shape[1] / _SAMPLE_RATE
    print_step(
        f"    音声長: {audio_duration:.1f}秒 ({audio_duration / 60:.1f}分)"
    )

    # チャンクサイズより短い場合は通常の generate() を使用
    if audio_duration <= chunk_seconds:
        print_step("    通常の generate() を使用（音声がチャンクサイズ以下）")
        del audio_tensor
        mx.metal.clear_cache()
        gc.collect()

        result = _generate_standard(model, wav_path)
    else:
        # チャンクエンコード方式（メモリ最適化）
        print_step("    チャンクエンコード方式でメモリ最適化実行")
        speech_features = _chunked_encode_speech(
            model, audio_tensor, chunk_seconds
        )

        # 音声テンソルはもう不要
        del audio_tensor
        mx.metal.clear_cache()
        gc.collect()

        context = VIBEVOICE_CONTEXT.strip() if VIBEVOICE_CONTEXT.strip() else None
        result = _generate_with_precomputed_features(
            model, speech_features, audio_duration, context
        )

        del speech_features
        mx.metal.clear_cache()
        gc.collect()

    # result からセグメントと話者分離情報を抽出する
    if isinstance(result, dict):
        raw_segments = result.get("segments", [])
    else:
        raw_segments = (
            result.segments
            if hasattr(result, "segments") and result.segments
            else []
        )

    segments, diarization = _parse_vibevoice_segments(raw_segments)

    if not segments:
        raise PipelineError(
            "VibeVoice-ASR: 文字起こし結果が空です。"
            "音声ファイルに発話が含まれていない可能性があります。"
        )

    speaker_ids = sorted(set(s.speaker_id for s in segments))
    print_step(
        f"  VibeVoice-ASR 完了: {len(segments)} セグメント, "
        f"話者数={len(speaker_ids)}, ID={speaker_ids}"
    )

    return segments, diarization


def _parse_vibevoice_segments(
    raw_segments: list,
) -> Tuple[List[Segment], List[DiarizationSegment]]:
    """VibeVoice-ASR のセグメントをパースする。"""
    segments: List[Segment] = []
    diarization: List[DiarizationSegment] = []
    idx = 0

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue

        text = (seg.get("text", "")).strip()
        if not text:
            continue

        if _is_non_speech(text):
            continue

        speaker_id_raw = seg.get("speaker_id")
        if speaker_id_raw is None:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue

        try:
            speaker_id_int = int(speaker_id_raw)
        except (ValueError, TypeError):
            speaker_id_int = 0
        speaker_label = f"SPEAKER_{speaker_id_int:02d}"

        segments.append(
            Segment(
                idx=idx,
                start=start,
                end=end,
                text_en=text,
                speaker_id=speaker_label,
            )
        )

        diarization.append(
            DiarizationSegment(
                start=start,
                end=end,
                speaker=speaker_label,
            )
        )

        idx += 1

    return segments, diarization


def release_vibevoice_model() -> None:
    """メモリ節約のため VibeVoice-ASR モデルを解放する。"""
    global _VIBEVOICE_MODEL

    if _VIBEVOICE_MODEL is not None:
        import mlx.core as mx
        del _VIBEVOICE_MODEL
        _VIBEVOICE_MODEL = None
        mx.metal.clear_cache()
        gc.collect()
        print_step("  VibeVoice-ASR モデルを解放しました")
