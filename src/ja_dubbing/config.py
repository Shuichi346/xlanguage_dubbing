#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定数・設定値の定義。
.envから環境変数を読み込み、全設定値を一元管理する。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    """環境変数を取得する。"""
    return os.getenv(key, default)


def _env_float(key: str, default: float) -> float:
    """環境変数をfloatで取得する。"""
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    """環境変数をintで取得する。"""
    v = os.getenv(key)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    """環境変数をboolで取得する。"""
    v = os.getenv(key)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


# =========================
# パス設定
# =========================

VIDEO_FOLDER = Path(_env("VIDEO_FOLDER", "./input_videos"))
TEMP_ROOT = Path(_env("TEMP_ROOT", "./temp"))
OUTPUT_SUFFIX = _env("OUTPUT_SUFFIX", "_jaDub.mp4")
KEEP_TEMP = _env_bool("KEEP_TEMP", True)

# =========================
# ASR共通
# =========================

ASR_ENGINE = _env("ASR_ENGINE", "whisper")  # "whisper" or "vibevoice"

# =========================
# whisper.cpp（CLIバイナリ + VAD）
# =========================

WHISPER_MODEL = _env("WHISPER_MODEL", "large-v3-turbo")
WHISPER_LANG = _env("WHISPER_LANG", "en")
VAD_MODEL = _env("VAD_MODEL", "silero-v6.2.0")

WHISPER_CPP_DIR = Path(_env("WHISPER_CPP_DIR", "./whisper.cpp"))

# whisper.cpp のネイティブサンプリングレート（16kHz 固定）
WHISPER_SAMPLE_RATE = 16000

# =========================
# VibeVoice-ASR
# =========================

VIBEVOICE_MODEL = _env("VIBEVOICE_MODEL", "mlx-community/VibeVoice-ASR-8bit")
VIBEVOICE_MAX_TOKENS = _env_int("VIBEVOICE_MAX_TOKENS", 32768)
VIBEVOICE_CONTEXT = _env("VIBEVOICE_CONTEXT", "")

# VibeVoice-ASR のネイティブサンプリングレート（24kHz）
VIBEVOICE_SAMPLE_RATE = 24000

VIBEVOICE_CHUNK_REFERENCE_AVAILABLE_GB = _env_float(
    "VIBEVOICE_CHUNK_REFERENCE_AVAILABLE_GB", 14.0
)
VIBEVOICE_CHUNK_REFERENCE_SECONDS = _env_int(
    "VIBEVOICE_CHUNK_REFERENCE_SECONDS", 600
)
VIBEVOICE_CHUNK_SAFETY_MARGIN = _env_float(
    "VIBEVOICE_CHUNK_SAFETY_MARGIN", 0.80
)
VIBEVOICE_CHUNK_MIN_SECONDS = _env_int("VIBEVOICE_CHUNK_MIN_SECONDS", 120)
VIBEVOICE_CHUNK_MAX_SECONDS = _env_int("VIBEVOICE_CHUNK_MAX_SECONDS", 1800)
VIBEVOICE_PREFILL_STEP_SIZE = _env_int("VIBEVOICE_PREFILL_STEP_SIZE", 512)
VIBEVOICE_MEMORY_LIMIT_RATIO = _env_float("VIBEVOICE_MEMORY_LIMIT_RATIO", 0.90)

# =========================
# pyannote.audio
# =========================

HF_AUTH_TOKEN = _env("HF_AUTH_TOKEN", "")
PYANNOTE_MODEL = _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")

# =========================
# CAT-Translate（英日翻訳）
# =========================

CAT_TRANSLATE_REPO = _env("CAT_TRANSLATE_REPO", "mradermacher/CAT-Translate-7b-GGUF")
CAT_TRANSLATE_FILE = _env("CAT_TRANSLATE_FILE", "CAT-Translate-7b.Q8_0.gguf")
CAT_TRANSLATE_N_GPU_LAYERS = _env_int("CAT_TRANSLATE_N_GPU_LAYERS", -1)
CAT_TRANSLATE_N_CTX = _env_int("CAT_TRANSLATE_N_CTX", 4096)
CAT_TRANSLATE_RETRIES = _env_int("CAT_TRANSLATE_RETRIES", 3)
CAT_TRANSLATE_RETRY_BACKOFF_SEC = _env_float("CAT_TRANSLATE_RETRY_BACKOFF_SEC", 1.5)
CAT_TRANSLATE_REPEAT_PENALTY = _env_float("CAT_TRANSLATE_REPEAT_PENALTY", 1.2)

# =========================
# OmniVoice（ボイスクローン + 再生時間制御）
# =========================

OMNIVOICE_MODEL = _env("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")
# MPS では float32 が安定（bfloat16/float16 は muffled voice の問題あり）
OMNIVOICE_DTYPE = _env("OMNIVOICE_DTYPE", "float32")
# Diffusion ステップ数（32 が高品質、16 で高速化）
OMNIVOICE_NUM_STEP = _env_int("OMNIVOICE_NUM_STEP", 32)
# Classifier-free guidance scale
OMNIVOICE_GUIDANCE_SCALE = _env_float("OMNIVOICE_GUIDANCE_SCALE", 2.0)
# 読み上げ速度（1.0 が標準、>1.0 で速く、<1.0 で遅く）
OMNIVOICE_SPEED = _env_float("OMNIVOICE_SPEED", 1.0)
# 元セグメントの長さに対する倍率（duration 制御に使用）
OMNIVOICE_DURATION_SCALE = _env_float("OMNIVOICE_DURATION_SCALE", 1.15)
# 参照音声の最小・最大・目標秒数
OMNIVOICE_REFERENCE_MIN_SEC = _env_float("OMNIVOICE_REFERENCE_MIN_SEC", 3.0)
OMNIVOICE_REFERENCE_MAX_SEC = _env_float("OMNIVOICE_REFERENCE_MAX_SEC", 15.0)
OMNIVOICE_REFERENCE_TARGET_SEC = _env_float("OMNIVOICE_REFERENCE_TARGET_SEC", 8.0)
# 品質バリデーション: 生成音声の長さが target_duration のこの倍率を超えたら再生成
OMNIVOICE_DURATION_TOLERANCE = _env_float("OMNIVOICE_DURATION_TOLERANCE", 0.5)
# 品質リトライ回数
OMNIVOICE_QUALITY_RETRIES = _env_int("OMNIVOICE_QUALITY_RETRIES", 2)
# OmniVoice のネイティブサンプリングレート（24kHz）
OMNIVOICE_SAMPLE_RATE = 24000

# =========================
# 音声設定（最終出力ミックス用）
# =========================

TTS_SAMPLE_RATE = _env_int("TTS_SAMPLE_RATE", 48000)
TTS_CHANNELS = _env_int("TTS_CHANNELS", 2)

# =========================
# セグメント設定
# =========================

MIN_SEGMENT_SEC = _env_float("MIN_SEGMENT_SEC", 0.35)

MERGE_MAX_SEC = _env_float("MERGE_MAX_SEC", 12.0)
MERGE_MAX_CHARS = _env_int("MERGE_MAX_CHARS", 320)
MERGE_GAP_SEC = _env_float("MERGE_GAP_SEC", 0.85)
MERGE_FORCE_IF_VERY_SHORT_SEC = _env_float("MERGE_FORCE_IF_VERY_SHORT_SEC", 0.70)

# =========================
# spaCy設定
# =========================

SPACY_MODEL = _env("SPACY_MODEL", "en_core_web_sm")

SPACY_CHUNK_MAX_SEC = _env_float("SPACY_CHUNK_MAX_SEC", 18.0)
SPACY_CHUNK_MAX_CHARS = _env_int("SPACY_CHUNK_MAX_CHARS", 1100)
SPACY_CHUNK_GAP_SEC = _env_float("SPACY_CHUNK_GAP_SEC", 1.0)

SPACY_UNIT_MAX_SENTENCES = _env_int("SPACY_UNIT_MAX_SENTENCES", 2)
SPACY_UNIT_MERGE_MAX_CHARS = _env_int("SPACY_UNIT_MERGE_MAX_CHARS", 300)
SPACY_UNIT_MERGE_MAX_GAP_SEC = _env_float("SPACY_UNIT_MERGE_MAX_GAP_SEC", 1.20)
SPACY_MIN_WEIGHT = _env_int("SPACY_MIN_WEIGHT", 1)

# =========================
# 翻訳異常検出
# =========================

OUTPUT_REPEAT_THRESHOLD = _env_int("OUTPUT_REPEAT_THRESHOLD", 3)
INPUT_REPEAT_THRESHOLD = _env_int("INPUT_REPEAT_THRESHOLD", 4)
INPUT_UNIQUE_RATIO_THRESHOLD = _env_float("INPUT_UNIQUE_RATIO_THRESHOLD", 0.3)

# =========================
# 音量ミックス
# =========================

ENGLISH_VOLUME = _env_float("ENGLISH_VOLUME", 0.10)
JAPANESE_VOLUME = _env_float("JAPANESE_VOLUME", 1.00)

# =========================
# 出力設定
# =========================

OUTPUT_SIZE = _env_int("OUTPUT_SIZE", 720)
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}
