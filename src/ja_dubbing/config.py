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

# whisper.cpp のインストール先（setup_whisper.sh がクローンする場所）
WHISPER_CPP_DIR = Path(_env("WHISPER_CPP_DIR", "./whisper.cpp"))

# =========================
# VibeVoice-ASR
# =========================

VIBEVOICE_MODEL = _env("VIBEVOICE_MODEL", "mlx-community/VibeVoice-ASR-8bit")
VIBEVOICE_MAX_TOKENS = _env_int("VIBEVOICE_MAX_TOKENS", 32768)
VIBEVOICE_CONTEXT = _env("VIBEVOICE_CONTEXT", "")

# VibeVoice エンコーダチャンク設定（メモリ最適化）
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
PYANNOTE_MODEL = _env("PYANNOTE_MODEL", "pyannote/speaker-diarization-3.1")

# =========================
# plamo-translate-cli (PLaMo翻訳)
# =========================

PLAMO_TRANSLATE_PRECISION = _env("PLAMO_TRANSLATE_PRECISION", "8bit")
PLAMO_TRANSLATE_RETRIES = _env_int("PLAMO_TRANSLATE_RETRIES", 3)
PLAMO_TRANSLATE_RETRY_BACKOFF_SEC = _env_float("PLAMO_TRANSLATE_RETRY_BACKOFF_SEC", 1.5)

# =========================
# MioTTS-Inference
# =========================

MIOTTS_API_URL = _env("MIOTTS_API_URL", "http://localhost:8001")
MIOTTS_HTTP_TIMEOUT = _env_float("MIOTTS_HTTP_TIMEOUT", 600.0)
MIOTTS_REFERENCE_MAX_SEC = _env_float("MIOTTS_REFERENCE_MAX_SEC", 20.0)
MIOTTS_MAX_TEXT_LENGTH = _env_int("MIOTTS_MAX_TEXT_LENGTH", 500)
MIOTTS_TTS_RETRIES = _env_int("MIOTTS_TTS_RETRIES", 2)

# MioTTS LLMバックエンド
MIOTTS_LLM_PORT = _env_int("MIOTTS_LLM_PORT", 8000)
MIOTTS_LLM_MODEL = _env(
    "MIOTTS_LLM_MODEL", "hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf"
)
MIOTTS_DEVICE = _env("MIOTTS_DEVICE", "mps")
MIOTTS_CODEC_MODEL = _env("MIOTTS_CODEC_MODEL", "Aratako/MioCodec-25Hz-44.1kHz-v2")
MIOTTS_INFERENCE_DIR = _env("MIOTTS_INFERENCE_DIR", "./MioTTS-Inference")

# =========================
# 音声設定
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

# 翻訳「出力」（日本語）の異常検出
GLITCH_PHRASE = _env("GLITCH_PHRASE", "を徹底的に")
GLITCH_MIN_REPEAT = _env_int("GLITCH_MIN_REPEAT", 3)

# 翻訳「入力」（英語）の繰り返し検出
INPUT_REPEAT_THRESHOLD = _env_int("INPUT_REPEAT_THRESHOLD", 4)
INPUT_UNIQUE_RATIO_THRESHOLD = _env_float("INPUT_UNIQUE_RATIO_THRESHOLD", 0.3)

# 翻訳API1回あたりのタイムアウト（秒）
TRANSLATE_TIMEOUT_SEC = _env_float("TRANSLATE_TIMEOUT_SEC", 120.0)

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
