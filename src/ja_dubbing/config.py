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
# TTSエンジン共通
# =========================

TTS_ENGINE = _env("TTS_ENGINE", "miotts")  # "miotts" or "kokoro" or "gptsovits" or "t5gemma"

# =========================
# MioTTS-Inference
# =========================

MIOTTS_API_URL = _env("MIOTTS_API_URL", "http://localhost:8001")
MIOTTS_HTTP_TIMEOUT = _env_float("MIOTTS_HTTP_TIMEOUT", 600.0)
MIOTTS_REFERENCE_MAX_SEC = _env_float("MIOTTS_REFERENCE_MAX_SEC", 20.0)
MIOTTS_MAX_TEXT_LENGTH = _env_int("MIOTTS_MAX_TEXT_LENGTH", 500)
# HTTPエラー等のネットワーク障害リトライ回数
MIOTTS_TTS_RETRIES = _env_int("MIOTTS_TTS_RETRIES", 2)
# 品質バリデーション失敗時の再生成リトライ回数（HTTP リトライとは独立）
MIOTTS_QUALITY_RETRIES = _env_int("MIOTTS_QUALITY_RETRIES", 2)

# MioTTS LLMバックエンド
MIOTTS_LLM_PORT = _env_int("MIOTTS_LLM_PORT", 8000)
MIOTTS_LLM_MODEL = _env(
    "MIOTTS_LLM_MODEL", "hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf"
)
MIOTTS_DEVICE = _env("MIOTTS_DEVICE", "mps")
MIOTTS_CODEC_MODEL = _env("MIOTTS_CODEC_MODEL", "Aratako/MioCodec-25Hz-44.1kHz-v2")
MIOTTS_INFERENCE_DIR = _env("MIOTTS_INFERENCE_DIR", "./MioTTS-Inference")

# MioTTS LLMサンプリングパラメータ（音声品質チューニング）
MIOTTS_LLM_TEMPERATURE = _env_float("MIOTTS_LLM_TEMPERATURE", 0.5)
MIOTTS_LLM_TOP_P = _env_float("MIOTTS_LLM_TOP_P", 1.0)
MIOTTS_LLM_MAX_TOKENS = _env_int("MIOTTS_LLM_MAX_TOKENS", 700)
MIOTTS_LLM_REPETITION_PENALTY = _env_float("MIOTTS_LLM_REPETITION_PENALTY", 1.1)
MIOTTS_LLM_PRESENCE_PENALTY = _env_float("MIOTTS_LLM_PRESENCE_PENALTY", 0.0)
MIOTTS_LLM_FREQUENCY_PENALTY = _env_float("MIOTTS_LLM_FREQUENCY_PENALTY", 0.3)

# MioTTS 品質バリデーション（生成音声の長さ比率チェック）
# 日本語1文字あたりの想定発話秒数の下限・上限
MIOTTS_DURATION_PER_CHAR_MIN = _env_float("MIOTTS_DURATION_PER_CHAR_MIN", 0.05)
MIOTTS_DURATION_PER_CHAR_MAX = _env_float("MIOTTS_DURATION_PER_CHAR_MAX", 0.5)
# 品質バリデーションの最小テキスト長（これ未満はバリデーションをスキップ）
MIOTTS_VALIDATION_MIN_CHARS = _env_int("MIOTTS_VALIDATION_MIN_CHARS", 4)

# =========================
# Kokoro TTS
# =========================

KOKORO_MODEL = _env("KOKORO_MODEL", "kokoro")
KOKORO_VOICE = _env("KOKORO_VOICE", "jf_alpha")
KOKORO_SPEED = _env_float("KOKORO_SPEED", 1.0)

# =========================
# T5Gemma-TTS（ボイスクローン + 再生時間制御）
# =========================

T5GEMMA_MODEL_DIR = _env("T5GEMMA_MODEL_DIR", "Aratako/T5Gemma-TTS-2b-2b")
T5GEMMA_XCODEC2_MODEL = _env(
    "T5GEMMA_XCODEC2_MODEL", "NandemoGHS/Anime-XCodec2-44.1kHz-v2"
)
T5GEMMA_TOP_K = _env_int("T5GEMMA_TOP_K", 30)
T5GEMMA_TOP_P = _env_float("T5GEMMA_TOP_P", 0.9)
T5GEMMA_TEMPERATURE = _env_float("T5GEMMA_TEMPERATURE", 0.8)
T5GEMMA_SEED = _env_int("T5GEMMA_SEED", 1)
T5GEMMA_STOP_REPETITION = _env_int("T5GEMMA_STOP_REPETITION", 3)
T5GEMMA_DURATION_SCALE = _env_float("T5GEMMA_DURATION_SCALE", 1.15)
T5GEMMA_REFERENCE_MAX_SEC = _env_float("T5GEMMA_REFERENCE_MAX_SEC", 15.0)
T5GEMMA_REFERENCE_MIN_SEC = _env_float("T5GEMMA_REFERENCE_MIN_SEC", 3.0)
T5GEMMA_REFERENCE_TARGET_SEC = _env_float("T5GEMMA_REFERENCE_TARGET_SEC", 8.0)
T5GEMMA_CPU_CODEC = _env_bool("T5GEMMA_CPU_CODEC", True)
T5GEMMA_DURATION_TOLERANCE = _env_float("T5GEMMA_DURATION_TOLERANCE", 0.5)
T5GEMMA_QUALITY_RETRIES = _env_int("T5GEMMA_QUALITY_RETRIES", 2)

# =========================
# GPT-SoVITS（V2ProPlus ゼロショットボイスクローン）
# =========================

GPTSOVITS_API_URL = _env("GPTSOVITS_API_URL", "http://127.0.0.1:9880")
GPTSOVITS_HTTP_TIMEOUT = _env_float("GPTSOVITS_HTTP_TIMEOUT", 300.0)
GPTSOVITS_TTS_RETRIES = _env_int("GPTSOVITS_TTS_RETRIES", 2)
GPTSOVITS_DIR = _env("GPTSOVITS_DIR", "./GPT-SoVITS")
GPTSOVITS_CONDA_ENV = _env("GPTSOVITS_CONDA_ENV", "gptsovits")

# GPT-SoVITS API パラメータ
GPTSOVITS_TEXT_LANG = _env("GPTSOVITS_TEXT_LANG", "ja")
GPTSOVITS_PROMPT_LANG = _env("GPTSOVITS_PROMPT_LANG", "en")
GPTSOVITS_TOP_K = _env_int("GPTSOVITS_TOP_K", 15)
GPTSOVITS_TOP_P = _env_float("GPTSOVITS_TOP_P", 1.0)
GPTSOVITS_TEMPERATURE = _env_float("GPTSOVITS_TEMPERATURE", 1.0)
GPTSOVITS_TEXT_SPLIT_METHOD = _env("GPTSOVITS_TEXT_SPLIT_METHOD", "cut5")
GPTSOVITS_BATCH_SIZE = _env_int("GPTSOVITS_BATCH_SIZE", 1)
GPTSOVITS_SPEED_FACTOR = _env_float("GPTSOVITS_SPEED_FACTOR", 1.0)
GPTSOVITS_MEDIA_TYPE = _env("GPTSOVITS_MEDIA_TYPE", "wav")
GPTSOVITS_REPETITION_PENALTY = _env_float("GPTSOVITS_REPETITION_PENALTY", 1.35)

# GPT-SoVITS リファレンス音声設定
# ゼロショット参照音声: 3〜10秒（推奨5秒）
GPTSOVITS_REFERENCE_MIN_SEC = _env_float("GPTSOVITS_REFERENCE_MIN_SEC", 3.0)
GPTSOVITS_REFERENCE_MAX_SEC = _env_float("GPTSOVITS_REFERENCE_MAX_SEC", 10.0)
GPTSOVITS_REFERENCE_TARGET_SEC = _env_float("GPTSOVITS_REFERENCE_TARGET_SEC", 5.0)

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

# 翻訳「出力」（日本語）の異常検出: 同一フレーズがN回以上繰り返されていたら異常とみなす
OUTPUT_REPEAT_THRESHOLD = _env_int("OUTPUT_REPEAT_THRESHOLD", 3)

# 翻訳「入力」（英語）の繰り返し検出
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
