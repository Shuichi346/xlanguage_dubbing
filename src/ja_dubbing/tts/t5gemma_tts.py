#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5Gemma-TTS による音声合成処理。
ボイスクローンと再生時間制御をサポートし、外部サーバー不要で動作する。
"""

from __future__ import annotations

import gc
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torchaudio

from ja_dubbing.audio.ffmpeg import ffprobe_duration_sec
from ja_dubbing.config import (
    MIN_SEGMENT_SEC,
    T5GEMMA_CPU_CODEC,
    T5GEMMA_DURATION_SCALE,
    T5GEMMA_DURATION_TOLERANCE,
    T5GEMMA_MODEL_DIR,
    T5GEMMA_QUALITY_RETRIES,
    T5GEMMA_SEED,
    T5GEMMA_STOP_REPETITION,
    T5GEMMA_TEMPERATURE,
    T5GEMMA_TOP_K,
    T5GEMMA_TOP_P,
    T5GEMMA_XCODEC2_MODEL,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
)
from ja_dubbing.core.models import Segment, TtsMeta
from ja_dubbing.tts.reference import SpeakerReferenceCache
from ja_dubbing.utils import (
    PipelineError,
    ensure_dir,
    force_memory_cleanup,
    print_step,
    run_cmd,
    sanitize_text_for_tts,
    which_or_raise,
)

_T5GEMMA_MODEL = None
_TEXT_TOKENIZER = None
_AUDIO_TOKENIZER = None
_MODEL_CONFIG = None
# モデルロードが回復不能な形で失敗した場合にセットされるフラグ
_MODEL_LOAD_FAILED = False

_REPLACE_MAP = {
    r"\t": "",
    r"\[n\]": "",
    r" ": "",
    r"　": "",
    r"[;▼♀♂《》≪≫①②③④⑤⑥]": "",
    r"[\u02d7\u2010-\u2015\u2043\u2212\u23af\u23e4\u2500\u2501\u2e3a\u2e3b]": "",
    r"[\uff5e\u301C]": "ー",
    r"？": "?",
    r"！": "!",
    r"[●◯〇]": "○",
    r"♥": "♡",
}
_FULLWIDTH_ALPHA_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(
            list(range(0xFF21, 0xFF3B)) + list(range(0xFF41, 0xFF5B)),
            list(range(0x41, 0x5B)) + list(range(0x61, 0x7B)),
        )
    }
)
_HALFWIDTH_KATAKANA_CHARS = (
    "ｦｧｨｩｪｫｬｭｮｯｰｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉ"
    "ﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ"
)
_FULLWIDTH_KATAKANA_CHARS = (
    "ヲァィゥェォャュョッーアイウエオカキクケコサシスセソタチツテトナニヌネノ"
    "ハヒフヘホマミムメモヤユヨラリルレロワン"
)
_HALFWIDTH_KATAKANA_TO_FULLWIDTH = str.maketrans(
    _HALFWIDTH_KATAKANA_CHARS, _FULLWIDTH_KATAKANA_CHARS
)
_FULLWIDTH_DIGITS_TO_HALFWIDTH = str.maketrans(
    {
        chr(full): chr(half)
        for full, half in zip(range(0xFF10, 0xFF1A), range(0x30, 0x3A))
    }
)
_SPP_DEFAULT = {"en": 0.085, "ja": 0.10, "zh": 0.27, "other": 0.11}
_SPP_MINMAX = {
    "en": (0.06, 0.12),
    "ja": (0.07, 0.15),
    "zh": (0.18, 0.36),
    "other": (0.07, 0.18),
}
_MIN_DURATION_SEC = 0.5
_MAX_DURATION_SEC = 120.0
_G2P_EN = None


class TTSQualityError(Exception):
    """TTS 品質バリデーション失敗を示す例外。"""


class AudioTokenizer:
    """XCodec2 ベースの音声トークナイザ。"""

    def __init__(
        self,
        device: Any,
        backend: str = "xcodec2",
        model_name: Optional[str] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        if backend != "xcodec2":
            raise ValueError(
                f"Only xcodec2 backend is supported now (got {backend})."
            )

        try:
            from huggingface_hub import hf_hub_download
            from safetensors import safe_open
            from xcodec2.configuration_bigcodec import BigCodecConfig
            from xcodec2.modeling_xcodec2 import XCodec2Model
        except ImportError as exc:
            raise PipelineError(
                "T5Gemma-TTS の依存が不足しています。\n"
                "  uv sync を実行してください。"
            ) from exc

        self._device = torch.device(device)
        model_id = model_name or T5GEMMA_XCODEC2_MODEL
        ckpt_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        ckpt: dict[str, torch.Tensor] = {}
        with safe_open(ckpt_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                ckpt[key.replace(".beta", ".bias")] = handle.get_tensor(key)

        codec_config = BigCodecConfig.from_pretrained(model_id)
        self.codec = XCodec2Model.from_pretrained(
            None, config=codec_config, state_dict=ckpt
        )
        self.codec.eval()
        self.codec.to(self._device)

        self.sample_rate = int(
            sample_rate or getattr(self.codec.config, "sample_rate", 44100)
        )
        encode_sr = getattr(self.codec.config, "encoder_sample_rate", None)
        if encode_sr is None:
            feature_extractor = getattr(self.codec, "feature_extractor", None)
            encode_sr = (
                getattr(feature_extractor, "sampling_rate", None)
                if feature_extractor is not None
                else None
            )
        self.encode_sample_rate = int(encode_sr) if encode_sr else 16000
        self.channels = 1

    @property
    def device(self) -> torch.device:
        """トークナイザが動作するデバイスを返す。"""
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """音声波形を XCodec2 のコード列へ変換する。"""
        wav = wav.to(self.device)
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        codes = self.codec.encode_code(
            input_waveform=wav,
            sample_rate=self.encode_sample_rate,
        )
        if codes.ndim == 2:
            codes = codes.unsqueeze(-1)
        return codes.permute(0, 2, 1).contiguous().to(dtype=torch.long)

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        """XCodec2 のコード列を音声波形へ戻す。"""
        codes = frames
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        codes = codes.long().to(self.device)
        return self.codec.decode_code(codes)


def _torch_ge_29() -> bool:
    """PyTorch 2.9 以上か判定する。"""
    try:
        version = torch.__version__.split("+")[0]
        major, minor = map(int, version.split(".")[:2])
        return (major, minor) >= (2, 9)
    except Exception:
        return False


def _seed_everything(seed: int) -> None:
    """乱数シードを固定する。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _get_device() -> str:
    """推論デバイスを返す。"""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_audio(
    audio_path: str | Path,
    offset: int = -1,
    num_frames: int = -1,
) -> tuple[torch.Tensor, int]:
    """音声ファイルを読み込む。"""
    audio_path = str(audio_path)
    if _torch_ge_29():
        import soundfile as sf

        if offset != -1 and num_frames != -1:
            wav, sample_rate = sf.read(
                audio_path,
                start=offset,
                stop=offset + num_frames,
                dtype="float32",
            )
        else:
            wav, sample_rate = sf.read(audio_path, dtype="float32")
        wav_tensor = torch.from_numpy(wav)
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        elif wav_tensor.ndim == 2:
            wav_tensor = wav_tensor.T
        return wav_tensor, int(sample_rate)

    if offset != -1 and num_frames != -1:
        wav, sample_rate = torchaudio.load(
            audio_path,
            frame_offset=offset,
            num_frames=num_frames,
        )
    else:
        wav, sample_rate = torchaudio.load(audio_path)
    return wav, int(sample_rate)


def _tokenize_audio(
    tokenizer: AudioTokenizer,
    audio_path: str | Path,
    offset: int = -1,
    num_frames: int = -1,
) -> torch.Tensor:
    """音声ファイルを XCodec2 コード列へ変換する。"""
    wav, sample_rate = _load_audio(audio_path, offset, num_frames)
    target_sr = getattr(tokenizer, "encode_sample_rate", tokenizer.sample_rate)
    if sample_rate != target_sr:
        wav = wav.to(dtype=torch.float32)
        if _torch_ge_29():
            import torchaudio.functional as F

            wav = F.resample(wav, sample_rate, target_sr)
        else:
            wav = torchaudio.transforms.Resample(sample_rate, target_sr)(wav)
    if wav.shape[0] == 2:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0)
    with torch.no_grad():
        return tokenizer.encode(wav)


def _normalize_japanese_text(text: str) -> str:
    """日本語テキストを T5Gemma-TTS 向けに正規化する。"""
    for pattern, replacement in _REPLACE_MAP.items():
        text = re.sub(pattern, replacement, text)
    text = text.translate(_FULLWIDTH_ALPHA_TO_HALFWIDTH)
    text = text.translate(_FULLWIDTH_DIGITS_TO_HALFWIDTH)
    text = text.translate(_HALFWIDTH_KATAKANA_TO_FULLWIDTH)
    return re.sub(r"…{3,}", "……", text)


def _safe_detect_language(text: str) -> str:
    """大まかな言語コードを推定する。"""
    text = (text or "").strip()
    if not text:
        return "other"

    try:
        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = 0
        try:
            lang = detect(text)
            if lang.startswith("ja"):
                return "ja"
            if lang.startswith("zh") or lang == "yue":
                return "zh"
            if lang.startswith("en"):
                return "en"
        except LangDetectException:
            pass
    except ImportError:
        pass

    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def detect_language(text: str) -> str:
    """推定した言語コードを返す。"""
    return _safe_detect_language(text)


def normalize_text_with_lang(
    text: str,
    lang: Optional[str],
) -> Tuple[str, Optional[str]]:
    """言語に応じてテキストを正規化する。"""
    resolved_lang = lang.lower() if isinstance(lang, str) else None
    if not text:
        return text, resolved_lang
    if resolved_lang is None:
        resolved_lang = detect_language(text)
    if resolved_lang.startswith("ja"):
        return _normalize_japanese_text(text), resolved_lang
    return text, resolved_lang


def _canonicalize_lang(lang: Optional[str]) -> Optional[str]:
    """言語コードを en / ja / zh 系へ寄せる。"""
    if not lang:
        return None
    lang = lang.lower()
    if lang.startswith("ja"):
        return "ja"
    if lang.startswith("zh") or lang == "yue":
        return "zh"
    if lang.startswith("en"):
        return "en"
    return lang


def _phoneme_count_en(text: str) -> int:
    """英語の概算フォネーム数を返す。"""
    global _G2P_EN
    try:
        from g2p_en import G2p
    except ImportError:
        return max(len(text), 1)
    if _G2P_EN is None:
        _G2P_EN = G2p()
    phonemes = _G2P_EN(text)
    return len(
        [
            token
            for token in phonemes
            if token and token not in {" ", "<pad>", "<s>", "</s>", "<unk>"}
        ]
    )


def _phoneme_count_ja(text: str) -> int:
    """日本語の概算フォネーム数を返す。"""
    try:
        import pyopenjtalk
    except ImportError:
        return max(len(text), 1)
    phonemes = pyopenjtalk.g2p(text)
    return len(
        [token for token in phonemes.split(" ") if token and token not in {"pau", "sil"}]
    )


def _phoneme_count_zh(text: str) -> int:
    """中国語の概算フォネーム数を返す。"""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return max(len(text), 1)
    syllables = lazy_pinyin(text, style=Style.NORMAL, neutral_tone_with_five=True)
    return len([syllable for syllable in syllables if re.search(r"[A-Za-z]", syllable)])


def _phoneme_count(text: str, lang: str) -> int:
    """言語別の概算フォネーム数を返す。"""
    if lang == "en":
        return _phoneme_count_en(text)
    if lang == "ja":
        return _phoneme_count_ja(text)
    if lang == "zh":
        return _phoneme_count_zh(text)
    return max(len(text), 1)


def _punctuation_bonus_sec(text: str) -> float:
    """句読点ぶんのポーズ時間を概算する。"""
    text = (text or "").strip()
    major = len(re.findall(r"[.!?。！？]", text))
    minor = len(re.findall(r"[、，,;；:]", text))
    if text and text[-1] in ".!?。！？":
        major = max(0, major - 1)
    ellipsis = len(re.findall(r"(…|\.\.\.)", text))
    dash_pause = len(re.findall(r"(—|--)", text))
    bonus = major * 0.40 + minor * 0.20 + ellipsis * 1.0 + dash_pause * 0.12
    return min(10.0, bonus)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    """値を範囲内に収める。"""
    lower, upper = bounds
    return max(lower, min(upper, value))


def estimate_duration(
    target_text: str,
    reference_speech: Optional[str | Path] = None,
    reference_transcript: Optional[str] = None,
    target_lang: Optional[str] = None,
    reference_lang: Optional[str] = None,
) -> float:
    """テキストと参照音声から目安の長さを推定する。"""
    target_text = target_text or ""
    ref_has_audio = bool(reference_speech) and Path(str(reference_speech)).is_file()

    target_lang = _canonicalize_lang(target_lang) or (
        _safe_detect_language(target_text) if target_text else "en"
    )
    target_phonemes = max(_phoneme_count(target_text, target_lang), 1)

    seconds_per_phoneme = _SPP_DEFAULT.get(target_lang, _SPP_DEFAULT["other"])

    if ref_has_audio:
        try:
            if _torch_ge_29():
                import soundfile as sf

                info = sf.info(str(reference_speech))
                audio_duration = info.frames / info.samplerate
            else:
                info = torchaudio.info(str(reference_speech))
                audio_duration = info.num_frames / info.sample_rate
        except Exception:
            audio_duration = None

        if audio_duration and audio_duration > 0:
            ref_text = reference_transcript or target_text
            ref_lang = _canonicalize_lang(reference_lang) or _safe_detect_language(
                ref_text
            )
            ref_phonemes = max(_phoneme_count(ref_text, ref_lang), 1)
            seconds_per_phoneme = audio_duration / ref_phonemes
            seconds_per_phoneme = _clamp(
                seconds_per_phoneme,
                _SPP_MINMAX.get(ref_lang, _SPP_MINMAX["other"]),
            )

    punct_bonus = _punctuation_bonus_sec(target_text) * (0.3 if ref_has_audio else 1.0)
    duration = target_phonemes * seconds_per_phoneme + punct_bonus
    return max(_MIN_DURATION_SEC, min(duration, _MAX_DURATION_SEC))


def _get_model_dtype(device: str) -> torch.dtype:
    """デバイスに応じたモデル dtype を返す。"""
    if device == "cpu":
        return torch.float32
    return torch.bfloat16


def _patch_tied_weights_keys(model: Any) -> None:
    """transformers 5.x 互換パッチ: all_tied_weights_keys 属性を補完する。

    T5Gemma-TTS のリモートモデルコードが transformers 5.x の
    all_tied_weights_keys 属性に未対応の場合に、空の辞書を設定して
    AttributeError を回避する。
    """
    if not hasattr(model, "all_tied_weights_keys"):
        model.all_tied_weights_keys = {}


def _load_t5gemma_model():
    """T5Gemma-TTS 一式を遅延ロードする。"""
    global _AUDIO_TOKENIZER, _MODEL_CONFIG, _MODEL_LOAD_FAILED
    global _T5GEMMA_MODEL, _TEXT_TOKENIZER

    if _T5GEMMA_MODEL is not None:
        return _T5GEMMA_MODEL, _TEXT_TOKENIZER, _AUDIO_TOKENIZER, _MODEL_CONFIG

    if _MODEL_LOAD_FAILED:
        raise PipelineError(
            "T5Gemma-TTS モデルの初期化に既に失敗しています。"
            "transformers のバージョンを確認してください: "
            "uv pip install 'transformers>=4.57.0,<5.0.0'"
        )

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        _MODEL_LOAD_FAILED = True
        raise PipelineError(
            "transformers がインストールされていません。\n"
            "  uv sync を実行してください。"
        ) from exc

    device = _get_device()
    model_dtype = _get_model_dtype(device)

    print_step(f"  T5Gemma-TTS モデル初期化中: {T5GEMMA_MODEL_DIR} ({device})")
    try:
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                T5GEMMA_MODEL_DIR,
                trust_remote_code=True,
                torch_dtype=model_dtype,
            )
        except AttributeError as attr_err:
            if "all_tied_weights_keys" in str(attr_err):
                # transformers 5.x との非互換: モンキーパッチで回避
                print_step(
                    "  transformers 5.x 互換パッチを適用中..."
                )
                _apply_global_tied_weights_patch()
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    T5GEMMA_MODEL_DIR,
                    trust_remote_code=True,
                    torch_dtype=model_dtype,
                )
            else:
                raise
        except TypeError:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                T5GEMMA_MODEL_DIR,
                trust_remote_code=True,
                dtype=model_dtype,
            )
    except PipelineError:
        _MODEL_LOAD_FAILED = True
        raise
    except Exception as exc:
        _MODEL_LOAD_FAILED = True
        raise PipelineError(
            f"T5Gemma-TTS モデルの読み込みに失敗しました: {exc}\n"
            "transformers のバージョンを確認してください。T5Gemma-TTS は "
            "transformers 4.57.x が必要です:\n"
            "  uv pip install 'transformers>=4.57.0,<5.0.0'"
        ) from exc

    _patch_tied_weights_keys(model)

    if not getattr(model, "hf_device_map", None):
        model = model.to(device)
    model.eval()

    cfg = model.config
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(
        cfg, "t5gemma_model_name", None
    )
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    codec_device = "cpu" if T5GEMMA_CPU_CODEC else device
    audio_tokenizer = AudioTokenizer(
        device=codec_device,
        backend="xcodec2",
        model_name=T5GEMMA_XCODEC2_MODEL,
    )

    _T5GEMMA_MODEL = model
    _TEXT_TOKENIZER = text_tokenizer
    _AUDIO_TOKENIZER = audio_tokenizer
    _MODEL_CONFIG = cfg

    print_step(
        f"  T5Gemma-TTS モデル初期化完了: codec={audio_tokenizer.sample_rate}Hz "
        f"codec_device={codec_device}"
    )
    return _T5GEMMA_MODEL, _TEXT_TOKENIZER, _AUDIO_TOKENIZER, _MODEL_CONFIG


def _apply_global_tied_weights_patch() -> None:
    """torch.nn.Module の __getattr__ にパッチを当てて all_tied_weights_keys を補完する。

    transformers 5.x が from_pretrained 内部で all_tied_weights_keys にアクセスするが、
    T5Gemma-TTS のリモートモデルコードがこの属性を定義していない場合の回避策。
    """
    import torch.nn

    _orig_getattr = torch.nn.Module.__getattr__

    def _patched_getattr(self, name):
        if name == "all_tied_weights_keys":
            return {}
        return _orig_getattr(self, name)

    torch.nn.Module.__getattr__ = _patched_getattr


def _encode_text(text_tokenizer, text: str) -> list[int]:
    """テキストを tokenizer で ID 列へ変換する。"""
    return text_tokenizer.encode(text.strip(), add_special_tokens=False)


def _strip_special_audio_tokens(
    frames: torch.Tensor,
    sep_token: Optional[int],
    eos_token: Optional[int],
) -> torch.Tensor:
    """区切りトークンと終端トークンを除去する。"""
    if frames.ndim != 3 or frames.shape[0] != 1:
        raise PipelineError(f"想定外のフレーム形状です: {tuple(frames.shape)}")

    sample_frames = frames[0, 0]
    mask = torch.ones_like(sample_frames, dtype=torch.bool)
    if sep_token is not None:
        mask &= sample_frames.ne(sep_token)
    if eos_token is not None:
        mask &= sample_frames.ne(eos_token)
    cleaned = sample_frames[mask]
    if cleaned.numel() == 0:
        raise PipelineError("生成された音声トークンが空です。")
    return cleaned.view(1, 1, -1)


def _build_target_length(
    cfg,
    prompt_frames: int,
    codec_sr: int,
    target_duration: float,
) -> torch.LongTensor:
    """推論時の目標トークン長を組み立てる。"""
    effective_delay_inc = 0
    if getattr(cfg, "parallel_pattern", 0) != 0:
        length = int(prompt_frames + codec_sr * target_duration + 2)
    else:
        length = int(prompt_frames + codec_sr * target_duration + effective_delay_inc)
    return torch.LongTensor([max(length, prompt_frames + 1)])


def _inference_one_sample(
    model,
    cfg,
    text_tokenizer,
    audio_tokenizer: AudioTokenizer,
    reference_speech: Optional[Path],
    reference_text: str,
    target_text: str,
    target_duration: Optional[float],
    seed: int,
) -> tuple[torch.Tensor, int]:
    """T5Gemma-TTS で 1 サンプルを推論する。"""
    device = _get_device()
    _seed_everything(seed)

    n_codebooks = int(getattr(cfg, "n_codebooks", 1))
    if n_codebooks != 1:
        raise PipelineError("T5Gemma-TTS の XCodec2 backend は 1 codebook 前提です。")

    codec_sr = int(getattr(cfg, "encodec_sr", 50))
    y_sep_token = getattr(cfg, "y_sep_token", None)
    x_sep_token = getattr(cfg, "x_sep_token", None)
    eos_token = getattr(cfg, "eos", getattr(cfg, "eog", None))
    add_eos_token = getattr(cfg, "add_eos_to_text", 0)
    add_bos_token = getattr(cfg, "add_bos_to_text", 0)

    has_reference_audio = reference_speech is not None and reference_speech.exists()
    if has_reference_audio:
        encoded_frames = _tokenize_audio(audio_tokenizer, reference_speech)
    else:
        encoded_frames = torch.empty(
            (1, n_codebooks, 0),
            dtype=torch.long,
            device=audio_tokenizer.device,
        )

    if encoded_frames.ndim == 2:
        encoded_frames = encoded_frames.unsqueeze(0)
    if encoded_frames.ndim != 3 or encoded_frames.shape[0] != 1:
        raise PipelineError(f"参照音声フレーム形状が不正です: {tuple(encoded_frames.shape)}")
    if encoded_frames.shape[2] == 1:
        encoded_frames = encoded_frames.transpose(1, 2).contiguous()
    if encoded_frames.shape[1] != 1:
        raise PipelineError(f"参照音声 codebook 数が不正です: {tuple(encoded_frames.shape)}")

    if y_sep_token is not None and has_reference_audio and encoded_frames.shape[2] > 0:
        encoded_frames = torch.cat(
            [
                encoded_frames,
                torch.full(
                    (1, n_codebooks, 1),
                    y_sep_token,
                    dtype=torch.long,
                    device=encoded_frames.device,
                ),
            ],
            dim=2,
        )

    original_audio = encoded_frames.transpose(2, 1).contiguous()
    prompt_frames = original_audio.shape[1]

    normalized_target_text, target_lang = normalize_text_with_lang(target_text, "ja")
    normalized_reference_text, reference_lang = normalize_text_with_lang(
        reference_text or "",
        None,
    )

    text_tokens = _encode_text(text_tokenizer, normalized_target_text)
    if normalized_reference_text:
        prefix_tokens = _encode_text(text_tokenizer, normalized_reference_text)
        if x_sep_token is not None:
            text_tokens = prefix_tokens + [x_sep_token] + text_tokens
        else:
            text_tokens = prefix_tokens + text_tokens

    if add_eos_token:
        text_tokens.append(add_eos_token)
    if add_bos_token:
        text_tokens = [add_bos_token] + text_tokens

    text_tokens_tensor = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens_tensor.shape[-1]])

    target_generation_length = (
        float(target_duration)
        if target_duration is not None
        else estimate_duration(
            target_text=normalized_target_text,
            reference_speech=reference_speech,
            reference_transcript=normalized_reference_text,
            target_lang=target_lang,
            reference_lang=reference_lang,
        )
    )
    tgt_y_lens = _build_target_length(
        cfg,
        prompt_frames=prompt_frames,
        codec_sr=codec_sr,
        target_duration=target_generation_length,
    )

    started_at = time.time()
    _concat_frames, gen_frames = model.inference_tts(
        text_tokens_tensor.to(device),
        text_tokens_lens.to(device),
        original_audio.to(device),
        tgt_y_lens=tgt_y_lens.to(device),
        top_k=T5GEMMA_TOP_K,
        top_p=T5GEMMA_TOP_P,
        min_p=0.0,
        temperature=T5GEMMA_TEMPERATURE,
        stop_repetition=T5GEMMA_STOP_REPETITION,
        silence_tokens=[],
        prompt_frames=prompt_frames,
        num_samples=1,
    )
    elapsed = time.time() - started_at

    clean_frames = _strip_special_audio_tokens(gen_frames, y_sep_token, eos_token)
    waveform = audio_tokenizer.decode(clean_frames)

    generated_tokens = clean_frames.shape[-1]
    audio_duration = generated_tokens / codec_sr
    tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0.0
    print_step(
        f"    T5Gemma 推論完了: {generated_tokens} token / {audio_duration:.2f}秒 "
        f"({tokens_per_sec:.2f} token/s)"
    )
    return waveform, audio_tokenizer.sample_rate


def _save_audio(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
    """音声波形を WAV として保存する。"""
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PipelineError(
            "soundfile がインストールされていません。\n"
            "  uv sync を実行してください。"
        ) from exc

    wav_np = waveform.squeeze().detach().cpu().numpy()
    sf.write(str(path), wav_np, sample_rate)


def t5gemma_synthesize(
    text_ja: str,
    out_wav: Path,
    reference_speech: Optional[Path] = None,
    reference_text: str = "",
    target_duration: Optional[float] = None,
    seed: int = T5GEMMA_SEED,
) -> None:
    """T5Gemma-TTS で音声を合成する。"""
    ensure_dir(out_wav.parent)
    model, text_tokenizer, audio_tokenizer, cfg = _load_t5gemma_model()
    waveform, sample_rate = _inference_one_sample(
        model=model,
        cfg=cfg,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        reference_speech=reference_speech,
        reference_text=reference_text,
        target_text=text_ja,
        target_duration=target_duration,
        seed=seed,
    )
    _save_audio(out_wav, waveform, sample_rate)


def _validate_t5gemma_quality(
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
    if ratio > T5GEMMA_DURATION_TOLERANCE:
        raise TTSQualityError(
            f"生成音声の長さ({audio_duration_sec:.1f}秒)が "
            f"目標({target_duration_sec:.1f}秒)から "
            f"{ratio * 100:.0f}%乖離（許容: {T5GEMMA_DURATION_TOLERANCE * 100:.0f}%）"
        )


def _convert_to_flac(in_wav: Path, out_flac: Path) -> None:
    """WAV をプロジェクト標準の FLAC に変換する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_flac.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(in_wav),
        "-ac",
        str(TTS_CHANNELS),
        "-ar",
        str(TTS_SAMPLE_RATE),
        "-c:a",
        "flac",
        str(out_flac),
    ]
    run_cmd(cmd)


def _synthesize_with_quality_retry(
    text: str,
    tmp_wav: Path,
    reference_speech: Path,
    reference_text: str,
    target_duration: float,
) -> None:
    """品質チェック付きで T5Gemma-TTS を実行する。"""
    attempts = T5GEMMA_QUALITY_RETRIES + 1

    for attempt in range(1, attempts + 1):
        current_seed = T5GEMMA_SEED + attempt - 1
        t5gemma_synthesize(
            text_ja=text,
            out_wav=tmp_wav,
            reference_speech=reference_speech,
            reference_text=reference_text,
            target_duration=target_duration,
            seed=current_seed,
        )

        if not tmp_wav.exists() or tmp_wav.stat().st_size <= 100:
            if attempt < attempts:
                print_step(
                    f"    品質リトライ {attempt}/{attempts}: "
                    "生成ファイルが空 → 再生成"
                )
                tmp_wav.unlink(missing_ok=True)
                continue
            raise PipelineError("T5Gemma-TTS: 生成ファイルが空です。")

        try:
            duration = ffprobe_duration_sec(tmp_wav)
            _validate_t5gemma_quality(duration, target_duration, text)
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


def generate_segment_tts_t5gemma(
    seg: Segment,
    out_audio_stub: Path,
    ref_cache: SpeakerReferenceCache,
    segno: int = 0,
) -> Optional[TtsMeta]:
    """T5Gemma-TTS でセグメントのボイスクローン音声を生成する。"""
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

    reference_speech = ref_cache.get_t5gemma_reference_path(seg.speaker_id)
    if reference_speech is None:
        print_step(
            f"    警告: 話者 {seg.speaker_id} の T5Gemma リファレンスがありません。"
        )
        return None

    reference_text = ref_cache.get_t5gemma_prompt_text(seg.speaker_id)
    target_duration = max(seg.duration * T5GEMMA_DURATION_SCALE, _MIN_DURATION_SEC)

    tmp_wav = out_audio_stub.with_suffix(".t5gemma.wav")

    try:
        _synthesize_with_quality_retry(
            text=text,
            tmp_wav=tmp_wav,
            reference_speech=reference_speech,
            reference_text=reference_text,
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


def release_t5gemma_model() -> None:
    """T5Gemma-TTS 一式を解放してメモリを回復する。"""
    global _AUDIO_TOKENIZER, _MODEL_CONFIG, _T5GEMMA_MODEL, _TEXT_TOKENIZER

    for name in (
        "_AUDIO_TOKENIZER",
        "_MODEL_CONFIG",
        "_T5GEMMA_MODEL",
        "_TEXT_TOKENIZER",
    ):
        globals()[name] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    force_memory_cleanup()
