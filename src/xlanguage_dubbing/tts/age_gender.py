"""Estimate coarse voice attributes with audEERING's local speech models."""

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from xlanguage_dubbing.utils import print_step

HF_MODEL_ID = "audeering/wav2vec2-large-robust-24-ft-age-gender"
HF_MODEL_REVISION = "bad2737f529168bc3d3fcd97edfaba9583c7d36d"
LEGACY_ONNX_REVISION = "w2v2-L-robust-6-age-gender-1.1.1"
SAMPLE_RATE = 16000
MAX_SAMPLE_SECONDS = 10


@dataclass(frozen=True)
class VoiceProfile:
    """Coarse synthesis attributes, including an explicit uncertainty fallback."""

    gender: str = "male"
    age_group: str = "middle_aged"
    estimated_age: float | None = None
    confidence: float = 0.0
    fallback_reason: str = ""


def profile_from_outputs(outputs: dict) -> VoiceProfile:
    """Convert raw ONNX logits in female/male/child order into adult attributes."""
    age_values = np.asarray(outputs["logits_age"], dtype=np.float64).reshape(-1)
    gender_values = np.asarray(outputs["logits_gender"], dtype=np.float64).reshape(-1)
    if age_values.size != 1 or gender_values.size != 3:
        raise ValueError("Unexpected age/gender output shape")
    if not np.isfinite(age_values).all() or not np.isfinite(gender_values).all():
        raise ValueError("Non-finite age/gender predictions")

    # The ONNX export returns raw logits; the Torch example applies softmax itself.
    probabilities = np.exp(gender_values - np.max(gender_values))
    probabilities /= probabilities.sum()
    winner = int(np.argmax(probabilities))
    confidence = float(probabilities[winner])
    age = float(age_values[0] * 100.0)
    if winner == 2:
        return VoiceProfile(fallback_reason="Child prediction has no supported adult gender")
    if confidence < 0.6:
        return VoiceProfile(fallback_reason="Gender confidence is below 0.6")
    if not 18.0 <= age <= 100.0:
        return VoiceProfile(fallback_reason="Estimated age is outside the supported adult range")
    return VoiceProfile(
        gender=("female", "male")[winner],
        age_group="young_adult" if age < 40.0 else "middle_aged",
        estimated_age=age,
        confidence=confidence,
    )


class AgeGenderEstimator:
    """Lazily load one CPU model and release it before TTS model loading."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.expanduser()
        self._model = None
        self._load_attempted = False
        self._load_error = ""
        self._closed = False

    def _download_model(self) -> Path:
        from huggingface_hub import snapshot_download

        print_step("  年齢・性別推定モデルを準備中 (audEERING 24層, Hugging Face, CPU)")
        # The Hub uses resumable downloads, file locks and atomic publication.
        return Path(
            snapshot_download(
                HF_MODEL_ID,
                revision=HF_MODEL_REVISION,
                local_dir=str(self.model_dir / "hf24"),
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
                max_workers=2,
            )
        )

    def _load_model(self):
        if self._load_attempted:
            if self._model is None:
                raise RuntimeError(self._load_error)
            return self._model
        self._load_attempted = True
        try:
            # Preserve supplied ONNX exports and already downloaded legacy caches.
            onnx_root = next(
                (
                    root
                    for root in (self.model_dir, self.model_dir / LEGACY_ONNX_REVISION)
                    if (root / "model.onnx").is_file()
                ),
                None,
            )
            if onnx_root is not None:
                audonnx = importlib.import_module("audonnx")
                self._model = audonnx.load(str(onnx_root), device="cpu", num_workers=2)
            else:
                from xlanguage_dubbing.tts.age_gender_torch import HuggingFacePredictor

                self._model = HuggingFacePredictor(self._download_model())
            return self._model
        except Exception as exc:
            self._load_error = (
                "Age/gender model unavailable; check the Hugging Face connection "
                "and model cache (local ONNX requires the voice-design extra): "
                f"{exc}"
            )
            raise RuntimeError(self._load_error) from exc

    def estimate(self, audio_path: Path) -> VoiceProfile:
        """Return adult male attributes on unavailable, ambiguous, or invalid input."""
        try:
            if self._closed:
                raise RuntimeError("Age/gender estimator is closed")
            signal, sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=True,
                frames=SAMPLE_RATE * MAX_SAMPLE_SECONDS,
            )
            if sample_rate != SAMPLE_RATE:
                raise ValueError("Age/gender input must be 16000 Hz")
            signal = signal.mean(axis=1, dtype=np.float32)
            if signal.size < SAMPLE_RATE:
                raise ValueError("Less than one second of speech is available")
            if not np.isfinite(signal).all():
                raise ValueError("Reference speech contains non-finite samples")
            if float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))) < 1e-5:
                raise ValueError("Reference speech is silent")
            model = self._load_model()
            profile = profile_from_outputs(model(signal, SAMPLE_RATE))
        except Exception as exc:
            profile = VoiceProfile(fallback_reason=str(exc))
        if profile.fallback_reason:
            print_step(f"  年齢・性別不明 → 大人の男性を使用: {profile.fallback_reason}")
        return profile

    def close(self) -> None:
        """Drop the ONNX session so its memory is available to synthesis."""
        self._model = None
        self._closed = True
        gc.collect()
