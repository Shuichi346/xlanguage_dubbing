from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing.tts import age_gender


class AgeGenderTests(unittest.TestCase):
    def test_model_storage_stays_in_project_when_launched_from_temp(self) -> None:
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from xlanguage_dubbing.config import AGE_GENDER_MODEL_DIR; print(AGE_GENDER_MODEL_DIR)",
                ],
                cwd=temporary_directory,
                env={
                    **os.environ,
                    "PYTHONPATH": str(project / "src"),
                    "AGE_GENDER_MODEL_DIR": "./models/age-gender",
                },
                capture_output=True,
                text=True,
                check=True,
            )
        self.assertEqual(Path(result.stdout.strip()), project / "models" / "age-gender")

    def test_onnx_logits_order_and_age_scaling(self) -> None:
        profile = age_gender.profile_from_outputs(
            {
                "logits_age": [[0.26]],
                "logits_gender": [[4.0, -1.0, -2.0]],
            }
        )
        self.assertEqual((profile.gender, profile.age_group), ("female", "young_adult"))
        self.assertEqual(profile.estimated_age, 26.0)
        self.assertGreater(profile.confidence, 0.98)
        profile = age_gender.profile_from_outputs(
            {
                "logits_age": [[0.54]],
                "logits_gender": [[-3.0, 6.0, 0.0]],
            }
        )
        self.assertEqual((profile.gender, profile.age_group), ("male", "middle_aged"))
        self.assertEqual(profile.estimated_age, 54.0)

    def test_uncertain_child_and_invalid_age_use_adult_male(self) -> None:
        for age, gender in (
            (0.45, [0.1, 0.2, 0.0]),
            (0.12, [-2, -3, 5]),
            (0.12, [5, -3, -2]),
            (1.01, [5, -3, -2]),
        ):
            with self.subTest(age=age, gender=gender):
                profile = age_gender.profile_from_outputs(
                    {
                        "logits_age": [[age]],
                        "logits_gender": [gender],
                    }
                )
                self.assertEqual((profile.gender, profile.age_group), ("male", "middle_aged"))
                self.assertIsNone(profile.estimated_age)
                self.assertTrue(profile.fallback_reason)

    def test_malformed_predictions_are_rejected(self) -> None:
        for age, gender in ((float("nan"), [1, 2, 3]), (0.4, [1, 2]), (0.4, [1, 2, float("inf")])):
            with self.subTest(age=age, gender=gender), self.assertRaises(ValueError):
                age_gender.profile_from_outputs({"logits_age": [[age]], "logits_gender": [gender]})

    def test_failed_optional_import_is_attempted_only_once(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "voice.wav"
            sf.write(path, np.full(32000, 0.1, dtype=np.float32), 16000)
            model_dir = Path(work) / "model"
            model_dir.mkdir()
            (model_dir / "model.onnx").touch()
            estimator = age_gender.AgeGenderEstimator(model_dir)
            with mock.patch.object(
                age_gender.importlib, "import_module", side_effect=ImportError("audonnx")
            ) as load:
                first = estimator.estimate(path)
                second = estimator.estimate(path)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first, second)
            self.assertIn("voice-design", first.fallback_reason)
            self.assertFalse((model_dir / "hf24").exists())

    def test_invalid_audio_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            estimator = age_gender.AgeGenderEstimator(Path(work) / "model")
            with mock.patch.object(estimator, "_load_model") as load:
                for name, waveform, rate in (
                    ("silence", np.zeros(16000), 16000),
                    ("short", np.ones(15000) * 0.1, 16000),
                    ("rate", np.ones(24000) * 0.1, 24000),
                    ("nan", np.full(16000, np.nan), 16000),
                ):
                    path = Path(work) / f"{name}.wav"
                    sf.write(path, waveform, rate, subtype="FLOAT")
                    profile = estimator.estimate(path)
                    self.assertTrue(profile.fallback_reason)
                self.assertTrue(estimator.estimate(Path(work) / "missing.wav").fallback_reason)
            load.assert_not_called()

    def test_inference_is_float32_bounded_and_session_is_released(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "voice.wav"
            sf.write(path, np.ones((16000 * 12, 2)) * 0.1, 16000)
            estimator = age_gender.AgeGenderEstimator(Path(work) / "model")
            model = mock.Mock(
                return_value={
                    "logits_age": [[0.35]],
                    "logits_gender": [[-3, 4, -2]],
                }
            )
            estimator._model = model
            estimator._load_attempted = True
            profile = estimator.estimate(path)
            self.assertFalse(profile.fallback_reason)
            signal, rate = model.call_args.args
            self.assertEqual(
                (signal.shape, signal.dtype, rate), ((160000,), np.dtype("float32"), 16000)
            )
            estimator.close()
            self.assertIsNone(estimator._model)
            self.assertIn("closed", estimator.estimate(path).fallback_reason)

    def test_hf_download_is_pinned_and_uses_only_required_safe_files(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            estimator = age_gender.AgeGenderEstimator(Path(work))
            with mock.patch(
                "huggingface_hub.snapshot_download", return_value=str(Path(work) / "hf24")
            ) as download:
                result = estimator._download_model()
            self.assertEqual(result, Path(work) / "hf24")
            download.assert_called_once_with(
                age_gender.HF_MODEL_ID,
                revision=age_gender.HF_MODEL_REVISION,
                local_dir=str(result),
                max_workers=2,
                allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
            )

    def test_local_onnx_export_bypasses_all_network_downloads(self) -> None:
        for legacy in (False, True):
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as work:
                model_dir = Path(work)
                root = model_dir / age_gender.LEGACY_ONNX_REVISION if legacy else model_dir
                root.mkdir(exist_ok=True)
                (root / "model.onnx").touch()
                estimator = age_gender.AgeGenderEstimator(model_dir)
                audonnx = mock.Mock()
                with (
                    mock.patch.object(age_gender.importlib, "import_module", return_value=audonnx),
                    mock.patch.object(estimator, "_download_model") as download,
                ):
                    self.assertIs(estimator._load_model(), audonnx.load.return_value)
                    audonnx.load.assert_called_once_with(str(root), device="cpu", num_workers=2)
                    download.assert_not_called()

    def test_hf_failure_falls_back_once_without_zenodo_request(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "voice.wav"
            sf.write(path, np.full(32000, 0.1, dtype=np.float32), 16000)
            estimator = age_gender.AgeGenderEstimator(Path(work) / "models")
            with mock.patch.object(
                estimator, "_download_model", side_effect=TimeoutError("Hugging Face timed out")
            ) as download:
                first = estimator.estimate(path)
                second = estimator.estimate(path)
            download.assert_called_once()
            self.assertIn("Hugging Face timed out", first.fallback_reason)
            self.assertEqual(first, second)
            self.assertEqual(first.gender, "male")
            estimator.close()

    def test_default_hf_predictor_is_loaded_once_and_released(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "voice.wav"
            sf.write(path, np.full(32000, 0.1, dtype=np.float32), 16000)
            estimator = age_gender.AgeGenderEstimator(Path(work) / "models")
            predictor = mock.Mock(
                return_value={
                    "logits_age": [[0.35]],
                    "logits_gender": [[-3, 4, -2]],
                }
            )
            with (
                mock.patch.object(
                    estimator, "_download_model", return_value=Path(work)
                ) as download,
                mock.patch(
                    "xlanguage_dubbing.tts.age_gender_torch.HuggingFacePredictor",
                    return_value=predictor,
                ) as factory,
            ):
                self.assertFalse(estimator.estimate(path).fallback_reason)
                self.assertFalse(estimator.estimate(path).fallback_reason)
            download.assert_called_once()
            factory.assert_called_once_with(Path(work))
            self.assertEqual(predictor.call_count, 2)
            estimator.close()
            self.assertIsNone(estimator._model)


if __name__ == "__main__":
    unittest.main()
