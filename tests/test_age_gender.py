from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing.tts import age_gender


class AgeGenderTests(unittest.TestCase):
    def test_onnx_logits_order_and_age_scaling(self) -> None:
        profile = age_gender.profile_from_outputs({
            "logits_age": [[0.26]], "logits_gender": [[4.0, -1.0, -2.0]],
        })
        self.assertEqual((profile.gender, profile.age_group), ("female", "young_adult"))
        self.assertEqual(profile.estimated_age, 26.0)
        self.assertGreater(profile.confidence, 0.98)
        profile = age_gender.profile_from_outputs({
            "logits_age": [[0.54]], "logits_gender": [[-3.0, 6.0, 0.0]],
        })
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
                profile = age_gender.profile_from_outputs({
                    "logits_age": [[age]], "logits_gender": [gender],
                })
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
            estimator = age_gender.AgeGenderEstimator(Path(work) / "model")
            with mock.patch.object(age_gender.importlib, "import_module", side_effect=ImportError("audonnx")) as load:
                first = estimator.estimate(path)
                second = estimator.estimate(path)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(first, second)
            self.assertIn("voice-design", first.fallback_reason)
            self.assertFalse((Path(work) / "model").exists())

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
            model = mock.Mock(return_value={
                "logits_age": [[0.35]], "logits_gender": [[-3, 4, -2]],
            })
            estimator._model = model
            estimator._load_attempted = True
            profile = estimator.estimate(path)
            self.assertFalse(profile.fallback_reason)
            signal, rate = model.call_args.args
            self.assertEqual((signal.shape, signal.dtype, rate), ((160000,), np.dtype("float32"), 16000))
            estimator.close()
            self.assertIsNone(estimator._model)
            self.assertIn("closed", estimator.estimate(path).fallback_reason)

    def test_download_is_staged_before_atomic_publication(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            model_dir = Path(work) / "models"
            estimator = age_gender.AgeGenderEstimator(model_dir)
            audeer = mock.Mock()

            def extract(archive, root):
                self.assertFalse((model_dir / age_gender.MODEL_REVISION).exists())
                (Path(root) / "model.onnx").write_bytes(b"onnx")

            audeer.extract_archive.side_effect = extract
            with (
                mock.patch.object(age_gender.importlib, "import_module", return_value=audeer),
                mock.patch.object(age_gender.urllib.request, "urlopen", return_value=io.BytesIO(b"archive")) as download,
            ):
                result = estimator._download_model()
            self.assertEqual((result / "model.onnx").read_bytes(), b"onnx")
            self.assertEqual(estimator._download_model(), result)
            download.assert_called_once_with(age_gender.MODEL_URL, timeout=30)

    def test_concurrent_download_reuses_the_completed_model(self) -> None:
        for race_at_rename in (False, True):
            with self.subTest(race_at_rename=race_at_rename), tempfile.TemporaryDirectory() as work:
                model_dir = Path(work) / "models"
                destination = model_dir / age_gender.MODEL_REVISION
                estimator = age_gender.AgeGenderEstimator(model_dir)
                audeer = mock.Mock()

                def publish_other_model(path=destination):
                    path.mkdir()
                    (path / "model.onnx").write_bytes(b"other complete model")

                def extract(archive, root, race=race_at_rename):
                    (Path(root) / "model.onnx").write_bytes(b"our model")
                    if not race:
                        publish_other_model()

                def competing_replace(source, target):
                    publish_other_model()
                    raise OSError("Directory not empty")

                audeer.extract_archive.side_effect = extract
                with (
                    mock.patch.object(age_gender.importlib, "import_module", return_value=audeer),
                    mock.patch.object(age_gender.urllib.request, "urlopen", return_value=io.BytesIO(b"archive")),
                ):
                    if race_at_rename:
                        with mock.patch.object(Path, "replace", competing_replace):
                            result = estimator._download_model()
                    else:
                        result = estimator._download_model()
                self.assertEqual(result, destination)
                self.assertEqual((result / "model.onnx").read_bytes(), b"other complete model")

    def test_download_timeout_cleans_staging_and_falls_back_without_retrying(self) -> None:
        with tempfile.TemporaryDirectory() as work:
            model_dir = Path(work) / "models"
            path = Path(work) / "voice.wav"
            sf.write(path, np.full(32000, 0.1, dtype=np.float32), 16000)
            estimator = age_gender.AgeGenderEstimator(model_dir)
            with (
                mock.patch.object(age_gender.importlib, "import_module", return_value=mock.Mock()),
                mock.patch.object(age_gender.urllib.request, "urlopen", side_effect=TimeoutError("timed out")) as download,
            ):
                first = estimator.estimate(path)
                second = estimator.estimate(path)
            download.assert_called_once_with(age_gender.MODEL_URL, timeout=30)
            self.assertIn("timed out", first.fallback_reason)
            self.assertEqual(first, second)
            self.assertEqual(list(model_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
