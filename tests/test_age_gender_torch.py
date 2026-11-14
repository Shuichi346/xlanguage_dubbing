from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model

from xlanguage_dubbing.tts.age_gender import profile_from_outputs
from xlanguage_dubbing.tts.age_gender_torch import AgeGenderModel, HuggingFacePredictor


def _tiny_config() -> Wav2Vec2Config:
    return Wav2Vec2Config.from_dict({
        "hidden_size": 8,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
        "conv_dim": [4, 4],
        "conv_stride": [2, 2],
        "conv_kernel": [5, 3],
        "num_conv_pos_embeddings": 8,
        "num_conv_pos_embedding_groups": 2,
        "feat_extract_norm": "layer",
        "mask_time_prob": 0.0,
    })


class AgeGenderTorchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16000, padding_value=0.0,
            do_normalize=True, return_attention_mask=False,
        )
        self.extractor.save_pretrained(self.root)

    def _save_prediction_checkpoint(self) -> None:
        model = AgeGenderModel(_tiny_config())
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("parametrizations.weight.original1"):
                    # A unit denominator keeps zero weight-normalized kernels finite.
                    parameter.fill_(1.0)
                else:
                    parameter.zero_()
            model.age.out_proj.bias.copy_(torch.tensor([0.34]))
            model.gender.out_proj.bias.copy_(torch.tensor([-2.0, 3.0, -4.0]))
        model.save_pretrained(self.root, safe_serialization=True)

    def test_reloaded_safetensors_preserve_both_heads_and_raw_output_contract(self) -> None:
        self._save_prediction_checkpoint()
        self.assertTrue((self.root / "model.safetensors").is_file())
        predictor = HuggingFacePredictor(self.root)
        signal = np.sin(np.arange(1600, dtype=np.float32) * 0.1)
        outputs = predictor(signal, 16000)
        self.assertFalse(predictor.model.training)
        for parameter in predictor.model.parameters():
            self.assertEqual(parameter.device.type, "cpu")
            self.assertEqual(parameter.dtype, torch.float32)
        self.assertEqual(outputs["logits_age"].shape, (1, 1))
        self.assertEqual(outputs["logits_gender"].shape, (1, 3))
        self.assertEqual(outputs["logits_age"].dtype, np.float32)
        self.assertEqual(outputs["logits_gender"].dtype, np.float32)
        np.testing.assert_allclose(outputs["logits_age"], [[0.34]])
        np.testing.assert_array_equal(outputs["logits_gender"], [[-2.0, 3.0, -4.0]])
        profile = profile_from_outputs(outputs)
        self.assertEqual(profile.gender, "male")
        self.assertEqual(profile.age_group, "young_adult")
        assert profile.estimated_age is not None
        self.assertAlmostEqual(profile.estimated_age, 34.0, places=4)
        self.assertFalse(profile.fallback_reason)

    def test_feature_extractor_normalizes_audio_before_inference_without_gradients(self) -> None:
        self._save_prediction_checkpoint()
        predictor = HuggingFacePredictor(self.root)
        signal = 2.0 + 0.15 * np.sin(np.arange(1600, dtype=np.float32) * 0.17)
        observed = []

        def capture(_module, arguments):
            values = arguments[0]
            observed.append({
                "values": values.clone(),
                "grad_enabled": torch.is_grad_enabled(),
                "inference_mode": torch.is_inference_mode_enabled(),
                "requires_grad": values.requires_grad,
            })

        handle = predictor.model.register_forward_pre_hook(capture)
        try:
            predictor(signal, 16000)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 1)
        recorded = observed[0]
        values = recorded["values"]
        expected = self.extractor(signal, sampling_rate=16000, return_tensors="pt")["input_values"]
        torch.testing.assert_close(values, expected)
        self.assertAlmostEqual(float(values.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(values.std(unbiased=False)), 1.0, places=4)
        self.assertFalse(recorded["grad_enabled"])
        self.assertTrue(recorded["inference_mode"])
        self.assertFalse(recorded["requires_grad"])

    def test_base_checkpoint_without_prediction_heads_is_rejected(self) -> None:
        base = Wav2Vec2Model(_tiny_config())
        base.save_pretrained(self.root, safe_serialization=True)
        with self.assertRaisesRegex(ValueError, "Incomplete age/gender checkpoint"):
            HuggingFacePredictor(self.root)


if __name__ == "__main__":
    unittest.main()
