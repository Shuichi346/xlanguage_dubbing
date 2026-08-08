from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from xlanguage_dubbing.tts import irodori_reference_worker as worker


class _FakeCodec:
    sample_rate = 48000
    model = types.SimpleNamespace(hop_length=1920)

    def __init__(self) -> None:
        self.encode_calls: list[dict] = []

    def encode_waveform(
        self,
        waveform,
        *,
        sample_rate,
        normalize_db,
        ensure_max,
    ):
        marker = float(len(self.encode_calls) + 1)
        self.encode_calls.append(
            {
                "shape": tuple(waveform.shape),
                "dtype": waveform.dtype,
                "sample_rate": sample_rate,
                "normalize_db": normalize_db,
                "ensure_max": ensure_max,
            }
        )
        return torch.full((1, 2, 32), marker, dtype=torch.float32)


class IrodoriReferenceWorkerTests(unittest.TestCase):
    def test_codec_loads_once_and_clips_are_encoded_in_order_then_trimmed(self) -> None:
        fake_codec = _FakeCodec()
        load_calls: list[dict] = []

        class FakeDACVAECodec:
            @classmethod
            def load(cls, **kwargs):
                load_calls.append(kwargs)
                return fake_codec

        package_module = types.ModuleType("irodori_tts")
        codec_module = types.ModuleType("irodori_tts.codec")
        codec_module.DACVAECodec = FakeDACVAECodec

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            clip_paths = []
            for index in range(3):
                clip_path = root / f"clip_{index}.wav"
                clip_path.write_bytes(b"clip")
                clip_paths.append(str(clip_path))
            output_paths = {
                speaker: root / f"speaker_{speaker}.pt" for speaker in "ABC"
            }
            manifest = {
                "codec": {
                    "repo_id": "Aratako/Semantic-DACVAE-Japanese-32dim",
                    "device": "cpu",
                    "precision": "fp32",
                    "deterministic_encode": True,
                    "normalize_db": -16.0,
                    "sample_rate": 48000,
                    "latent_dim": 32,
                    "max_seconds": 0.12,
                },
                "speakers": [
                    {
                        "speaker_id": speaker,
                        "clips": clip_paths,
                        "output_path": str(output_paths[speaker]),
                    }
                    for speaker in "ABC"
                ],
            }

            with (
                mock.patch.dict(
                    sys.modules,
                    {
                        "irodori_tts": package_module,
                        "irodori_tts.codec": codec_module,
                    },
                ),
                mock.patch.object(
                    worker,
                    "_load_audio",
                    return_value=(
                        torch.zeros((1, 480), dtype=torch.float32),
                        48000,
                    ),
                ) as load_audio_mock,
            ):
                worker.build_reference_latents(manifest)

            self.assertEqual(len(load_calls), 1)
            self.assertEqual(load_calls[0]["dtype"], torch.float32)
            self.assertTrue(load_calls[0]["deterministic_encode"])
            self.assertEqual(load_calls[0]["normalize_db"], -16.0)
            self.assertEqual(load_audio_mock.call_args_list[0].args[0], Path(clip_paths[0]))
            self.assertEqual(load_audio_mock.call_args_list[1].args[0], Path(clip_paths[1]))
            self.assertEqual(len(fake_codec.encode_calls), 6)
            for call in fake_codec.encode_calls:
                self.assertEqual(call["shape"], (1, 1, 480))
                self.assertEqual(call["dtype"], torch.float32)
                self.assertEqual(call["sample_rate"], 48000)
                self.assertEqual(call["normalize_db"], -16.0)
                self.assertTrue(call["ensure_max"])

            latent = torch.load(
                output_paths["A"],
                map_location="cpu",
                weights_only=True,
            )
            self.assertEqual(tuple(latent.shape), (3, 32))
            self.assertEqual(latent.dtype, torch.float32)
            self.assertTrue(torch.equal(latent[:, 0], torch.tensor([1.0, 1.0, 2.0])))
            self.assertTrue(all(path.is_file() for path in output_paths.values()))
            self.assertEqual(list(root.glob(".*.tmp.*")), [])

    def test_atomic_save_preserves_existing_cache_when_write_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "speaker.pt"
            output_path.write_bytes(b"existing-cache")

            with mock.patch.object(torch, "save", side_effect=RuntimeError("write failed")):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    worker._atomic_save_tensor(
                        torch.zeros((2, 32), dtype=torch.float32),
                        output_path,
                    )

            self.assertEqual(output_path.read_bytes(), b"existing-cache")
            self.assertEqual(list(output_path.parent.glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
