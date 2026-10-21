from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from xlanguage_dubbing import irodori_tts, omnivoice_tts, voxcpm2_tts
from xlanguage_dubbing.utils import PipelineError


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b"RIFF" + b"\0" * 256


class VoiceDesignEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        omnivoice_tts._DESIGNED_VOICE_PROMPTS.clear()
        self.addCleanup(omnivoice_tts._DESIGNED_VOICE_PROMPTS.clear)

    def test_voxcpm_design_prefix_has_no_original_reference(self) -> None:
        model = mock.Mock(tts_model=SimpleNamespace(sample_rate=48000))
        model.generate.return_value = np.ones(4800, dtype=np.float32) * 0.1
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "voice.wav"
            with (
                mock.patch.object(voxcpm2_tts, "_get_voxcpm2_model", return_value=model),
                mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            ):
                voxcpm2_tts.voxcpm2_design_voice("こんにちは。", output, "大人の男性", 42, "ja")
            self.assertEqual(sf.info(output).samplerate, 48000)
        call = model.generate.call_args.kwargs
        self.assertEqual(call["text"], "(大人の男性)こんにちは。")
        self.assertNotIn("reference_wav_path", call)
        self.assertNotIn("prompt_wav_path", call)
        self.assertNotIn("language", call)

    def test_voxcpm_clone_keeps_reference_only_contract(self) -> None:
        model = mock.Mock(tts_model=SimpleNamespace(sample_rate=48000))
        model.generate.return_value = np.ones(1000, dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "reference.wav"
            reference.touch()
            with mock.patch.object(voxcpm2_tts, "_get_voxcpm2_model", return_value=model):
                voxcpm2_tts.voxcpm2_synthesize(
                    "こんにちは。", root / "clone.wav", reference, "unused transcript",
                )
        call = model.generate.call_args.kwargs
        self.assertEqual(call["reference_wav_path"], str(reference))
        self.assertNotIn("prompt_text", call)
        self.assertNotIn("prompt_wav_path", call)

    def test_omnivoice_uses_only_selected_attributes_and_target_language(self) -> None:
        model = mock.Mock(sampling_rate=24000)
        model.generate.return_value = [np.ones(2400, dtype=np.float32) * 0.1]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "voice.wav"
            with (
                mock.patch.object(omnivoice_tts, "_get_omnivoice_model", return_value=model),
                mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            ):
                omnivoice_tts.omnivoice_design_voice(
                    "Hello.", output, "female", "middle_aged", 12, "en",
                )
            self.assertEqual(sf.info(output).samplerate, 24000)
        call = model.generate.call_args.kwargs
        self.assertEqual(call["instruct"], "female, middle-aged, moderate pitch")
        self.assertEqual(call["language"], "en")
        self.assertNotIn("ref_audio", call)
        self.assertNotIn("duration", call)

    def test_design_seed_is_repeatable_and_restores_cpu_rng(self) -> None:
        samples = []

        def generate(**_kwargs):
            waveform = torch.rand(1000).numpy()
            samples.append(waveform)
            return waveform

        model = SimpleNamespace(
            generate=generate, tts_model=SimpleNamespace(sample_rate=48000),
        )
        before = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch.object(voxcpm2_tts, "_get_voxcpm2_model", return_value=model),
                mock.patch.object(torch.backends.mps, "is_available", return_value=False),
            ):
                for index, seed in enumerate([1, 1, 2]):
                    voxcpm2_tts.voxcpm2_design_voice(
                        "Hello.", Path(folder) / f"{index}.wav", "adult male", seed, "en",
                    )
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        np.testing.assert_array_equal(samples[0], samples[1])
        self.assertFalse(np.array_equal(samples[0], samples[2]))

    def test_mps_rng_is_restored_after_generation_error(self) -> None:
        model = mock.Mock()
        model.generate.side_effect = RuntimeError("generation failed")
        mps_state = torch.tensor([1], dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch.object(voxcpm2_tts, "_get_voxcpm2_model", return_value=model),
                mock.patch.object(torch.backends.mps, "is_available", return_value=True),
                mock.patch.object(torch.mps, "get_rng_state", return_value=mps_state),
                mock.patch.object(torch.mps, "manual_seed") as seed,
                mock.patch.object(torch.mps, "set_rng_state") as restore,
                self.assertRaises(PipelineError),
            ):
                voxcpm2_tts.voxcpm2_design_voice(
                    "Hello.", Path(folder) / "voice.wav", "adult male", 23, "en",
                )
        seed.assert_called_once_with(23)
        restore.assert_called_once_with(mps_state)

    def test_irodori_design_payload_has_caption_and_no_reference(self) -> None:
        requests = []

        def urlopen(request, *, timeout):
            requests.append(request)
            return _Response()

        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch.object(irodori_tts, "ensure_irodori_tts_server"),
                mock.patch.object(irodori_tts.urllib.request, "urlopen", side_effect=urlopen),
            ):
                irodori_tts.irodori_design_voice(
                    "こんにちは。", Path(folder) / "voice.wav", "落ち着いた男性", 5, "ja",
                )
        payload = json.loads(requests[0].data)
        options = payload["irodori"]
        self.assertEqual(payload["response_format"], "wav")
        self.assertEqual(options["caption"], "落ち着いた男性")
        self.assertTrue(options["no_ref"])
        self.assertEqual(options["seed"], 5)
        self.assertFalse(options["chunking_enabled"])
        self.assertEqual(options["num_steps"], irodori_tts.IRODORI_TTS_NUM_STEPS)
        self.assertEqual(options["t_schedule_mode"], "sway")
        for key in ("ref_wav", "ref_latent", "seconds", "style_prompt"):
            self.assertNotIn(key, options)

    def test_invalid_language_or_attributes_fail_before_loading_models(self) -> None:
        output = Path("unused.wav")
        with (
            mock.patch.object(voxcpm2_tts, "_get_voxcpm2_model") as vox,
            mock.patch.object(omnivoice_tts, "_get_omnivoice_model") as omni,
            mock.patch.object(irodori_tts, "ensure_irodori_tts_server") as server,
        ):
            with self.assertRaises(PipelineError):
                voxcpm2_tts.voxcpm2_design_voice("Hallo", output, "male", 1, "de")
            with self.assertRaises(PipelineError):
                omnivoice_tts.omnivoice_design_voice("Hello", output, "female", "child", 1, "en")
            with self.assertRaises(PipelineError):
                irodori_tts.irodori_design_voice("Hello", output, "male", 1, "en")
        vox.assert_not_called()
        omni.assert_not_called()
        server.assert_not_called()

    def test_omnivoice_design_cloning_reuses_prompt_and_replaces_changed_reference(self) -> None:
        model = mock.Mock(sampling_rate=24000)
        model.generate.return_value = [np.full(2400, 0.1, dtype=np.float32)]
        prompts = [object(), object(), object()]
        model.create_voice_clone_prompt.side_effect = prompts
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "designed_speaker.wav"
            sf.write(reference, np.full(2400, 0.1), 24000)
            with mock.patch.object(omnivoice_tts, "_get_omnivoice_model", return_value=model):
                for index in range(2):
                    omnivoice_tts.omnivoice_synthesize(
                        f"Line {index}.", root / f"line_{index}.wav", reference,
                        "The designed reference.", cache_voice_prompt=True,
                    )
                model.create_voice_clone_prompt.assert_called_once_with(
                    ref_audio=str(reference.resolve()), ref_text="The designed reference.",
                )
                for call in model.generate.call_args_list:
                    self.assertIs(call.kwargs["voice_clone_prompt"], prompts[0])
                    self.assertNotIn("ref_audio", call.kwargs)
                    self.assertNotIn("ref_text", call.kwargs)
                sf.write(reference, np.full(4800, 0.2), 24000)
                omnivoice_tts.omnivoice_synthesize(
                    "Next line.", root / "next.wav", reference,
                    "The designed reference.", cache_voice_prompt=True,
                )
                self.assertEqual(model.create_voice_clone_prompt.call_count, 2)
                self.assertIs(model.generate.call_args.kwargs["voice_clone_prompt"], prompts[1])
                omnivoice_tts.omnivoice_synthesize(
                    "Another line.", root / "another.wav", reference,
                    "A corrected transcript.", cache_voice_prompt=True,
                )
                self.assertEqual(model.create_voice_clone_prompt.call_count, 3)
                self.assertIs(model.generate.call_args.kwargs["voice_clone_prompt"], prompts[2])
                self.assertEqual(len(omnivoice_tts._DESIGNED_VOICE_PROMPTS), 1)

    def test_omnivoice_original_refs_are_not_cached_and_release_drops_designed_prompts(self) -> None:
        model = mock.Mock(sampling_rate=24000)
        model.generate.return_value = [np.full(2400, 0.1, dtype=np.float32)]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "speaker.wav"
            sf.write(reference, np.full(2400, 0.1), 24000)
            with (
                mock.patch.object(omnivoice_tts, "_get_omnivoice_model", return_value=model),
                mock.patch.object(torch.mps, "empty_cache"),
            ):
                omnivoice_tts.omnivoice_synthesize(
                    "Original clone.", root / "original.wav", reference, "Reference.",
                )
                model.create_voice_clone_prompt.assert_not_called()
                self.assertEqual(model.generate.call_args.kwargs["ref_audio"], str(reference))
                omnivoice_tts.omnivoice_synthesize(
                    "Designed clone.", root / "design.wav", reference, "Reference.",
                    cache_voice_prompt=True,
                )
                self.assertEqual(len(omnivoice_tts._DESIGNED_VOICE_PROMPTS), 1)
                omnivoice_tts.release_omnivoice_model()
                self.assertEqual(omnivoice_tts._DESIGNED_VOICE_PROMPTS, {})
                omnivoice_tts.omnivoice_synthesize(
                    "New model session.", root / "new.wav", reference, "Reference.",
                    cache_voice_prompt=True,
                )
                self.assertEqual(model.create_voice_clone_prompt.call_count, 2)


if __name__ == "__main__":
    unittest.main()
