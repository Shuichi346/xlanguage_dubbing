from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from xlanguage_dubbing import config, irodori_tts, omnivoice_tts, voxcpm2_tts
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.tts import irodori_reference_worker as worker
from xlanguage_dubbing.tts import voice_design
from xlanguage_dubbing.tts.age_gender import VoiceProfile
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache, _file_sha256


def _write_voice(path: Path, *, rate: int = 24000, frequency: int = 220, seconds: int = 2):
    time = np.arange(rate * seconds) / rate
    subtype = "PCM_16" if path.suffix == ".flac" else "FLOAT"
    sf.write(path, 0.15 * np.sin(2 * np.pi * frequency * time), rate, subtype=subtype)


def _segments(speakers: str) -> list[Segment]:
    return [
        Segment(index, index * 2.0, index * 2.0 + 2.0, "original", "翻訳済みです。", speaker)
        for index, speaker in enumerate(speakers)
    ]


class VoiceDesignCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.media = self.root / "source.wav"
        _write_voice(self.media, rate=16000, seconds=24)
        self.cache = SpeakerReferenceCache(self.root / "speaker_refs", "voxcpm2")
        self.estimator = mock.Mock()
        self.estimator.estimate.return_value = VoiceProfile(
            gender="female", age_group="young_adult", estimated_age=28.0, confidence=0.99,
        )
        self.generations: list[dict] = []

    def _extract(self, _media, output, **kwargs):
        self.assertEqual(kwargs["sample_rate"], 16000)
        self.assertEqual(kwargs["channels"], 1)
        _write_voice(output, rate=16000)

    def _generate(self, engine, language, profile, seed, path):
        self.estimator.close.assert_called()
        self.generations.append({
            "engine": engine, "language": language, "profile": profile, "seed": seed,
        })
        _write_voice(path, frequency=180 + seed % 400)

    @contextmanager
    def _generation(self, engine="voxcpm2", language="ja"):
        with ExitStack() as stack:
            for name, value in {
                "TTS_ENGINE": engine, "OUTPUT_LANG": language, "VOICE_DESIGN_SEED": 123,
                "AGE_GENDER_MODEL_DIR": self.root / "age_gender",
            }.items():
                stack.enter_context(mock.patch.object(config, name, value))
            stack.enter_context(mock.patch.object(
                voice_design, "AgeGenderEstimator", return_value=self.estimator,
            ))
            stack.enter_context(mock.patch.object(
                voice_design, "extract_audio_segment", side_effect=self._extract,
            ))
            generate = stack.enter_context(mock.patch.object(
                voice_design, "_generate_sample", side_effect=self._generate,
            ))
            yield generate

    def test_four_speakers_generate_once_and_resume_without_estimation(self):
        segments = _segments("ABCDABCD")
        with self._generation() as generate:
            fingerprint = voice_design.build_designed_references(self.media, segments, self.cache)
            self.assertEqual(generate.call_count, 4)
            self.assertEqual(self.estimator.estimate.call_count, 4)
            paths = {speaker: self.cache.get_omnivoice_reference_path(speaker) for speaker in "ABCD"}
            hashes = {speaker: _file_sha256(path) for speaker, path in paths.items() if path}
            self.assertEqual(len(set(hashes.values())), 4)
            generate.reset_mock()
            self.estimator.estimate.reset_mock()
            resumed = SpeakerReferenceCache(self.cache.cache_dir, "voxcpm2")
            self.assertEqual(
                voice_design.build_designed_references(self.media, segments, resumed), fingerprint,
            )
            generate.assert_not_called()
            self.estimator.estimate.assert_not_called()
            for speaker in "ABCD":
                self.assertEqual(resumed.get_omnivoice_reference_path(speaker), paths[speaker])

    def test_generated_refs_replace_old_segment_refs_and_are_reused_for_every_line(self):
        segments = _segments("ABCDABCD")
        old_reference = self.root / "old_segment.wav"
        _write_voice(old_reference)
        self.cache._omnivoice_segment_refs.update(dict.fromkeys(range(1, 9), old_reference))
        calls = []

        def synthesize(**kwargs):
            calls.append(kwargs)
            _write_voice(kwargs["tmp_wav"])

        def convert(_source, destination):
            _write_voice(destination)

        with self._generation():
            voice_design.build_designed_references(self.media, segments, self.cache)
        with (
            mock.patch.object(voxcpm2_tts, "_synthesize_with_quality_retry", side_effect=synthesize),
            mock.patch.object(voxcpm2_tts, "_convert_to_flac", side_effect=convert),
            mock.patch.object(voxcpm2_tts, "ffprobe_duration_sec", return_value=2.0),
        ):
            for segno, segment in enumerate(segments, 1):
                self.assertIsNone(self.cache.get_omnivoice_segment_reference_path(segno))
                voxcpm2_tts.generate_segment_tts_voxcpm2(
                    segment, self.root / f"segment_{segno}", self.cache, segno,
                )
        self.assertEqual(len(calls), 8)
        for index, speaker in enumerate("ABCDABCD"):
            self.assertEqual(calls[index]["ref_audio_path"], self.cache.get_omnivoice_reference_path(speaker))
            self.assertEqual(calls[index]["ref_text"], voice_design.REFERENCE_TEXT["ja"])
        self.assertEqual(calls[0]["ref_audio_path"], calls[4]["ref_audio_path"])

    def test_corruption_rebuilds_only_affected_speaker_with_same_seed(self):
        segments = _segments("ABCDABCD")
        with self._generation() as generate:
            voice_design.build_designed_references(self.media, segments, self.cache)
            paths = {speaker: self.cache.get_omnivoice_reference_path(speaker) for speaker in "ABCD"}
            originals = {speaker: _file_sha256(path) for speaker, path in paths.items() if path}
            original_seeds = [entry["seed"] for entry in self.generations]
            affected = paths["B"]
            assert affected is not None
            affected.write_bytes(b"truncated wav")
            generate.reset_mock()
            self.estimator.estimate.reset_mock()
            voice_design.build_designed_references(self.media, segments, self.cache)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(self.estimator.estimate.call_count, 1)
            self.assertEqual(self.generations[-1]["seed"], original_seeds[1])
            for speaker, path in paths.items():
                assert path is not None
                self.assertEqual(_file_sha256(path), originals[speaker])

    def test_unknown_speaker_is_adult_male_without_acoustic_inference(self):
        segments = [Segment(0, 0.0, 2.0, "unknown", "こんにちは", "")]
        with self._generation():
            voice_design.build_designed_references(self.media, segments, self.cache)
        self.estimator.estimate.assert_not_called()
        self.assertEqual(len(self.generations), 1)
        profile = self.generations[0]["profile"]
        self.assertEqual(profile.gender, "male")
        self.assertEqual(profile.age_group, "middle_aged")
        self.assertTrue(profile.fallback_reason)
        self.assertIsNotNone(self.cache.get_omnivoice_reference_path(""))

    def test_english_reference_matches_supplied_passage_translation(self):
        expected = (
            "The weather has been lovely since this morning. Lots of people are walking along "
            "the street in front of the station. It is a bit of a detour, but let's walk to "
            "that building with the blue roof. We might discover something new."
        )
        with self._generation(language="en"):
            voice_design.build_designed_references(self.media, _segments("A"), self.cache)
        self.assertEqual(self.cache.get_omnivoice_prompt_text("A"), expected)
        with mock.patch.object(omnivoice_tts, "omnivoice_design_voice") as synthesize:
            voice_design._generate_sample(
                "omnivoice", "en", VoiceProfile(), 1, self.root / "sample.wav",
            )
        self.assertEqual(synthesize.call_args.args[0], expected)
        self.assertEqual(synthesize.call_args.args[-1], "en")

    def test_irodori_latents_use_short_fp32_clips_and_rebuild_only_corruption(self):
        manifests = []

        def fake_worker(manifest_path, *, server_dir):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests.append(manifest)
            codec = manifest["codec"]
            self.assertEqual(codec["precision"], "fp32")
            self.assertTrue(codec["deterministic_encode"])
            self.assertEqual(codec["normalize_db"], -16.0)
            for speaker in manifest["speakers"]:
                for clip in speaker["clips"]:
                    info = sf.info(clip)
                    self.assertEqual(info.samplerate, 48000)
                    self.assertEqual(info.channels, 1)
                    self.assertEqual(info.subtype, "FLOAT")
                    self.assertLessEqual(info.duration, 10.0)
                latent = torch.full((50, 32), float(ord(speaker["speaker_id"])), dtype=torch.float32)
                path = Path(speaker["output_path"])
                worker._atomic_save_tensor(latent, path)
                worker._update_cache_metadata(manifest, speaker, latent, path)

        with (
            self._generation(engine="irodori") as generate,
            mock.patch.object(voice_design, "_run_irodori_reference_worker", side_effect=fake_worker),
            mock.patch.object(irodori_tts, "release_irodori_tts_server") as release,
        ):
            voice_design.build_designed_references(self.media, _segments("ABAB"), self.cache)
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(len(manifests), 1)
            self.assertEqual(len(manifests[0]["speakers"]), 2)
            release.assert_called_once()
            first = self.cache.get_irodori_speaker_reference_latent_path("A")
            second = self.cache.get_irodori_speaker_reference_latent_path("B")
            assert first is not None and second is not None
            first_hash = _file_sha256(first)
            second.write_bytes(b"broken tensor")
            generate.reset_mock()
            voice_design.build_designed_references(self.media, _segments("ABAB"), self.cache)
            generate.assert_not_called()
            self.assertEqual(len(manifests), 2)
            self.assertEqual([item["speaker_id"] for item in manifests[-1]["speakers"]], ["B"])
            self.assertEqual(_file_sha256(first), first_hash)
            tensor = torch.load(second, map_location="cpu", weights_only=True)
            self.assertEqual(tensor.shape, (50, 32))
            self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
