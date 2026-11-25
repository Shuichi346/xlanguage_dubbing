"""Reference audio/text matching, engine limits, and safe resumability."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.tts import reference
from xlanguage_dubbing.utils import PipelineError


class ReferenceSettingsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.media = self.root / "source.wav"
        self.media.write_bytes(b"source identity")
        self.extract = self.enterContext(mock.patch.object(
            reference, "extract_audio_segment", side_effect=self._extract,
        ))
        self.transcribe = self.enterContext(mock.patch(
            "xlanguage_dubbing.asr.transcribe_reference_audio", return_value="clipped words",
        ))
        self.enterContext(mock.patch.multiple(
            reference, OMNIVOICE_REFERENCE_MIN_SEC=3.0, OMNIVOICE_REFERENCE_MAX_SEC=15.0,
            OMNIVOICE_REFERENCE_TARGET_SEC=8.0, VOXCPM2_REFERENCE_MIN_SEC=4.0,
            VOXCPM2_REFERENCE_MAX_SEC=30.0, VOXCPM2_REFERENCE_TARGET_SEC=12.0,
        ))

    @staticmethod
    def _extract(_source, output, *, start, end, sample_rate, channels):
        sf.write(output, np.zeros((round((end - start) * sample_rate), channels)), sample_rate)

    def _cache(self, engine="omnivoice"):
        return reference.SpeakerReferenceCache(self.root / engine, engine)

    def test_clipped_omni_reference_uses_only_matching_audio_transcript(self):
        cache = self._cache()
        segment = Segment(0, 2.0, 32.0, "entire thirty-second transcript", speaker_id="A",
                          detected_lang="zh-hant")

        def transcribe(path, *, language):
            self.assertAlmostEqual(sf.info(path).duration, 15.0)
            self.assertEqual(language, "zh")
            return " words from the clipped audio "

        self.transcribe.side_effect = transcribe
        cache.build_omnivoice_segment_references(self.media, [segment])
        self.assertEqual(cache.get_omnivoice_segment_prompt_text(1), "words from the clipped audio")
        self.assertEqual(self.extract.call_args.kwargs["start"], 2.0)
        self.assertEqual(self.extract.call_args.kwargs["end"], 17.0)

    def test_untrimmed_omni_reference_reuses_corresponding_source_text_without_asr(self):
        cache = self._cache()
        segment = Segment(0, 1.0, 4.0, " exact   source words ", speaker_id="A", detected_lang="en")
        cache.build_omnivoice_segment_references(self.media, [segment])
        self.assertEqual(cache.get_omnivoice_segment_prompt_text(1), "exact source words")
        self.transcribe.assert_not_called()

    def test_vox_uses_its_limits_names_and_no_reference_asr(self):
        cache = self._cache("voxcpm2")
        segment = Segment(0, 0.0, 35.0, "ignored reference transcript", speaker_id="../speaker")
        intervals = [DiarizationSegment(0, 8, "../speaker"), DiarizationSegment(9, 21, "../speaker")]
        cache.build_omnivoice_references(self.media, intervals, [segment])
        self.assertEqual((self.extract.call_args.kwargs["start"], self.extract.call_args.kwargs["end"]), (9, 21))
        speaker_path = cache.get_omnivoice_reference_path("../speaker")
        assert speaker_path is not None
        self.assertEqual(speaker_path.parent, cache.cache_dir.resolve())
        self.assertTrue(speaker_path.name.startswith("voxcpm2_ref_"))
        cache.build_omnivoice_segment_references(self.media, [segment])
        path = cache.get_omnivoice_segment_reference_path(1)
        assert path is not None
        self.assertTrue(path.name.startswith("voxcpm2_seg_ref_"))
        self.assertAlmostEqual(sf.info(path).duration, 30.0)
        self.assertEqual(cache.get_omnivoice_segment_prompt_text(1), "")
        self.transcribe.assert_not_called()

    def test_reference_resume_reuses_valid_audio_and_repairs_corruption(self):
        segment = Segment(0, 0.0, 30.0, "source", speaker_id="A", detected_lang="en")
        cache = self._cache()
        cache.build_omnivoice_segment_references(self.media, [segment])
        path = cache.get_omnivoice_segment_reference_path(1)
        assert path is not None
        self.extract.reset_mock()
        self.transcribe.reset_mock()
        resumed = self._cache()
        resumed.build_omnivoice_segment_references(self.media, [segment])
        self.extract.assert_not_called()
        self.transcribe.assert_not_called()
        self.assertEqual(resumed.get_omnivoice_segment_reference_path(1), path)
        path.write_bytes(b"corrupt audio")
        resumed.reload_omnivoice_segment_references(1)
        self.assertIsNone(resumed.get_omnivoice_segment_reference_path(1))
        resumed.build_omnivoice_segment_references(self.media, [segment])
        self.assertEqual(self.extract.call_count, 1)
        self.assertEqual(self.transcribe.call_count, 1)

    def test_source_text_and_limit_changes_invalidate_segment_references(self):
        cache = self._cache()
        segment = Segment(0, 0.0, 30.0, "source", speaker_id="A", detected_lang="en")
        cache.build_omnivoice_segment_references(self.media, [segment])
        self.media.write_bytes(b"replacement source identity")
        cache.build_omnivoice_segment_references(self.media, [segment])
        cache.build_omnivoice_segment_references(self.media, [replace(segment, text_src="corrected")])
        with mock.patch.object(reference, "OMNIVOICE_REFERENCE_MAX_SEC", 12.0):
            cache.build_omnivoice_segment_references(self.media, [replace(segment, text_src="corrected")])
        self.assertEqual(self.extract.call_count, 4)
        self.assertEqual(self.extract.call_args.kwargs["end"], 12.0)

    def test_legacy_metadata_is_not_trusted_and_removed_turns_are_not_reloaded(self):
        cache = self._cache()
        segments = [Segment(0, 0, 3, "first", speaker_id="A"),
                    Segment(1, 4, 7, "second", speaker_id="A")]
        cache.build_omnivoice_segment_references(self.media, segments)
        meta = cache._segment_meta_path()
        payload = json.loads(meta.read_text())
        payload["1"].pop("schema")
        meta.write_text(json.dumps(payload))
        cache.build_omnivoice_segment_references(self.media, segments[:1])
        self.assertEqual(self.extract.call_count, 3)
        resumed = self._cache()
        resumed.reload_omnivoice_segment_references(2)
        self.assertIsNotNone(resumed.get_omnivoice_segment_reference_path(1))
        self.assertIsNone(resumed.get_omnivoice_segment_reference_path(2))

    def test_empty_clipped_transcript_is_retried_instead_of_caching_full_text(self):
        cache = self._cache()
        segment = Segment(0, 0, 30, "entire transcript", speaker_id="A")
        self.transcribe.return_value = ""
        with self.assertRaisesRegex(PipelineError, "文字起こしが空"):
            cache.build_omnivoice_segment_references(self.media, [segment])
        self.assertIsNone(cache.get_omnivoice_segment_reference_path(1))
        self.assertFalse(cache._segment_meta_path().exists())
        self.transcribe.return_value = "clipped transcript"
        cache.build_omnivoice_segment_references(self.media, [segment])
        self.assertEqual(cache.get_omnivoice_segment_prompt_text(1), "clipped transcript")

    def test_invalid_reference_limits_fail_before_extracting(self):
        cache = self._cache("voxcpm2")
        for maximum in (2.0, float("inf"), float("nan")):
            with mock.patch.object(reference, "VOXCPM2_REFERENCE_MAX_SEC", maximum):
                with self.assertRaises(PipelineError):
                    cache.build_omnivoice_segment_references(self.media, [])
        self.extract.assert_not_called()


if __name__ == "__main__":
    unittest.main()
