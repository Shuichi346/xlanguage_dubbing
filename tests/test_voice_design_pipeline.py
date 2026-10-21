from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from xlanguage_dubbing import cli, config
from xlanguage_dubbing.asr import nemo_speech
from xlanguage_dubbing.core import pipeline
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.tts import voice_design
from xlanguage_dubbing.utils import PipelineError


class VoiceDesignPipelineTests(unittest.TestCase):
    def test_language_engine_boundaries(self):
        for engine in ("omnivoice", "voxcpm2"):
            for language in ("ja", "en"):
                voice_design.validate_voice_design(engine, language)
            with self.assertRaises(PipelineError):
                voice_design.validate_voice_design(engine, "fr")
        for engine in ("irodori", "irodori_tts", "irodori-tts"):
            voice_design.validate_voice_design(engine, "ja")
            with self.assertRaises(PipelineError):
                voice_design.validate_voice_design(engine, "en")
        with self.assertRaises(PipelineError):
            voice_design.validate_voice_design("unsupported", "ja")

    def test_namespaces_isolate_modes_languages_engines_and_seeds(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            video = root / "video.mp4"
            stack.enter_context(mock.patch.object(cli, "TEMP_ROOT", root))
            stack.enter_context(mock.patch.object(cli, "get_asr_engine", return_value="nemo-speech"))
            stack.enter_context(mock.patch.object(config, "OUTPUT_SUFFIX", "_dub.mp4"))
            paths = set()
            outputs = set()
            for separated in (True, False):
                for engine, language in (("voxcpm2", "ja"), ("voxcpm2", "en"), ("omnivoice", "ja"), ("irodori", "ja")):
                    for seed in (42, 43):
                        with (
                            mock.patch.object(cli, "ENABLE_AUDIO_SEPARATION", separated),
                            mock.patch.object(cli, "ENABLE_VOICE_DESIGN", True),
                            mock.patch.multiple(config, ENABLE_VOICE_DESIGN=True,
                                                TTS_ENGINE=engine, OUTPUT_LANG=language,
                                                VOICE_DESIGN_SEED=seed),
                        ):
                            path = cli._work_dir_for_video(video)
                            self.assertIn("nemo-speech/voice_design/", str(path))
                            self.assertIn("video_rawaudio" if not separated else "video/", str(path))
                            paths.add(path)
                            outputs.add(voice_design.output_path_for_video(video))
            self.assertEqual(len(paths), 16)
            self.assertEqual(len(outputs), 8)
            with mock.patch.object(config, "ENABLE_VOICE_DESIGN", False):
                self.assertEqual(voice_design.output_path_for_video(video), root / "video_dub.mp4")

    def test_pipeline_designs_after_translation_release_using_correct_audio(self):
        for separated in (True, False):
            with self.subTest(separated=separated):
                self._assert_pipeline_source(separated)

    def _assert_pipeline_source(self, separated):
        class StopAtTTS(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            vocals, background = root / "vocals.wav", root / "background.wav"
            vocals.touch()
            background.touch()
            segments = [Segment(0, 1, 4, "Hello", "こんにちは。", "A", "en")]
            diarization = [DiarizationSegment(1, 4, "A")]
            cache = mock.Mock(cache_dir=root / "work" / "speaker_refs")
            events = []

            def design(source, supplied, supplied_cache):
                self.assertEqual(events, ["translate", "release"])
                self.assertEqual(source, vocals if separated else video)
                self.assertEqual(supplied, segments)
                self.assertIs(supplied_cache, cache)
                events.append("design")
                return "voice_fingerprint"

            def tts(_segments, audio_dir, work_dir, _progress, supplied_cache):
                self.assertEqual(events, ["translate", "release", "design"])
                self.assertEqual(work_dir.parent, root / "work" / "render")
                self.assertEqual(audio_dir, work_dir / "seg_audio")
                self.assertIs(supplied_cache, cache)
                raise StopAtTTS

            with ExitStack() as stack:
                replacements = {
                    "ENABLE_VOICE_DESIGN": True, "TTS_ENGINE": "voxcpm2", "OUTPUT_LANG": "ja",
                    "get_asr_engine": lambda: "nemo-speech", "ENABLE_AUDIO_SEPARATION": separated,
                    "ffprobe_duration_sec": mock.Mock(return_value=10),
                    "ffprobe_has_audio": mock.Mock(return_value=True),
                    "separate_voice_and_background": mock.Mock(return_value=(vocals, background)),
                    "extract_wav_for_whisper": lambda _source, target: target.touch(),
                    "force_memory_cleanup": mock.Mock(), "merge_segments": lambda value: value,
                    "chunk_segments_for_spacy": lambda value, **_kwargs: value,
                    "split_segments_by_spacy_sentences": lambda value: value,
                    "merge_sentence_units": lambda value, **_kwargs: value,
                    "translate_segments_resumable": lambda *args, **kwargs: events.append("translate") or segments,
                    "release_all_translation_models": lambda: events.append("release"),
                    "_run_tts_voxcpm2": tts,
                }
                for name, value in replacements.items():
                    stack.enter_context(mock.patch.object(pipeline, name, value))
                stack.enter_context(mock.patch.multiple(config, ENABLE_VOICE_DESIGN=True,
                                                        TTS_ENGINE="voxcpm2", OUTPUT_LANG="ja"))
                stack.enter_context(mock.patch.object(nemo_speech, "nemo_speech_transcribe",
                                                      return_value=(segments, diarization)))
                stack.enter_context(mock.patch.object(voice_design, "build_designed_references", side_effect=design))
                with self.assertRaises(StopAtTTS):
                    pipeline.process_one_video(video, mock.Mock(), cache)
                cache.build_omnivoice_references.assert_not_called()
                cache.build_omnivoice_segment_references.assert_not_called()
                cache.build_irodori_speaker_references.assert_not_called()


if __name__ == "__main__":
    unittest.main()
