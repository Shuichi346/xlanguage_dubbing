from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest import mock

from xlanguage_dubbing import cli, config, irodori_tts
from xlanguage_dubbing.asr import nemo_speech
from xlanguage_dubbing.core import pipeline
from xlanguage_dubbing.core.models import DiarizationSegment, Segment, TtsMeta
from xlanguage_dubbing.core.progress import ProgressStore
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
            with mock.patch.multiple(config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="omnivoice"):
                self.assertEqual(voice_design.output_path_for_video(video), root / "video_dub.mp4")

    def test_irodori_original_outputs_isolate_model_and_steps_from_legacy_output(self):
        video = Path("/tmp/video.mp4")
        with mock.patch.multiple(
            config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="irodori", OUTPUT_LANG="ja",
            OUTPUT_SUFFIX="_dub.mp4", IRODORI_HF_CHECKPOINT="Aratako/Irodori-TTS-v4.1-Small-MF",
            IRODORI_TTS_NUM_STEPS=4,
        ):
            original = voice_design.output_path_for_video(video)
            self.assertNotEqual(original, Path("/tmp/video_dub.mp4"))
            self.assertEqual(original, voice_design.output_path_for_video(video))
            with mock.patch.object(config, "IRODORI_TTS_NUM_STEPS", 8):
                self.assertNotEqual(original, voice_design.output_path_for_video(video))
            with mock.patch.object(config, "IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-v4-Small"):
                self.assertNotEqual(original, voice_design.output_path_for_video(video))

    def test_meanflow_ignores_inactive_sway_settings_in_all_namespaces(self):
        video = Path("/tmp/video.mp4")
        with mock.patch.multiple(
            config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="irodori", OUTPUT_LANG="ja",
            IRODORI_HF_CHECKPOINT="Aratako/Irodori-TTS-v4.1-Small-MF",
            IRODORI_TTS_NUM_STEPS=4, IRODORI_TTS_T_SCHEDULE_MODE="sway",
            IRODORI_TTS_SWAY_COEFF=-1.0,
        ):
            settings = voice_design.irodori_synthesis_settings()
            output = voice_design.output_path_for_video(video)
            namespace = voice_design.design_namespace()
            self.assertEqual(settings["sampling"], {"num_steps": 4})
            with mock.patch.multiple(config, IRODORI_TTS_T_SCHEDULE_MODE="linear", IRODORI_TTS_SWAY_COEFF=0.0):
                self.assertEqual(settings, voice_design.irodori_synthesis_settings())
                self.assertEqual(output, voice_design.output_path_for_video(video))
                self.assertEqual(namespace, voice_design.design_namespace())

    def test_irodori_render_fingerprint_tracks_reference_content_and_translated_text(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="irodori", OUTPUT_LANG="ja",
            IRODORI_HF_CHECKPOINT="Aratako/Irodori-TTS-v4.1-Small-MF", IRODORI_TTS_NUM_STEPS=4,
        ):
            latent = Path(directory) / "speaker.pt"
            latent.write_bytes(b"first latent")
            cache = mock.Mock()
            cache.get_irodori_speaker_reference_latent_path.return_value = latent
            segments = [Segment(0, 1, 4, "Hello", "こんにちは。", "A", "en")]
            initial = voice_design.irodori_render_fingerprint(segments, cache)
            self.assertEqual(initial, voice_design.irodori_render_fingerprint(segments, cache))
            latent.write_bytes(b"changed latent")
            self.assertNotEqual(initial, voice_design.irodori_render_fingerprint(segments, cache))
            latent.write_bytes(b"first latent")
            segments[0] = replace(segments[0], text_tgt="こんばんは。")
            self.assertNotEqual(initial, voice_design.irodori_render_fingerprint(segments, cache))

    def test_irodori_designed_namespace_isolates_checkpoint(self):
        with mock.patch.multiple(
            config, ENABLE_VOICE_DESIGN=True, TTS_ENGINE="irodori", OUTPUT_LANG="ja",
            IRODORI_HF_CHECKPOINT="Aratako/Irodori-TTS-v4-Small",
        ):
            namespace = voice_design.design_namespace()
            output = voice_design.output_path_for_video(Path("/tmp/video.mp4"))
            with mock.patch.object(config, "IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-v4.1-Small-MF"):
                self.assertNotEqual(namespace, voice_design.design_namespace())
                self.assertNotEqual(output, voice_design.output_path_for_video(Path("/tmp/video.mp4")))

    def test_irodori_pipeline_does_not_reuse_legacy_output_or_shared_tts_metadata(self):
        class StopAtGeneration(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            (root / "video_dub.mp4").write_bytes(b"legacy output")
            work = root / "work"
            work.mkdir()
            segments = [Segment(0, 1.0, 4.0, "Hello", "こんにちは。", "A", "en")]
            for name in ("segments_src.json", "segments_translated.json"):
                pipeline.save_segments_json_atomic(segments, work / name)
            progress = ProgressStore(work / "progress.json", video)
            progress.set_step("asr_done", True)
            progress.set_step("translate_done", True)
            progress.set_artifact("detected_lang", "en")
            progress.save()
            legacy_flac = work / "seg_audio" / "seg_00001.flac"
            legacy_flac.parent.mkdir()
            legacy_flac.write_bytes(b"legacy audio")
            pipeline.save_tts_meta_atomic(work / "tts_meta.json", {
                1: TtsMeta(segno=1, flac_path=str(legacy_flac), duration_sec=3.0),
            })
            latent = work / "reference.pt"
            latent.write_bytes(b"MF reference")
            cache = mock.Mock(cache_dir=work / "speaker_refs")
            cache.get_irodori_speaker_reference_latent_path.return_value = latent
            cache.get_irodori_speaker_reference_duration.return_value = 3.0
            stack.enter_context(mock.patch.multiple(
                config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="irodori", OUTPUT_LANG="ja",
                OUTPUT_SUFFIX="_dub.mp4", IRODORI_HF_CHECKPOINT="Aratako/Irodori-TTS-v4.1-Small-MF",
                IRODORI_TTS_NUM_STEPS=4,
            ))
            for name, value in {
                "ENABLE_VOICE_DESIGN": False, "TTS_ENGINE": "irodori", "OUTPUT_LANG": "ja",
                "ENABLE_AUDIO_SEPARATION": False, "get_asr_engine": lambda: "nemo-speech",
                "ffprobe_duration_sec": mock.Mock(return_value=10),
                "ffprobe_has_audio": mock.Mock(return_value=True),
                "extract_wav_for_whisper": lambda _source, target: target.touch(),
                "force_memory_cleanup": mock.Mock(), "release_all_translation_models": mock.Mock(),
            }.items():
                stack.enter_context(mock.patch.object(pipeline, name, value))
            stack.enter_context(mock.patch.object(irodori_tts, "ensure_irodori_tts_server"))
            generate = stack.enter_context(mock.patch.object(
                irodori_tts, "generate_segment_tts_irodori", side_effect=StopAtGeneration,
            ))
            with self.assertRaises(StopAtGeneration):
                pipeline.process_one_video(video, mock.Mock(), cache)
            generated_stub = generate.call_args.args[1]
            expected = work / "render" / voice_design.irodori_render_fingerprint(segments, cache)[:20]
            self.assertEqual(generated_stub, expected / "seg_audio" / "seg_00001")
            self.assertTrue(legacy_flac.exists())

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
