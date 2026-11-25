"""Regression tests for source identity, recoverable TTS and publication."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing import cli, config
from xlanguage_dubbing.core import cache, pipeline
from xlanguage_dubbing.core.models import Segment, TtsMeta
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache
from xlanguage_dubbing.tts.voice_design import output_path_for_video
from xlanguage_dubbing.utils import PipelineError


def write_audio(path: Path, *, rate: int = 48000, channels: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full((rate // 10, channels), 0.01, dtype=np.float32), rate)


class CacheIdentityTests(unittest.TestCase):
    def test_folder_scan_does_not_redub_generated_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.touch()
            (root / ("generated" + config.OUTPUT_SUFFIX)).touch()
            completed = root / "other-output.mp4"
            completed.touch()
            cache.completion_path(completed).write_text("{}")
            self.assertEqual(cli.list_videos(root), [source])

    def test_same_name_inputs_replacements_and_active_settings_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a" / "clip.mp4", root / "b" / "clip.mp4"
            a.parent.mkdir()
            b.parent.mkdir()
            a.write_bytes(b"original")
            b.write_bytes(b"original")
            with mock.patch.object(cli, "TEMP_ROOT", root / "temp"):
                first = cli._work_dir_for_video(a)
                self.assertNotEqual(first, cli._work_dir_for_video(b))
                a.write_bytes(b"replaced")
                self.assertNotEqual(first, cli._work_dir_for_video(a))
                first = cache.input_namespace(a)
                with mock.patch.object(config, "ASR_ENGINE", "whisper"):
                    whisper = cache.input_namespace(a)
                with mock.patch.object(config, "ASR_ENGINE", "vibevoice"):
                    self.assertNotEqual(whisper, cache.input_namespace(a))
                self.assertEqual(first, cache.input_namespace(a))

    def test_final_names_separate_engines_audio_modes_and_mix_settings(self):
        video = Path("/tmp/name-only-test.mp4")
        with mock.patch.multiple(
            config,
            ENABLE_VOICE_DESIGN=False,
            TTS_ENGINE="omnivoice",
            ENABLE_AUDIO_SEPARATION=False,
        ):
            original = output_path_for_video(video)
            for name, value in (
                ("TTS_ENGINE", "voxcpm2"),
                ("ENABLE_AUDIO_SEPARATION", True),
                ("OMNIVOICE_NUM_STEP", 999),
                ("ORIGINAL_VOLUME", 0.77),
            ):
                with self.subTest(setting=name), mock.patch.object(config, name, value):
                    self.assertNotEqual(original, output_path_for_video(video))

    def test_completion_rejects_unverified_corrupt_or_changed_references(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.multiple(
            config, TTS_ENGINE="omnivoice", ENABLE_VOICE_DESIGN=False, KEEP_TEMP=True,
        ):
            root = Path(directory)
            video, output = root / "input.mp4", root / "output.mp4"
            video.write_bytes(b"input")
            output.write_bytes(b"validated output")
            refs = root / "refs"
            reference = refs / "ovref_speaker.wav"
            write_audio(reference)
            self.assertFalse(cache.completed_output_is_valid(output, video, refs))
            cache.publish_completion(output, video, refs)
            self.assertTrue(cache.completed_output_is_valid(output, video, refs))
            (refs / "unrelated_reference.wav").write_bytes(b"another engine")
            self.assertTrue(cache.completed_output_is_valid(output, video, refs))
            reference.write_bytes(b"changed reference")
            self.assertFalse(cache.completed_output_is_valid(output, video, refs))
            cache.publish_completion(output, video, refs)
            output.write_bytes(b"corrupt")
            self.assertFalse(cache.completed_output_is_valid(output, video, refs))

    def test_completion_survives_intentional_temporary_cache_removal(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(config, "KEEP_TEMP", False),
        ):
            root = Path(directory)
            video, output = root / "input.mp4", root / "output.mp4"
            video.write_bytes(b"input")
            output.write_bytes(b"validated output")
            references = root / "refs"
            write_audio(references / "speaker.wav")
            cache.publish_completion(output, video, references)
            references.rename(root / "archived_refs")
            references.mkdir()
            self.assertTrue(cache.completed_output_is_valid(output, video, references))

    def test_changed_input_never_reuses_legacy_extracted_audio(self):
        class StopAfterExtraction(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            video, work = root / "input.mp4", root / "work"
            work.mkdir()
            video.write_bytes(b"original")
            stack.enter_context(
                mock.patch.multiple(
                    config,
                    ASR_ENGINE="nemo-speech",
                    ENABLE_VOICE_DESIGN=False,
                    TTS_ENGINE="omnivoice",
                )
            )
            progress = ProgressStore(work / "progress.json", video)
            progress.save()
            old_audio = work / "audio_original_16k.wav"
            write_audio(old_audio, rate=16000, channels=1)
            video.write_bytes(b"replacement")
            stack.enter_context(
                mock.patch.multiple(
                    pipeline,
                    ENABLE_AUDIO_SEPARATION=False,
                    ENABLE_VOICE_DESIGN=False,
                    TTS_ENGINE="omnivoice",
                )
            )
            stack.enter_context(
                mock.patch.object(pipeline, "ffprobe_duration_sec", return_value=1)
            )
            stack.enter_context(
                mock.patch.object(pipeline, "ffprobe_has_audio", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(pipeline, "get_asr_engine", return_value="nemo-speech")
            )
            extract = stack.enter_context(
                mock.patch.object(
                    pipeline,
                    "extract_wav_for_whisper",
                    side_effect=lambda _source, target: write_audio(
                        target, rate=16000, channels=1
                    ),
                )
            )
            transcribe = stack.enter_context(
                mock.patch(
                    "xlanguage_dubbing.asr.nemo_speech.nemo_speech_transcribe",
                    side_effect=StopAfterExtraction,
                )
            )
            with self.assertRaises(StopAfterExtraction):
                pipeline.process_one_video(
                    video, mock.Mock(), SpeakerReferenceCache(work / "speaker_refs")
                )
            self.assertEqual(extract.call_args.args[0], video)
            self.assertNotEqual(transcribe.call_args.args[0], old_audio)
            self.assertTrue(old_audio.exists())


class RecoverableTtsTests(unittest.TestCase):
    def test_failure_does_not_complete_and_resume_retries_only_failed_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            progress = ProgressStore(root / "progress.json", video)
            (root / "tts_meta.json").write_text('[{"segno": "broken", "duration_sec": "invalid"}]')
            segments = [Segment(i, i, i + 1, "source", "translated", "A", "en") for i in range(2)]
            generated = []

            def generate(_segment, stub, number):
                generated.append(number)
                if number == 1 and generated == [1]:
                    raise RuntimeError("temporary synthesis failure")
                audio = stub.with_suffix(".flac")
                write_audio(audio)
                return TtsMeta(number, str(audio), 0.1)

            audio_dir = root / "audio"
            args = (segments, audio_dir, root, progress, lambda *args: "TTS", generate)
            with self.assertRaises(PipelineError):
                pipeline._run_tts_loop(*args)
            self.assertEqual(progress.step("tts"), {"done_count": 1, "total": 2, "failed": [1]})
            self.assertEqual(set(pipeline.load_tts_meta(root / "tts_meta.json")), {2})
            generated.clear()

            # On retry the failure has been resolved; the valid second turn is reused.
            def recovered(segment, stub, number):
                generated.append(number)
                audio = stub.with_suffix(".flac")
                write_audio(audio)
                return TtsMeta(number, str(audio), 0.1)

            pipeline._run_tts_loop(*args[:-1], recovered)
            self.assertEqual(generated, [1])
            self.assertEqual(progress.step("tts")["done_count"], 2)
            self.assertEqual(set(pipeline.load_tts_meta(root / "tts_meta.json")), {1, 2})

    def test_interrupted_generation_never_publishes_partial_flac(self):
        class Interrupted(BaseException):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            output = root / "audio" / "seg_00001.flac"
            output.parent.mkdir()
            output.write_bytes(b"previous corrupt audio")

            def generate(_segment, stub, _number):
                write_audio(stub.with_suffix(".flac"))
                raise Interrupted

            with self.assertRaises(Interrupted):
                pipeline._run_tts_loop(
                    [Segment(0, 0, 1, "source", "translated")],
                    output.parent,
                    root,
                    ProgressStore(root / "progress.json", video),
                    lambda *args: "TTS",
                    generate,
                )
            self.assertEqual(output.read_bytes(), b"previous corrupt audio")
            self.assertFalse(list(output.parent.glob(".tts-*")))
            self.assertEqual(pipeline.load_tts_meta(root / "tts_meta.json"), {})


if __name__ == "__main__":
    unittest.main()
