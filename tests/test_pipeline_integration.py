"""Exercise the production pipeline with deterministic synthetic speech/media."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing import cli, config
from xlanguage_dubbing.audio import demucs, ffmpeg
from xlanguage_dubbing.core import cache, pipeline
from xlanguage_dubbing.core.models import DiarizationSegment, Segment, TtsMeta
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.translation import plamo
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache
from xlanguage_dubbing.tts.voice_design import output_path_for_video
from xlanguage_dubbing.utils import PipelineError, run_cmd


class PipelineMediaTests(unittest.TestCase):
    def test_real_retime_mux_and_resume_in_both_source_modes(self):
        for separated in (False, True):
            with (
                self.subTest(separated=separated),
                tempfile.TemporaryDirectory() as directory,
                ExitStack() as stack,
            ):
                root = Path(directory)
                video = root / "input.mp4"
                run_cmd(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=blue:s=160x90:r=30:d=3",
                        "-f",
                        "lavfi",
                        "-i",
                        "sine=frequency=440:sample_rate=48000:duration=3",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        str(video),
                    ]
                )
                stack.enter_context(
                    mock.patch.multiple(
                        config,
                        ASR_ENGINE="nemo-speech",
                        TTS_ENGINE="omnivoice",
                        INPUT_LANG="en",
                        OUTPUT_LANG="ja",
                        ENABLE_AUDIO_SEPARATION=separated,
                        ENABLE_VOICE_DESIGN=False,
                        OUTPUT_SIZE=90,
                        ORIGINAL_VOLUME=0.17,
                    )
                )
                stack.enter_context(
                    mock.patch.multiple(
                        pipeline,
                        TTS_ENGINE="omnivoice",
                        INPUT_LANG="en",
                        OUTPUT_LANG="ja",
                        ENABLE_AUDIO_SEPARATION=separated,
                        ENABLE_VOICE_DESIGN=False,
                        ORIGINAL_VOLUME=0.17,
                        KEEP_TEMP=True,
                    )
                )
                stack.enter_context(mock.patch.object(cli, "TEMP_ROOT", root / "temp"))
                stack.enter_context(
                    mock.patch.multiple(
                        cli, ENABLE_AUDIO_SEPARATION=separated, ENABLE_VOICE_DESIGN=False
                    )
                )
                stack.enter_context(
                    mock.patch.object(pipeline, "get_asr_engine", return_value="nemo-speech")
                )
                stack.enter_context(
                    mock.patch.object(cli, "get_asr_engine", return_value="nemo-speech")
                )
                stack.enter_context(mock.patch.object(ffmpeg, "OUTPUT_SIZE", 90))
                segments = [
                    Segment(0, 0, 1, "Hello.", speaker_id="A", detected_lang="en"),
                    Segment(1, 0.25, 0.8, "Good morning.", speaker_id="B", detected_lang="en"),
                ]
                turns = [DiarizationSegment(s.start, s.end, s.speaker_id) for s in segments]
                transcribe = stack.enter_context(
                    mock.patch(
                        "xlanguage_dubbing.asr.nemo_speech.nemo_speech_transcribe",
                        return_value=(segments, turns),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "xlanguage_dubbing.asr.transcribe_reference_audio", return_value="Hello."
                    )
                )
                for name in ("merge_segments", "split_segments_by_spacy_sentences"):
                    stack.enter_context(
                        mock.patch.object(pipeline, name, side_effect=lambda value: value)
                    )
                for name in ("chunk_segments_for_spacy", "merge_sentence_units"):
                    stack.enter_context(
                        mock.patch.object(
                            pipeline, name, side_effect=lambda value, **kwargs: value
                        )
                    )
                translated = [replace(s, text_tgt="こんにちは。") for s in segments]
                stack.enter_context(
                    mock.patch.object(
                        pipeline, "translate_segments_resumable", return_value=translated
                    )
                )
                stack.enter_context(mock.patch.object(pipeline, "release_translation_model"))
                stack.enter_context(mock.patch.object(pipeline, "force_memory_cleanup"))
                stack.enter_context(
                    mock.patch("xlanguage_dubbing.omnivoice_tts.release_omnivoice_model")
                )

                def synthesize(_segment, stub, _reference, segno=0):
                    path = stub.with_suffix(".flac")
                    duration = 0.6 if segno == 1 else 0.7
                    tone = (
                        np.sin(np.arange(round(duration * config.TTS_SAMPLE_RATE)) * 0.08) * 0.05
                    )
                    sf.write(
                        path,
                        np.repeat(tone[:, None], config.TTS_CHANNELS, axis=1),
                        config.TTS_SAMPLE_RATE,
                    )
                    return TtsMeta(segno, str(path), duration)

                generate = stack.enter_context(
                    mock.patch(
                        "xlanguage_dubbing.omnivoice_tts.generate_segment_tts_omnivoice",
                        side_effect=synthesize,
                    )
                )
                mux = stack.enter_context(
                    mock.patch.object(
                        pipeline,
                        "mux_retimed_video_with_tracks",
                        wraps=pipeline.mux_retimed_video_with_tracks,
                    )
                )
                if separated:
                    wav = root / "stem.wav"
                    run_cmd(
                        [
                            "ffmpeg",
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            str(video),
                            "-vn",
                            "-c:a",
                            "pcm_s16le",
                            str(wav),
                        ]
                    )
                    stack.enter_context(
                        mock.patch.object(
                            pipeline, "separate_voice_and_background", return_value=(wav, wav)
                        )
                    )
                work = cli._work_dir_for_video(video)
                references = SpeakerReferenceCache(work / "speaker_refs", tts_engine="omnivoice")
                pipeline.process_one_video(video, mock.Mock(), references)
                output = output_path_for_video(video)
                self.assertTrue(
                    ffmpeg.validate_media_file(output, require_video=True, require_audio=True)
                )
                self.assertTrue(
                    cache.completed_output_is_valid(output, video, references.cache_dir)
                )
                self.assertEqual(
                    mux.call_args.kwargs["original_volume"], 1.0 if separated else 0.17
                )
                self.assertEqual(
                    set(pipeline.load_tts_meta(next(work.glob("render/*/tts_meta.json")))), {1, 2}
                )
                pipeline.process_one_video(video, mock.Mock(), references)
                self.assertEqual(transcribe.call_count, 1)
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(mux.call_count, 1)
                if not separated:
                    # Repair two corrupt utterances with changed individual durations
                    # but unchanged total length: old joined video must not be reused.
                    cache.completion_path(output).write_text("{}")
                    for path in work.glob("render/*/seg_audio/*.flac"):
                        path.write_bytes(b"interrupted audio")

                    def regenerated(segment, stub, supplied, segno=0):
                        meta = synthesize(segment, stub, supplied, segno=3 - segno)
                        return replace(meta, segno=segno)

                    generate.side_effect = regenerated
                    pipeline.process_one_video(video, mock.Mock(), references)
                    self.assertEqual(
                        len(list(work.glob("render/*/retime/*/video_retimed.mp4"))), 2
                    )
                    self.assertTrue(
                        cache.completed_output_is_valid(output, video, references.cache_dir)
                    )
                    self.assertEqual(mux.call_count, 2)


class TranslationRecordTests(unittest.TestCase):
    def test_interrupt_resumes_each_committed_record_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            path = root / "translation.json"
            progress = ProgressStore(root / "progress.json", video)
            source = [Segment(i, i, i + 1, f"source {i}", detected_lang="en") for i in range(3)]
            client = mock.Mock()
            client.translate.side_effect = [
                ("translated first", "stop"),
                RuntimeError("interrupted"),
            ]
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                plamo.translate_segments_resumable(client, source, path, progress)
            self.assertFalse(progress.step("translate_done"))
            self.assertTrue((root / "translation.records" / "00001.json").is_file())
            client.reset_mock()
            client.translate.side_effect = [
                ("translated second", "stop"),
                ("translated third", "stop"),
            ]
            result = plamo.translate_segments_resumable(client, source, path, progress)
            self.assertEqual(client.translate.call_count, 2)
            self.assertEqual(result[0].text_tgt, "translated first")
            client.reset_mock()
            client.translate.side_effect = [("changed", "stop")]
            changed = [replace(source[0], text_src="changed source"), *source[1:]]
            plamo.translate_segments_resumable(client, changed, path, progress)
            self.assertEqual(client.translate.call_count, 1)


class DemucsPublicationTests(unittest.TestCase):
    def test_failed_separation_never_replaces_cached_stems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            demucs_root = root / "demucs"
            demucs_root.mkdir()
            sf.write(demucs_root / "demucs_input.wav", np.zeros((4410, 2)), 44100)
            stems = demucs_root / demucs.DEMUCS_MODEL / "demucs_input"
            stems.mkdir(parents=True)
            voice = stems / "vocals.wav"
            voice.write_bytes(b"old partial voice")

            def interrupted(command):
                self.assertEqual(command[command.index("--two-stems") + 1], "vocals")
                self.assertEqual(command[command.index("-d") + 1], demucs.DEMUCS_DEVICE)
                staged = (
                    Path(command[command.index("-o") + 1]) / demucs.DEMUCS_MODEL / "demucs_input"
                )
                staged.mkdir(parents=True)
                (staged / "vocals.wav").write_bytes(b"new partial voice")
                raise PipelineError("separation failed")

            with (
                mock.patch.object(demucs, "run_cmd", side_effect=interrupted),
                self.assertRaises(PipelineError),
            ):
                demucs.separate_voice_and_background(root / "input.mp4", root)
            self.assertEqual(voice.read_bytes(), b"old partial voice")
            self.assertFalse((stems / "no_vocals.wav").exists())


if __name__ == "__main__":
    unittest.main()
