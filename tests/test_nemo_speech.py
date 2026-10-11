from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from xlanguage_dubbing import asr, cli
from xlanguage_dubbing.asr import nemo_speech as nemo
from xlanguage_dubbing.core import pipeline
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.utils import PipelineError


def _word(text: str, start: float, end: float, speaker: int = 1) -> dict:
    return {"word": text, "start": start, "end": end, "speaker": speaker}


class NemoSpeechTests(unittest.TestCase):
    def test_turns_preserve_speakers_pauses_and_language(self) -> None:
        segments, diarization = nemo._parse_nemo_json({
            "languages": ["en-US"],
            "words": [
                _word("Hello", 1.0, 1.2), _word("there.", 1.2, 1.5),
                _word("Hi!", 1.5, 1.8, 2), _word("Again.", 4.0, 4.5, 2),
            ],
        })
        self.assertEqual([s.text_src for s in segments], ["Hello there.", "Hi!", "Again."])
        self.assertEqual([s.speaker_id for s in segments], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"])
        self.assertEqual((segments[0].start, segments[0].end), (1.0, 1.5))
        self.assertTrue(all(s.detected_lang == "en" for s in segments))
        self.assertEqual(diarization[0], DiarizationSegment(1.0, 1.5, "SPEAKER_00"))

    def test_all_eight_speakers_keep_distinct_ids(self) -> None:
        segments, diarization = nemo._parse_nemo_json({
            "words": [_word("Hello", float(i), i + 0.5, i) for i in range(1, 9)],
        })
        self.assertEqual(len({s.speaker_id for s in segments}), 8)
        self.assertEqual(diarization[-1].speaker, "SPEAKER_07")
        with self.assertRaisesRegex(PipelineError, "1〜8"):
            nemo._parse_nemo_json({"words": [_word("Hello", 0, 1, 9)]})

    def test_zero_duration_punctuation_is_attached_without_inventing_a_speaker(self) -> None:
        segments, _ = nemo._parse_nemo_json({
            "words": [_word("Hello", 0, 0.5), _word("!", 0.5, 0.5, 0)],
        })
        self.assertEqual(segments[0].text_src, "Hello!")
        self.assertEqual(segments[0].speaker_id, "SPEAKER_00")

    def test_missing_speaker_does_not_mislabel_words(self) -> None:
        segments, diarization = nemo._parse_nemo_json({
            "words": [_word("Hello", 0, 0.5, 0), _word("World", 0.5, 1)],
        })
        self.assertEqual(segments[0].speaker_id, "")
        self.assertEqual(len(diarization), 1)
        with self.assertRaisesRegex(PipelineError, "結果が空"):
            nemo._parse_nemo_json({"words": [_word("Hello", 0, 1, 0)]})

    def test_malformed_timestamps_and_schema_fail_clearly(self) -> None:
        for data in (
            {}, {"words": []}, {"words": ["word"]},
            {"words": [_word("Hi", float("nan"), 1)]},
            {"words": [_word("Hi", 2, 1)]},
            {"words": [_word("Hi", -1, 1)]},
        ):
            with self.subTest(data=data), self.assertRaises(PipelineError):
                nemo._parse_nemo_json(data)

    def test_cli_transcription_publishes_validated_json_and_selects_both_models(self) -> None:
        data = {"text": "Hello", "words": [_word("Hello", 0, 1)]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio with spaces.wav"
            path.write_bytes(b"audio")
            with (
                mock.patch.object(nemo, "resolve_nemo_speech_cli", return_value="/bin/nemo-speech"),
                mock.patch.object(nemo, "INPUT_LANG", "en"),
                mock.patch.object(nemo, "NEMO_SPEECH_MODEL", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
                mock.patch.object(nemo, "NEMO_SPEECH_DIAR_MODEL", "nvidia/Nemotron-3-Diarization"),
                mock.patch.object(nemo, "run_cmd", return_value=subprocess.CompletedProcess(
                    [], 0, stdout=json.dumps(data), stderr="progress on stderr",
                )) as run,
            ):
                segments, _ = nemo.nemo_speech_transcribe(path)
            command = run.call_args.args[0]
            self.assertEqual(command[:3], ["/bin/nemo-speech", "transcribe", str(path.resolve())])
            self.assertIn("--diarize", command)
            self.assertEqual(command[command.index("--model") + 1], "nvidia/nemotron-3.5-asr-streaming-0.6b")
            self.assertEqual(command[command.index("--diar-model") + 1], "nvidia/Nemotron-3-Diarization")
            self.assertEqual(command[-2:], ["--language", "en"])
            self.assertEqual(segments[0].text_src, "Hello")
            self.assertEqual(json.loads((path.parent / "nemo_speech.json").read_text()), data)

    def test_invalid_cli_output_does_not_replace_valid_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            path.touch()
            result_path = path.parent / "nemo_speech.json"
            result_path.write_text("previous result")
            with (
                mock.patch.object(nemo, "resolve_nemo_speech_cli", return_value="/bin/nemo-speech"),
                mock.patch.object(nemo, "run_cmd", return_value=subprocess.CompletedProcess(
                    [], 0, stdout='{"words": []}', stderr="",
                )),
                self.assertRaises(PipelineError),
            ):
                nemo.nemo_speech_transcribe(path)
            self.assertEqual(result_path.read_text(), "previous result")

    def test_reference_dispatch_uses_nemo_without_diarization_or_auto_language_argument(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.wav"
            path.touch()
            with (
                mock.patch.object(asr, "ASR_ENGINE", "nemo-speech"),
                mock.patch.object(nemo, "INPUT_LANG", "auto"),
                mock.patch.object(nemo, "resolve_nemo_speech_cli", return_value="/bin/nemo-speech"),
                mock.patch.object(nemo, "run_cmd", return_value=subprocess.CompletedProcess(
                    [], 0, stdout='{"text": " Hello  world "}', stderr="",
                )) as run,
            ):
                self.assertEqual(asr.transcribe_reference_audio(path), "Hello world")
            command = run.call_args.args[0]
            self.assertNotIn("--diarize", command)
            self.assertNotIn("--diar-model", command)
            self.assertNotIn("--language", command)

    def test_default_binary_is_found_outside_shell_path(self) -> None:
        with (
            mock.patch.object(nemo, "NEMO_SPEECH_BIN", "nemo-speech"),
            mock.patch.object(nemo.shutil, "which", return_value=None),
            mock.patch.object(Path, "is_file", return_value=True),
            mock.patch.object(nemo, "which_or_raise", side_effect=lambda value: value),
        ):
            self.assertEqual(nemo.resolve_nemo_speech_cli(), str(Path.home() / ".local/bin/nemo-speech"))

    def test_work_directories_isolate_engines_and_audio_sources(self) -> None:
        with mock.patch.object(cli, "TEMP_ROOT", Path("temp")):
            for separated, base in ((True, "video"), (False, "video_rawaudio")):
                with mock.patch.object(cli, "ENABLE_AUDIO_SEPARATION", separated):
                    with mock.patch.object(asr, "ASR_ENGINE", "nemo-speech"):
                        self.assertEqual(cli._work_dir_for_video(Path("video.mp4")), Path("temp") / base / "nemo-speech")
                    with mock.patch.object(asr, "ASR_ENGINE", "whisper"):
                        self.assertEqual(cli._work_dir_for_video(Path("video.mp4")), Path("temp") / base)

    def test_pipeline_uses_nemo_speakers_and_correct_audio_for_references(self) -> None:
        # Stop at translation after exercising ASR, segment publication, and reference routing.
        class StopBeforeTranslation(Exception):
            pass

        for separated in (False, True):
            with self.subTest(separated=separated), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                video = root / "video.mp4"
                video.write_bytes(b"video")
                vocals, background = root / "vocals.wav", root / "background.wav"
                vocals.touch()
                background.touch()
                segments = [Segment(0, 1, 4, "Hello world.", speaker_id="SPEAKER_00", detected_lang="en")]
                diarization = [DiarizationSegment(1, 4, "SPEAKER_00")]
                cache = mock.Mock(cache_dir=root / "work" / "speaker_refs")
                with ExitStack() as stack:
                    replacements = {
                        "get_asr_engine": lambda: "nemo-speech",
                        "ENABLE_AUDIO_SEPARATION": separated,
                        "INPUT_LANG": "auto",
                        "TTS_ENGINE": "omnivoice",
                        "ffprobe_duration_sec": mock.Mock(return_value=10),
                        "ffprobe_has_audio": mock.Mock(return_value=True),
                        "separate_voice_and_background": mock.Mock(return_value=(vocals, background)),
                        "extract_wav_for_whisper": mock.Mock(side_effect=lambda _source, target: target.touch()),
                        "force_memory_cleanup": mock.Mock(),
                        "merge_segments": lambda value: value,
                        "chunk_segments_for_spacy": lambda value, **_kwargs: value,
                        "split_segments_by_spacy_sentences": lambda value: value,
                        "merge_sentence_units": lambda value, **_kwargs: value,
                        "translate_segments_resumable": mock.Mock(side_effect=StopBeforeTranslation),
                    }
                    for key, value in replacements.items():
                        stack.enter_context(mock.patch.object(pipeline, key, value))
                    transcribe = stack.enter_context(mock.patch.object(
                        nemo, "nemo_speech_transcribe", return_value=(segments, diarization),
                    ))
                    with self.assertRaises(StopBeforeTranslation):
                        pipeline.process_one_video(video, mock.Mock(), cache)
                    expected_audio = vocals if separated else video
                    replacements["extract_wav_for_whisper"].assert_called_once()
                    self.assertEqual(replacements["extract_wav_for_whisper"].call_args.args[0], expected_audio)
                    transcribe.assert_called_once()
                    cache.build_omnivoice_references.assert_called_once_with(expected_audio, diarization, segments)
                    cache.build_omnivoice_segment_references.assert_called_once_with(expected_audio, segments)
                    progress = json.loads((root / "work" / "progress.json").read_text())
                    self.assertTrue(progress["steps"]["asr_done"])
                    self.assertTrue(progress["steps"]["diarization_done"])
                    self.assertEqual(progress["artifacts"]["detected_lang"], "en")


if __name__ == "__main__":
    unittest.main()
