from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing import cli, config
from xlanguage_dubbing.audio.segment_io import save_segments_json_atomic
from xlanguage_dubbing.core import pipeline
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.lang_utils import (
    LANG_CODE_TO_NAME,
    normalize_lang_code,
    speech_lang_code,
    validate_translation_language,
)
from xlanguage_dubbing.translation import plamo
from xlanguage_dubbing.tts.voice_design import output_path_for_video
from xlanguage_dubbing.utils import PipelineError


class PlamoTranslationTests(unittest.TestCase):
    def test_official_language_set_and_chinese_script_aliases(self):
        self.assertEqual(len(LANG_CODE_TO_NAME), 53)
        for code in LANG_CODE_TO_NAME:
            self.assertEqual(validate_translation_language(code), code)
        for alias, expected in {
            "zh": "zh-hans", "zh-CN": "zh-hans", "zh_Hans": "zh-hans",
            "zh-TW": "zh-hant", "zh-HK": "zh-hant", "zh-Hant-TW": "zh-hant",
            "EN-us": "en",
        }.items():
            self.assertEqual(normalize_lang_code(alias), expected)
            self.assertEqual(speech_lang_code(alias), expected.split("-")[0])
        for code in ("uk", "pt", "fi", "auto", "", "xx"):
            with self.assertRaises(PipelineError):
                validate_translation_language(code)

    def test_pfml3_prompt_is_sent_raw_to_completion_endpoint(self):
        server = mock.Mock()
        server.request.return_value = {"choices": [{"text": "  会議は月曜日です。  ", "finish_reason": "stop"}]}
        with mock.patch.object(plamo, "_get_server", return_value=server):
            self.assertEqual(plamo._translate_text("The meeting is on Monday.", "en", "ja"), "会議は月曜日です。")
        endpoint, payload = server.request.call_args.args
        self.assertEqual(endpoint, "/v1/completions")
        prompt = payload["prompt"]
        self.assertEqual(prompt, (
            "<|plamo:tag|>dataset<|plamo:msg|>translation"
            "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en"
            "<|plamo:msg|>The meeting is on Monday."
            "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja<|plamo:msg|>"
        ))
        self.assertNotIn("<|plamo:bos|>", prompt)
        self.assertEqual(payload["stop"], ["<|plamo:tag|>", "<|plamo:bos|>"])
        self.assertEqual(payload["temperature"], 0)

    def test_all_pairs_use_same_model_and_unsupported_fails_before_loading(self):
        client = plamo.PlamoTranslateClient()
        with mock.patch.object(plamo, "_translate_text", return_value="translated") as infer:
            for source, target in (("en", "ja"), ("ja", "en"), ("fr", "de"), ("zh-TW", "ko")):
                self.assertEqual(client.translate("source", source, target), ("translated", "stop"))
                infer.assert_called_with("source", normalize_lang_code(source), normalize_lang_code(target))
            count = infer.call_count
            for source, target in (("pt", "ja"), ("en", "uk")):
                with self.assertRaises(PipelineError):
                    client.translate("source", source, target)
            self.assertEqual(infer.call_count, count)

    def test_empty_or_truncated_output_is_not_published_as_translation(self):
        server = mock.Mock()
        with mock.patch.object(plamo, "_get_server", return_value=server):
            for choice in ({"text": "", "finish_reason": "stop"}, {"text": "partial", "finish_reason": "length"}):
                server.request.return_value = {"choices": [choice]}
                with self.assertRaises(plamo.PlamoTranslateError):
                    plamo._translate_text("Hello", "en", "ja")

    def test_old_checkpoint_is_retranslated_and_current_checkpoint_resumes(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(plamo, "OUTPUT_LANG", "ja"):
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            path = root / "segments.json"
            progress = ProgressStore(root / "progress.json", video)
            source = Segment(0, 0, 2, "Hello", "", "speaker", "en")
            save_segments_json_atomic([replace(source, text_tgt="old model output")], path)
            progress.set_step("translate_done", True)
            client = mock.Mock()
            client.translate.return_value = ("こんにちは。", "stop")
            result = plamo.translate_segments_resumable(client, [source], path, progress)
            self.assertEqual(result[0].text_tgt, "こんにちは。")
            client.translate.assert_called_once()
            client.reset_mock()
            self.assertEqual(plamo.translate_segments_resumable(client, [source], path, progress), result)
            client.translate.assert_not_called()
            changed = replace(source, text_src="Good evening")
            plamo.translate_segments_resumable(client, [changed], path, progress)
            client.translate.assert_called_once_with("Good evening", source_lang="en", target_lang="ja")

    def test_failed_first_inference_cannot_publish_old_model_cache_on_restart(self):
        from xlanguage_dubbing.audio.segment_io import load_segments_json

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(plamo, "OUTPUT_LANG", "ja"):
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            path = root / "segments.json"
            progress = ProgressStore(root / "progress.json", video)
            source = Segment(0, 0, 2, "Hello", "", "speaker", "en")
            save_segments_json_atomic([replace(source, text_tgt="old output")], path)
            client = mock.Mock()
            client.translate.side_effect = RuntimeError("interrupted inference")
            with self.assertRaises(RuntimeError):
                plamo.translate_segments_resumable(client, [source], path, progress)
            self.assertEqual(load_segments_json(path)[0].text_tgt, "")
            client.translate.side_effect = None
            client.translate.return_value = ("こんにちは。", "stop")
            result = plamo.translate_segments_resumable(client, [source], path, progress)
            self.assertEqual(result[0].text_tgt, "こんにちは。")

    def test_unsupported_cached_segment_fails_even_when_translation_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            path = root / "segments.json"
            progress = ProgressStore(root / "progress.json", video)
            source = Segment(0, 0, 2, "source", "translated", "speaker", "uk")
            save_segments_json_atomic([source], path)
            progress.set_artifact("translation_namespace", plamo.translation_namespace())
            with self.assertRaises(PipelineError):
                plamo.translate_segments_resumable(mock.Mock(), [source], path, progress)

    def test_preflight_rejects_unsupported_config_before_media_checks(self):
        for source, target in (("uk", "ja"), ("auto", "pt")):
            with (
                mock.patch.multiple(cli, INPUT_LANG=source, OUTPUT_LANG=target),
                mock.patch.object(cli, "which_or_raise") as media_check,
                self.assertRaises(PipelineError),
            ):
                cli.preflight_checks()
            media_check.assert_not_called()

    def test_namespace_separates_old_outputs_languages_and_settings(self):
        video = Path("/tmp/video.mp4")
        with mock.patch.multiple(config, ENABLE_VOICE_DESIGN=False, TTS_ENGINE="omnivoice", INPUT_LANG="auto", OUTPUT_LANG="ja"):
            initial = plamo.translation_namespace()
            self.assertIn(initial, str(cli._work_dir_for_video(video)))
            self.assertIn(initial, output_path_for_video(video).name)
            self.assertNotEqual(output_path_for_video(video), video.with_name("video" + config.OUTPUT_SUFFIX))
            for name, value in (("OUTPUT_LANG", "en"), ("INPUT_LANG", "fr"), ("PLAMO_TRANSLATE_N_CTX", 4096)):
                with mock.patch.object(config, name, value):
                    self.assertNotEqual(plamo.translation_namespace(), initial)

    def test_pipeline_preserves_explicit_chinese_script_from_generic_asr(self):
        class StopAtTranslation(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.touch()
            work = root / "work"
            work.mkdir()
            segments = [Segment(0, 0, 2, "你好", "", "speaker", "zh")]
            save_segments_json_atomic(segments, work / "segments_src.json")
            progress = ProgressStore(work / "progress.json", video)
            progress.set_step("asr_done", True)
            progress.set_artifact("detected_lang", "zh")
            progress.save()
            cache = mock.Mock(cache_dir=work / "speaker_refs")
            with (
                mock.patch.multiple(
                    pipeline, INPUT_LANG="zh-hant", OUTPUT_LANG="ja", ENABLE_VOICE_DESIGN=True,
                    TTS_ENGINE="voxcpm2", ENABLE_AUDIO_SEPARATION=False,
                ),
                mock.patch.object(pipeline, "ffprobe_duration_sec", return_value=2),
                mock.patch.object(pipeline, "ffprobe_has_audio", return_value=True),
                mock.patch.object(pipeline, "extract_wav_for_whisper", side_effect=lambda _source, target: sf.write(target, np.zeros(160), 16000)),
                mock.patch.object(pipeline, "get_asr_engine", return_value="nemo-speech"),
                mock.patch.object(pipeline, "release_translation_model"),
                mock.patch.object(pipeline, "force_memory_cleanup"),
                mock.patch.object(pipeline, "translate_segments_resumable", side_effect=StopAtTranslation) as translate,
                self.assertRaises(StopAtTranslation),
            ):
                pipeline.process_one_video(video, mock.Mock(), cache)
            self.assertEqual(translate.call_args.args[1][0].detected_lang, "zh-hant")
            self.assertEqual(translate.call_args.kwargs["detected_lang"], "zh-hant")

    def test_release_stops_owned_server(self):
        server = mock.Mock()
        with mock.patch.object(plamo, "_SERVER", server), mock.patch.object(plamo, "force_memory_cleanup"):
            plamo.release_translation_model()
            server.close.assert_called_once()
            self.assertIsNone(plamo._SERVER)

    def test_server_starts_privately_and_closes_only_its_own_process(self):
        with (
            mock.patch.object(plamo, "resolve_llama_server", return_value="/opt/homebrew/bin/llama-server"),
            mock.patch.object(plamo, "_get_model_path", return_value="/tmp/model.gguf"),
            mock.patch.object(plamo.socket, "socket") as listener,
            mock.patch.object(plamo.subprocess, "Popen") as popen,
            mock.patch.object(plamo._PlamoServer, "request", return_value={"status": "ok"}),
        ):
            listener.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            process = popen.return_value
            process.poll.return_value = None
            server = plamo._PlamoServer()
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
            self.assertEqual(command[command.index("--port") + 1], "12345")
            self.assertEqual(command[command.index("--gpu-layers") + 1], str(plamo.PLAMO_TRANSLATE_N_GPU_LAYERS))
            self.assertIn("--api-key", command)
            server.close()
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=10)
            process.kill.assert_not_called()
            self.assertTrue(server.log.closed)

    def test_failed_server_start_closes_log_and_reports_native_error(self):
        with (
            mock.patch.object(plamo, "resolve_llama_server", return_value="llama-server"),
            mock.patch.object(plamo, "_get_model_path", return_value="/tmp/model.gguf"),
            mock.patch.object(plamo.socket, "socket") as listener,
            mock.patch.object(plamo.subprocess, "Popen") as popen,
        ):
            listener.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 12345)
            popen.return_value.poll.return_value = 1
            with self.assertRaisesRegex(PipelineError, "起動に失敗"):
                plamo._PlamoServer()
            self.assertTrue(popen.call_args.kwargs["stdout"].closed)
            popen.return_value.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
