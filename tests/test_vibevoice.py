"""Regression coverage for native VibeVoice GGUF input, output, and ownership."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.error import URLError

import numpy as np
import soundfile as sf

from xlanguage_dubbing import asr, config
from xlanguage_dubbing.asr import vibevoice as vibe
from xlanguage_dubbing.core import cache
from xlanguage_dubbing.core.models import DiarizationSegment
from xlanguage_dubbing.utils import PipelineError


def _turn(text: str = "Hello", start: float = 0, end: float = 0.5, speaker: int = 0) -> dict:
    return {"Start time": start, "End time": end, "Speaker ID": speaker, "Content": text}


def _response(text: str, finish_reason: str = "stop") -> dict:
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": text}}]}


def _write_audio(path: Path, *, rate: int = 16000, seconds: float = 1.25) -> None:
    sf.write(path, np.zeros(int(rate * seconds), dtype=np.float32), rate)


class VibeVoiceTranscriptTests(unittest.TestCase):
    def test_official_keys_preserve_speakers_overlaps_and_non_speech_filter(self) -> None:
        raw = [
            _turn("First", 0.1, 1.2, 0),
            _turn("[Music]", 0, 2, 9),
            _turn("Second", 0.8, 1.5, 2),
            _turn("  Again  ", 1.6, 2, 0),
        ]
        segments, diarization = vibe._parse_vibevoice_segments(raw, duration=2)
        self.assertEqual([s.idx for s in segments], [0, 1, 2])
        self.assertEqual([s.text_src for s in segments], ["First", "Second", "Again"])
        self.assertEqual([s.speaker_id for s in segments], ["SPEAKER_00", "SPEAKER_02", "SPEAKER_00"])
        self.assertEqual((segments[1].start, segments[1].end), (0.8, 1.5))
        self.assertEqual(diarization[1], DiarizationSegment(0.8, 1.5, "SPEAKER_02"))

    def test_native_q8_shortened_keys_match_observed_live_transcript(self) -> None:
        text = (
            "Good morning. Today we are testing speech recognition. "
            "The meeting starts at nine."
        )
        response = _response(json.dumps([
            {"Start": 0, "End": 4.81, "Speaker": 0, "Content": text},
        ]))
        segments, diarization = vibe._parse_vibevoice_segments(
            vibe._decode_transcript(response), duration=5,
        )
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text_src, text)
        self.assertEqual((segments[0].start, segments[0].end), (0, 4.81))
        self.assertEqual(segments[0].speaker_id, "SPEAKER_00")
        self.assertEqual(diarization, [DiarizationSegment(0, 4.81, "SPEAKER_00")])

    def test_bad_output_never_invents_timing_or_a_speaker(self) -> None:
        cases: list[Any] = [
            [], [{}], ["turn"], [_turn("[Silence]")], [_turn("")],
            [_turn(start=float("nan"))], [_turn(end=float("inf"))],
            [_turn(start=-1)], [_turn(start=1, end=1)], [_turn(end=3)],
            [_turn(speaker=-1)], [_turn(speaker=True)],
            [_turn() | {"Speaker ID": float("inf")}],
            [_turn() | {"Speaker ID": 1.5}], [_turn() | {"Content": None}],
            [{"start": 0, "end": 1, "text": "Old MLX", "speaker_id": 0}],
        ]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(PipelineError):
                vibe._parse_vibevoice_segments(raw, duration=1)

    def test_complete_plain_fenced_and_singleton_json_are_decoded(self) -> None:
        raw = [_turn()]
        for text in (json.dumps(raw), "```json\n" + json.dumps(raw) + "\n```", json.dumps(raw[0])):
            with self.subTest(text=text):
                self.assertEqual(vibe._decode_transcript(_response(text)), raw)

    def test_token_truncation_and_malformed_wrappers_are_rejected(self) -> None:
        responses = [
            _response(json.dumps([_turn()]), "length"),
            _response('[{"Content":"unfinished'), _response("null"),
            _response('"text"'), {}, {"choices": []},
            {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
        ]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(PipelineError):
                vibe._decode_transcript(response)

    def test_prompt_contains_actual_duration_after_audio_with_optional_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "reference with spaces.wav"
            _write_audio(wav)
            for context in ("", "  Project names: OpenAI, Q8  "):
                with self.subTest(context=context), mock.patch.object(vibe, "VIBEVOICE_CONTEXT", context):
                    payload, duration = vibe._transcription_payload(wav)
                self.assertEqual(duration, 1.25)
                content = payload["messages"][0]["content"]
                self.assertEqual(content[0]["type"], "input_audio")
                self.assertEqual(base64.b64decode(content[0]["input_audio"]["data"]), wav.read_bytes())
                self.assertIn("This is a 1.25 seconds audio", content[1]["text"])
                self.assertIn("Start time, End time, Speaker ID, Content", content[1]["text"])
                if context:
                    self.assertIn("with extra info: " + context.strip(), content[1]["text"])
                else:
                    self.assertNotIn("extra info", content[1]["text"])
                self.assertEqual(payload["temperature"], 0)
                self.assertFalse(payload["stream"])
                self.assertFalse(payload["cache_prompt"])

    def test_invalid_or_empty_audio_fails_before_server_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "empty.wav"
            _write_audio(wav, seconds=0)
            with mock.patch.object(vibe, "_get_server") as server:
                for path in (wav, wav.with_name("missing.wav")):
                    with self.subTest(path=path), self.assertRaises(PipelineError):
                        vibe._transcribe(path)
                server.assert_not_called()

    def test_main_transcription_publishes_validated_json_and_closes_server(self) -> None:
        raw = [_turn()]
        server = mock.Mock()
        server.request.return_value = _response(json.dumps(raw))
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "audio.wav"
            _write_audio(wav)
            with mock.patch.object(vibe, "_SERVER", server):
                segments, _ = vibe.vibevoice_transcribe(wav)
                self.assertIsNone(vibe._SERVER)
            self.assertEqual(segments[0].text_src, "Hello")
            self.assertEqual(json.loads((wav.parent / "vibevoice_gguf.json").read_text()), raw)
            self.assertEqual(server.request.call_args.args[0], "/v1/chat/completions")
            server.close.assert_called_once()

    def test_incomplete_main_transcription_keeps_previous_json_and_closes_server(self) -> None:
        server = mock.Mock()
        server.request.return_value = _response(json.dumps([_turn()]), "length")
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "audio.wav"
            _write_audio(wav)
            result = wav.parent / "vibevoice_gguf.json"
            result.write_text("previous validated result")
            with mock.patch.object(vibe, "_SERVER", server), self.assertRaises(PipelineError):
                vibe.vibevoice_transcribe(wav)
            self.assertEqual(result.read_text(), "previous validated result")
            server.close.assert_called_once()

    def test_reference_dispatch_uses_gguf_without_main_output_publication(self) -> None:
        server = mock.Mock()
        server.request.return_value = _response(json.dumps([_turn("Hello"), _turn("world", 0.5, 1)]))
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "reference.wav"
            _write_audio(wav)
            with mock.patch.object(asr, "ASR_ENGINE", "vibevoice"), mock.patch.object(vibe, "_SERVER", server):
                self.assertEqual(asr.transcribe_reference_audio(wav), "Hello world")
                self.assertIs(vibe._SERVER, server)
            server.close.assert_not_called()
            self.assertFalse((wav.parent / "vibevoice_gguf.json").exists())

    def test_failed_reference_transcription_closes_owned_server(self) -> None:
        server = mock.Mock()
        server.request.side_effect = PipelineError("failed generation")
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory) / "reference.wav"
            _write_audio(wav)
            with mock.patch.object(vibe, "_SERVER", server):
                self.assertEqual(vibe.transcribe_short_audio_vibevoice(wav), "")
                self.assertIsNone(vibe._SERVER)
            server.close.assert_called_once()

    def test_profile_changes_separate_old_mlx_analysis_and_reference_caches(self) -> None:
        with mock.patch.object(config, "ASR_ENGINE", "vibevoice"):
            current = cache.analysis_settings()
            self.assertEqual(current["VIBEVOICE_MODEL"], "ggml-org/VibeVoice-ASR-GGUF")
            self.assertEqual(current["VIBEVOICE_MODEL_FILE"], "VibeVoice-ASR-Q8_0.gguf")
            self.assertEqual(current["VIBEVOICE_MMPROJ_FILE"], "mmproj-VibeVoice-ASR-Q8_0.gguf")
            self.assertNotIn("VIBEVOICE_SERVER_BIN", current)
            with mock.patch.object(config, "VIBEVOICE_MODEL", "mlx-community/VibeVoice-ASR-8bit"):
                old = cache.analysis_settings()
            self.assertNotEqual(cache.fingerprint(current), cache.fingerprint(old))
            with mock.patch.object(config, "VIBEVOICE_CONTEXT", "Different hotwords"):
                self.assertNotEqual(cache.fingerprint(current), cache.fingerprint(cache.analysis_settings()))


class VibeVoiceServerTests(unittest.TestCase):
    def test_explicit_runtime_override_bypasses_isolated_project_build(self) -> None:
        with (
            mock.patch.object(vibe, "VIBEVOICE_SERVER_BIN", "/opt/homebrew/bin/llama-server"),
            mock.patch.object(vibe, "resolve_executable", return_value="/opt/homebrew/bin/llama-server") as resolve,
            mock.patch.object(Path, "is_file") as is_file,
        ):
            self.assertEqual(vibe._resolve_server(), "/opt/homebrew/bin/llama-server")
            resolve.assert_called_once_with("/opt/homebrew/bin/llama-server")
            is_file.assert_not_called()

    def test_blank_runtime_uses_isolated_project_build_before_path_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "models/llama.cpp-vibevoice/build/bin/llama-server"
            binary.parent.mkdir(parents=True)
            binary.touch()
            with (
                mock.patch.object(vibe, "VIBEVOICE_SERVER_BIN", ""),
                mock.patch.object(vibe, "_PROJECT_ROOT", root),
                mock.patch.object(vibe, "resolve_executable") as resolve,
            ):
                self.assertEqual(vibe._resolve_server(), str(binary))
                resolve.assert_not_called()

    def test_model_and_projector_download_from_one_official_immutable_revision(self) -> None:
        with mock.patch("huggingface_hub.hf_hub_download", side_effect=["model.gguf", "projector.gguf"]) as download:
            self.assertEqual(vibe._model_paths(), ("model.gguf", "projector.gguf"))
        self.assertEqual([c.kwargs["filename"] for c in download.call_args_list], [
            "VibeVoice-ASR-Q8_0.gguf", "mmproj-VibeVoice-ASR-Q8_0.gguf",
        ])
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["repo_id"], "ggml-org/VibeVoice-ASR-GGUF")
            self.assertEqual(call.kwargs["revision"], "e010fe1a89c48d696c11e57816c1da1b8406c305")
            self.assertEqual(call.kwargs["local_dir"], vibe._PROJECT_ROOT / "models/VibeVoice-ASR-GGUF")

    def _server_context(self, *, dead: bool = False, gpu_layers: int = -1) -> tuple[ExitStack, mock.Mock, mock.Mock]:
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(vibe, "_resolve_server", return_value="/bin/llama-server"))
        stack.enter_context(mock.patch.object(vibe, "_model_paths", return_value=("model.gguf", "audio.gguf")))
        stack.enter_context(mock.patch.object(vibe, "VIBEVOICE_N_CTX", 65536))
        stack.enter_context(mock.patch.object(vibe, "VIBEVOICE_MAX_TOKENS", 32768))
        stack.enter_context(mock.patch.object(vibe, "VIBEVOICE_N_GPU_LAYERS", gpu_layers))
        socket = stack.enter_context(mock.patch.object(vibe.socket, "socket"))
        socket.return_value.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 38217)
        popen = stack.enter_context(mock.patch.object(vibe.subprocess, "Popen"))
        process = popen.return_value
        process.poll.return_value = 1 if dead else None
        return stack, popen, process

    def test_launch_uses_matching_projector_loopback_auth_and_owned_cleanup(self) -> None:
        stack, popen, process = self._server_context(gpu_layers=0)
        request = stack.enter_context(mock.patch.object(vibe._VibeVoiceServer, "request", return_value={"status": "ok"}))
        server = vibe._VibeVoiceServer()
        command = popen.call_args.args[0]
        self.assertEqual(command[command.index("--mmproj") + 1], "audio.gguf")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertEqual(command[command.index("--port") + 1], "38217")
        self.assertEqual(command[command.index("--api-key") + 1], server.api_key)
        self.assertGreaterEqual(len(server.api_key), 32)
        self.assertIn("--no-context-shift", command)
        self.assertIn("--no-mmproj-offload", command)
        request.assert_called_once_with("/health", timeout=2)
        server.close()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)
        self.assertTrue(server.log.closed)

    def test_failed_startup_closes_log_without_touching_other_processes(self) -> None:
        _, popen, process = self._server_context(dead=True)
        with self.assertRaisesRegex(PipelineError, "setup_vibevoice.sh"):
            vibe._VibeVoiceServer()
        self.assertTrue(popen.call_args.kwargs["stdout"].closed)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_process_creation_failure_closes_log(self) -> None:
        _, popen, _ = self._server_context()
        popen.side_effect = OSError("cannot execute")
        with self.assertRaisesRegex(OSError, "cannot execute"):
            vibe._VibeVoiceServer()
        self.assertTrue(popen.call_args.kwargs["stdout"].closed)

    def test_readiness_failure_stops_owned_process(self) -> None:
        stack, popen, process = self._server_context()
        stack.enter_context(mock.patch.object(vibe.time, "monotonic", side_effect=[0, 0, 181]))
        stack.enter_context(mock.patch.object(vibe.time, "sleep"))
        stack.enter_context(mock.patch.object(vibe._VibeVoiceServer, "request", side_effect=URLError("not ready")))
        with self.assertRaisesRegex(PipelineError, "タイムアウト"):
            vibe._VibeVoiceServer()
        process.terminate.assert_called_once()
        self.assertTrue(popen.call_args.kwargs["stdout"].closed)

    def test_owned_process_that_ignores_termination_is_killed(self) -> None:
        stack, _, process = self._server_context()
        stack.enter_context(mock.patch.object(vibe._VibeVoiceServer, "request", return_value={}))
        server = vibe._VibeVoiceServer()
        process.wait.side_effect = [subprocess.TimeoutExpired("llama-server", 10), 0]
        server.close()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        self.assertTrue(server.log.closed)

    def test_http_request_sends_authentication_and_json_to_private_server(self) -> None:
        server = vibe._VibeVoiceServer.__new__(vibe._VibeVoiceServer)
        server.base_url = "http://127.0.0.1:38217"
        server.api_key = "private-token"
        server.opener = mock.MagicMock()
        server.opener.open.return_value.__enter__.return_value = io.BytesIO(b'{"choices": []}')
        self.assertEqual(server.request("/v1/chat/completions", {"text": "日本語"}, timeout=17), {"choices": []})
        request = server.opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:38217/v1/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer private-token")
        self.assertEqual(json.loads(request.data), {"text": "日本語"})
        self.assertEqual(server.opener.open.call_args.kwargs["timeout"], 17)


if __name__ == "__main__":
    unittest.main()
