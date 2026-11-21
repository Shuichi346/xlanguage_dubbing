"""Joint transcription and diarization with the official VibeVoice Q8 GGUF."""

from __future__ import annotations

import atexit
import base64
import json
import math
import re
import secrets
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

import soundfile as sf

from xlanguage_dubbing.config import (
    VIBEVOICE_CONTEXT,
    VIBEVOICE_MAX_TOKENS,
    VIBEVOICE_MMPROJ_FILE,
    VIBEVOICE_MODEL,
    VIBEVOICE_MODEL_FILE,
    VIBEVOICE_MODEL_REVISION,
    VIBEVOICE_N_CTX,
    VIBEVOICE_N_GPU_LAYERS,
    VIBEVOICE_SERVER_BIN,
)
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    print_step,
    resolve_executable,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NON_SPEECH_PATTERN = re.compile(r"^\[.*\]$")
_SERVER: _VibeVoiceServer | None = None


def _resolve_server() -> str:
    """Prefer the isolated official build without changing PLaMo's runtime."""
    if VIBEVOICE_SERVER_BIN:
        return resolve_executable(str(Path(VIBEVOICE_SERVER_BIN).expanduser()))
    local = _PROJECT_ROOT / "models/llama.cpp-vibevoice/build/bin/llama-server"
    if local.is_file():
        return str(local)
    try:
        return resolve_executable("llama-server")
    except PipelineError as exc:
        raise PipelineError(
            "VibeVoice 対応 llama-server が必要です。bash scripts/setup_vibevoice.sh を実行してください。"
        ) from exc


def _model_paths() -> tuple[str, str]:
    """Download the matching official model/projector at one immutable revision."""
    from huggingface_hub import hf_hub_download

    directory = _PROJECT_ROOT / "models" / "VibeVoice-ASR-GGUF"
    model, projector = (
        hf_hub_download(
            repo_id=VIBEVOICE_MODEL, filename=name, revision=VIBEVOICE_MODEL_REVISION,
            local_dir=directory,
        )
        for name in (VIBEVOICE_MODEL_FILE, VIBEVOICE_MMPROJ_FILE)
    )
    return model, projector


class _VibeVoiceServer:
    """Own a private localhost subprocess for main and reference transcription."""

    def __init__(self) -> None:
        if not 512 <= VIBEVOICE_N_CTX <= 131072:
            raise PipelineError("VIBEVOICE_N_CTX は 512〜131072 で指定してください。")
        if not 0 < VIBEVOICE_MAX_TOKENS < VIBEVOICE_N_CTX:
            raise PipelineError("VIBEVOICE_MAX_TOKENS は正数かつ VIBEVOICE_N_CTX 未満にしてください。")
        binary = _resolve_server()
        model, projector = _model_paths()
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.api_key = secrets.token_urlsafe(32)
        self.opener = build_opener(ProxyHandler({}))
        self.log = tempfile.TemporaryFile()
        command = [
            binary, "--model", model, "--mmproj", projector,
            "--ctx-size", str(VIBEVOICE_N_CTX),
            "--gpu-layers", str(VIBEVOICE_N_GPU_LAYERS),
            "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
            "--no-webui", "--no-context-shift", "--api-key", self.api_key,
        ]
        if VIBEVOICE_N_GPU_LAYERS == 0:
            command.append("--no-mmproj-offload")
        print_step("  VibeVoice-ASR Q8 GGUF サーバーを起動中...")
        try:
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise PipelineError(
                        "VibeVoice llama-server の起動に失敗しました。\n"
                        "対応版を準備: bash scripts/setup_vibevoice.sh\n" + self._log_tail()
                    )
                try:
                    self.request("/health", timeout=2)
                    return
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            raise PipelineError("VibeVoice llama-server の起動がタイムアウトしました。\n" + self._log_tail())
        except BaseException:
            self.close()
            raise

    def _log_tail(self) -> str:
        self.log.seek(0, 2)
        self.log.seek(max(0, self.log.tell() - 4000))
        return self.log.read().decode("utf-8", errors="replace")

    def request(self, path: str, payload: dict | None = None, *, timeout: float = 7200) -> dict:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with self.opener.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            if payload is None:
                raise
            detail = exc.read().decode("utf-8", errors="replace")[-2000:]
            raise PipelineError(f"VibeVoice HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if payload is None:
                raise
            raise PipelineError(f"VibeVoice サーバー通信に失敗しました: {exc}\n{self._log_tail()}") from exc

    def close(self) -> None:
        """Stop only the owned process, including after failed startup/inference."""
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        finally:
            self.log.close()


def _get_server() -> _VibeVoiceServer:
    global _SERVER
    if _SERVER is None:
        _SERVER = _VibeVoiceServer()
    return _SERVER


def release_vibevoice_model() -> None:
    """Release the owned native server before translation/TTS and on failures."""
    global _SERVER
    if _SERVER is not None:
        server, _SERVER = _SERVER, None
        server.close()
        print_step("  VibeVoice-ASR サーバーを終了しました")


atexit.register(release_vibevoice_model)


def _transcription_payload(wav_path: Path) -> tuple[dict, float]:
    """Match the native duration-aware prompt; the server resamples to 24 kHz."""
    try:
        info = sf.info(wav_path)
    except (RuntimeError, OSError) as exc:
        raise PipelineError(f"VibeVoice 入力音声を読めません: {wav_path}") from exc
    duration = info.frames / info.samplerate
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError("VibeVoice 入力音声が空です。")
    keys = "transcribe it with these keys: Start time, End time, Speaker ID, Content"
    if VIBEVOICE_CONTEXT.strip():
        prompt = (
            f"\nThis is a {duration:.2f} seconds audio, with extra info: "
            f"{VIBEVOICE_CONTEXT.strip()}\n\nPlease {keys}"
        )
    else:
        prompt = f"\nThis is a {duration:.2f} seconds audio, please {keys}"
    return {
        "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(wav_path.read_bytes()).decode("ascii"), "format": "wav",
            }},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0.0,
        "max_tokens": VIBEVOICE_MAX_TOKENS,
        "stream": False,
        "cache_prompt": False,
    }, duration


def _decode_transcript(response: dict) -> list[dict]:
    """Reject incomplete generations instead of publishing partial transcripts."""
    try:
        choice = response["choices"][0]
        if choice["finish_reason"] != "stop":
            raise PipelineError(
                "VibeVoice の生成が完了していません。VIBEVOICE_MAX_TOKENS / N_CTX を確認してください。"
            )
        text = choice["message"]["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        result = json.loads(text)
    except (KeyError, IndexError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise PipelineError("VibeVoice が完全な文字起こし JSON を返しませんでした。") from exc
    if isinstance(result, dict):
        result = [result]
    if not isinstance(result, list):
        raise PipelineError("VibeVoice の文字起こし JSON 形式が不正です。")
    return result


def _parse_vibevoice_segments(
    raw_segments: list[dict], *, duration: float | None = None,
) -> tuple[list[Segment], list[DiarizationSegment]]:
    """Normalize official JSON keys, preserving speaker IDs and overlapping turns."""
    segments: list[Segment] = []
    for raw in raw_segments:
        try:
            text = raw["Content"].strip()
            if not text or _NON_SPEECH_PATTERN.fullmatch(text):
                continue
            # The official processor accepts both training-prompt and generated
            # short keys; the pinned Q8 model emits Start/End/Speaker in practice.
            start = float(raw["Start time"] if "Start time" in raw else raw["Start"])
            end = float(raw["End time"] if "End time" in raw else raw["End"])
            speaker = raw["Speaker ID"] if "Speaker ID" in raw else raw["Speaker"]
            speaker_number = int(speaker)
            if isinstance(speaker, bool) or str(speaker_number) != str(speaker).strip():
                raise ValueError("invalid speaker")
            if (
                not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or speaker_number < 0
                or (duration is not None and end > duration + 0.5)
            ):
                raise ValueError("invalid timing or speaker")
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
            raise PipelineError("VibeVoice の時刻・話者・テキスト情報が不正です。") from exc
        segments.append(Segment(
            idx=len(segments), start=start, end=end, text_src=text,
            speaker_id=f"SPEAKER_{speaker_number:02d}",
        ))
    if not segments:
        raise PipelineError("VibeVoice-ASR: 文字起こし結果が空です。")
    return segments, [DiarizationSegment(s.start, s.end, s.speaker_id) for s in segments]


def _transcribe(wav_path: Path) -> tuple[list[dict], list[Segment], list[DiarizationSegment]]:
    payload, duration = _transcription_payload(wav_path)
    response = _get_server().request("/v1/chat/completions", payload)
    raw = _decode_transcript(response)
    segments, diarization = _parse_vibevoice_segments(raw, duration=duration)
    return raw, segments, diarization


def vibevoice_transcribe(wav_path: Path) -> tuple[list[Segment], list[DiarizationSegment]]:
    """Transcribe one whole input without resetting speakers across audio chunks."""
    print_step(f"  VibeVoice-ASR Q8 GGUF 実行中: {wav_path.name}")
    try:
        raw, segments, diarization = _transcribe(wav_path)
        atomic_write_json(wav_path.parent / "vibevoice_gguf.json", raw)
        print_step(f"  VibeVoice-ASR 完了: {len(segments)} セグメント")
        return segments, diarization
    finally:
        release_vibevoice_model()


def transcribe_short_audio_vibevoice(wav_path: Path) -> str:
    """Use the same native model for clipped reference audio, retaining it briefly."""
    if not wav_path.is_file():
        return ""
    try:
        _, segments, _ = _transcribe(wav_path)
        return " ".join(segment.text_src for segment in segments)
    except Exception as exc:
        release_vibevoice_model()
        print_step(f"    VibeVoice 参照音声文字起こし失敗: {exc}")
        return ""
