#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kokoro-FastAPI による速度優先 TTS。

Kokoro-FastAPI は OpenAI 互換の /v1/audio/speech API を提供する。
このモードはボイスクローンを行わず、日本語音声を固定 voice で生成する。
"""

from __future__ import annotations

import atexit
import json
import os
import platform
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from xlanguage_dubbing.audio.ffmpeg import ffprobe_duration_sec
from xlanguage_dubbing.config import (
    KOKORO_FASTAPI_AUTO_START,
    KOKORO_FASTAPI_BASE_URL,
    KOKORO_FASTAPI_DIR,
    KOKORO_FASTAPI_DOWNLOAD_UNIDIC,
    KOKORO_FASTAPI_MODEL,
    KOKORO_FASTAPI_REQUEST_TIMEOUT_SEC,
    KOKORO_FASTAPI_RESPONSE_FORMAT,
    KOKORO_FASTAPI_SPEED,
    KOKORO_FASTAPI_START_COMMAND,
    KOKORO_FASTAPI_START_TIMEOUT_SEC,
    KOKORO_FASTAPI_VOICE,
    MIN_SEGMENT_SEC,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
)
from xlanguage_dubbing.core.models import Segment, TtsMeta
from xlanguage_dubbing.utils import (
    PipelineError,
    ensure_dir,
    print_step,
    run_cmd,
    sanitize_text_for_tts,
    which_or_raise,
)

_SERVER_PROCESS: Optional[subprocess.Popen] = None
_SERVER_LOG_FILE = None
_SERVER_READY = False


def _api_url(path: str) -> str:
    """Kokoro-FastAPI の API URL を組み立てる。"""
    if not path.startswith("/"):
        path = "/" + path
    return f"{KOKORO_FASTAPI_BASE_URL}{path}"


def _request_json(path: str, *, timeout: float = 3.0) -> dict:
    req = urllib.request.Request(
        _api_url(path),
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _available_voice_ids() -> Optional[set[str]]:
    """API が応答した場合は voice ID 群、未応答なら None を返す。"""
    try:
        data = _request_json("/v1/audio/voices", timeout=2.0)
    except Exception:
        return None

    voices = data.get("voices")
    if not isinstance(voices, list):
        return set()

    return {
        str(v.get("id", ""))
        for v in voices
        if isinstance(v, dict)
    }


def _log_path() -> Path:
    return Path("/private/tmp/xlanguage_dubbing_kokoro_fastapi.log")


def _tail_server_log(max_chars: int = 4000) -> str:
    path = _log_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _select_start_command() -> list[str]:
    if KOKORO_FASTAPI_START_COMMAND:
        return shlex.split(KOKORO_FASTAPI_START_COMMAND)

    candidates: list[str] = []
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        candidates.append("./start-gpu_mac.sh")
    candidates.extend(["./start-gpu.sh", "./start-cpu.sh"])

    for candidate in candidates:
        script = KOKORO_FASTAPI_DIR / candidate.removeprefix("./")
        if script.exists():
            return [candidate]

    raise PipelineError(
        "Kokoro-FastAPI の起動スクリプトが見つかりません。\n"
        f"  KOKORO_FASTAPI_DIR={KOKORO_FASTAPI_DIR}\n"
        "  https://github.com/remsky/Kokoro-FastAPI を clone し、"
        "Direct Run (via uv) のセットアップを実行してください。"
    )


def _run_unidic_download_if_needed() -> None:
    """日本語 voice 用の UniDic 辞書を Kokoro-FastAPI 環境に用意する。"""
    if not KOKORO_FASTAPI_DOWNLOAD_UNIDIC:
        return

    marker = KOKORO_FASTAPI_DIR / ".xlanguage_dubbing_unidic_downloaded"
    if marker.exists():
        return

    print_step("  Kokoro-FastAPI: UniDic 辞書を確認/ダウンロード中")
    uv_bin = which_or_raise("uv")
    proc = subprocess.run(
        [uv_bin, "run", "python", "-m", "unidic", "download"],
        cwd=KOKORO_FASTAPI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise PipelineError(
            "Kokoro-FastAPI の UniDic ダウンロードに失敗しました。\n"
            "  手動で Kokoro-FastAPI ディレクトリ内から "
            "`uv run python -m unidic download` を実行してください。\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}"
        )
    marker.write_text("ok\n", encoding="utf-8")


def _stop_spawned_server() -> None:
    global _SERVER_PROCESS, _SERVER_LOG_FILE, _SERVER_READY

    if _SERVER_PROCESS is not None and _SERVER_PROCESS.poll() is None:
        _SERVER_PROCESS.terminate()
        try:
            _SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _SERVER_PROCESS.kill()
    _SERVER_PROCESS = None
    _SERVER_READY = False

    if _SERVER_LOG_FILE is not None:
        _SERVER_LOG_FILE.close()
        _SERVER_LOG_FILE = None


atexit.register(_stop_spawned_server)


def ensure_kokoro_fastapi_server() -> None:
    """Kokoro-FastAPI サーバーを利用可能な状態にする。"""
    global _SERVER_PROCESS, _SERVER_LOG_FILE, _SERVER_READY

    if _SERVER_READY:
        return

    voice_ids = _available_voice_ids()
    if voice_ids is not None:
        if voice_ids and KOKORO_FASTAPI_VOICE not in voice_ids:
            raise PipelineError(
                "Kokoro-FastAPI サーバーに指定 voice がありません。\n"
                f"  voice: {KOKORO_FASTAPI_VOICE}\n"
                f"  available: {sorted(voice_ids)}"
            )
        _SERVER_READY = True
        return

    if not KOKORO_FASTAPI_AUTO_START:
        raise PipelineError(
            "Kokoro-FastAPI サーバーに接続できません。\n"
            f"  URL: {KOKORO_FASTAPI_BASE_URL}\n"
            "  サーバーを起動するか KOKORO_FASTAPI_AUTO_START=true にしてください。"
        )

    if not KOKORO_FASTAPI_DIR.exists():
        raise PipelineError(
            "Kokoro-FastAPI のディレクトリが見つかりません。\n"
            f"  KOKORO_FASTAPI_DIR={KOKORO_FASTAPI_DIR}\n"
            "  例: git clone https://github.com/remsky/Kokoro-FastAPI.git "
            "../Kokoro-FastAPI"
        )

    _run_unidic_download_if_needed()

    cmd = _select_start_command()
    print_step(
        "  Kokoro-FastAPI サーバー起動中: "
        f"{' '.join(cmd)} ({KOKORO_FASTAPI_DIR})"
    )

    log_path = _log_path()
    _SERVER_LOG_FILE = log_path.open("a", encoding="utf-8")
    _SERVER_LOG_FILE.write(
        f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    )
    _SERVER_LOG_FILE.flush()

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _SERVER_PROCESS = subprocess.Popen(
        cmd,
        cwd=KOKORO_FASTAPI_DIR,
        stdout=_SERVER_LOG_FILE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )

    deadline = time.time() + KOKORO_FASTAPI_START_TIMEOUT_SEC
    while time.time() < deadline:
        voice_ids = _available_voice_ids()
        if voice_ids is not None:
            if voice_ids and KOKORO_FASTAPI_VOICE not in voice_ids:
                raise PipelineError(
                    "Kokoro-FastAPI サーバーに指定 voice がありません。\n"
                    f"  voice: {KOKORO_FASTAPI_VOICE}\n"
                    f"  available: {sorted(voice_ids)}"
                )
            print_step("  Kokoro-FastAPI サーバー起動完了")
            _SERVER_READY = True
            return
        if _SERVER_PROCESS.poll() is not None:
            raise PipelineError(
                "Kokoro-FastAPI サーバーが起動前に終了しました。\n"
                f"  log: {log_path}\n{_tail_server_log()}"
            )
        time.sleep(1.0)

    raise PipelineError(
        "Kokoro-FastAPI サーバーの起動がタイムアウトしました。\n"
        f"  URL: {KOKORO_FASTAPI_BASE_URL}\n"
        f"  log: {log_path}\n{_tail_server_log()}"
    )


def kokoro_fastapi_synthesize(text: str, out_audio: Path) -> None:
    """Kokoro-FastAPI で音声を生成する。"""
    ensure_dir(out_audio.parent)
    ensure_kokoro_fastapi_server()

    payload = {
        "model": KOKORO_FASTAPI_MODEL,
        "input": text,
        "voice": KOKORO_FASTAPI_VOICE,
        "response_format": KOKORO_FASTAPI_RESPONSE_FORMAT,
        "speed": KOKORO_FASTAPI_SPEED,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _api_url("/v1/audio/speech"),
        data=body,
        headers={
            "Accept": "audio/*",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=KOKORO_FASTAPI_REQUEST_TIMEOUT_SEC
        ) as res:
            audio = res.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineError(
            f"Kokoro-FastAPI 生成エラー: HTTP {exc.code}\n{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PipelineError(f"Kokoro-FastAPI 接続エラー: {exc}") from exc

    if len(audio) <= 100:
        raise PipelineError("Kokoro-FastAPI: 生成音声が空です。")

    out_audio.write_bytes(audio)


def _convert_to_flac(in_audio: Path, out_flac: Path) -> None:
    """Kokoro の出力音声をプロジェクト標準の FLAC に変換する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_flac.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(in_audio),
        "-ac", str(TTS_CHANNELS),
        "-ar", str(TTS_SAMPLE_RATE),
        "-c:a", "flac",
        str(out_flac),
    ]
    run_cmd(cmd)


def generate_segment_tts_kokoro_fastapi(
    seg: Segment,
    out_audio_stub: Path,
    segno: int = 0,
) -> Optional[TtsMeta]:
    """Kokoro-FastAPI でセグメント音声を生成する。"""
    if seg.duration < MIN_SEGMENT_SEC:
        return None

    text = sanitize_text_for_tts(seg.text_tgt)
    if not text:
        return None

    out_flac = out_audio_stub.with_suffix(".flac")
    if out_flac.exists():
        duration = ffprobe_duration_sec(out_flac)
        if duration > 0:
            return TtsMeta(
                segno=segno,
                flac_path=str(out_flac),
                duration_sec=float(duration),
            )

    suffix = "." + KOKORO_FASTAPI_RESPONSE_FORMAT.lstrip(".")
    tmp_audio = out_audio_stub.with_suffix(f".kokoro{suffix}")

    try:
        kokoro_fastapi_synthesize(text=text, out_audio=tmp_audio)
        _convert_to_flac(tmp_audio, out_flac)
        duration = ffprobe_duration_sec(out_flac)
    finally:
        try:
            if tmp_audio.exists():
                tmp_audio.unlink()
        except Exception:
            pass

    if duration <= 0:
        return None

    return TtsMeta(
        segno=segno,
        flac_path=str(out_flac),
        duration_sec=float(duration),
    )
