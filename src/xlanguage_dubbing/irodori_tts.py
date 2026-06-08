#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Irodori-TTS-Server による日本語ボイスクローン TTS。

サーバーは OpenAI 互換の /v1/audio/speech API を提供する。
このクライアントは caption/style prompt と seconds を送らず、
セグメント単位リファレンス音声を irodori.ref_wav として渡す。
"""

from __future__ import annotations

import atexit
import json
import os
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from xlanguage_dubbing.audio.ffmpeg import ffprobe_duration_sec
from xlanguage_dubbing.config import (
    IRODORI_CODEC_DEVICE,
    IRODORI_MODEL_DEVICE,
    IRODORI_TTS_API_KEY,
    IRODORI_TTS_AUTO_START,
    IRODORI_TTS_BASE_URL,
    IRODORI_TTS_DIR,
    IRODORI_TTS_MODEL,
    IRODORI_TTS_REQUEST_TIMEOUT_SEC,
    IRODORI_TTS_RESPONSE_FORMAT,
    IRODORI_TTS_SERVER_HOST,
    IRODORI_TTS_SERVER_PORT,
    IRODORI_TTS_SPEED,
    IRODORI_TTS_START_COMMAND,
    IRODORI_TTS_START_TIMEOUT_SEC,
    MIN_SEGMENT_SEC,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
)
from xlanguage_dubbing.core.models import Segment, TtsMeta
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache
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
    if not path.startswith("/"):
        path = "/" + path
    return f"{IRODORI_TTS_BASE_URL}{path}"


def _auth_headers() -> dict[str, str]:
    if not IRODORI_TTS_API_KEY:
        return {}
    return {"Authorization": f"Bearer {IRODORI_TTS_API_KEY}"}


def _request_json(path: str, *, timeout: float = 3.0) -> dict:
    headers = {"Accept": "application/json"}
    headers.update(_auth_headers())
    req = urllib.request.Request(
        _api_url(path),
        headers=headers,
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return json.loads(res.read().decode("utf-8"))


def _log_path() -> Path:
    return Path("/private/tmp/xlanguage_dubbing_irodori_tts.log")


def _tail_server_log(max_chars: int = 4000) -> str:
    path = _log_path()
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _select_start_command() -> list[str]:
    if IRODORI_TTS_START_COMMAND:
        return shlex.split(IRODORI_TTS_START_COMMAND)

    return [
        "uv",
        "run",
        "python",
        "-m",
        "irodori_openai_tts",
        "--host",
        IRODORI_TTS_SERVER_HOST,
        "--port",
        str(IRODORI_TTS_SERVER_PORT),
    ]


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env["IRODORI_MODEL_DEVICE"] = IRODORI_MODEL_DEVICE
    env["IRODORI_CODEC_DEVICE"] = IRODORI_CODEC_DEVICE
    env.setdefault("IRODORI_HF_CHECKPOINT", "Aratako/Irodori-TTS-500M-v3")
    if IRODORI_TTS_API_KEY:
        env.setdefault("IRODORI_API_KEY", IRODORI_TTS_API_KEY)
    return env


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


def ensure_irodori_tts_server() -> None:
    """Irodori-TTS-Server を利用可能な状態にする。"""
    global _SERVER_PROCESS, _SERVER_LOG_FILE, _SERVER_READY

    if _SERVER_READY:
        return

    try:
        _request_json("/health", timeout=2.0)
        _SERVER_READY = True
        return
    except Exception:
        pass

    if not IRODORI_TTS_AUTO_START:
        raise PipelineError(
            "Irodori-TTS-Server に接続できません。\n"
            f"  URL: {IRODORI_TTS_BASE_URL}\n"
            "  サーバーを起動するか IRODORI_TTS_AUTO_START=true にしてください。"
        )

    if not IRODORI_TTS_DIR.exists():
        raise PipelineError(
            "Irodori-TTS-Server のディレクトリが見つかりません。\n"
            f"  IRODORI_TTS_DIR={IRODORI_TTS_DIR}\n"
            "  例: git clone https://github.com/Aratako/Irodori-TTS-Server.git "
            "Irodori-TTS-Server && cd Irodori-TTS-Server && uv sync --extra cpu"
        )

    cmd = _select_start_command()
    print_step(
        "  Irodori-TTS-Server 起動中: "
        f"{' '.join(cmd)} ({IRODORI_TTS_DIR})"
    )

    log_path = _log_path()
    _SERVER_LOG_FILE = log_path.open("a", encoding="utf-8")
    _SERVER_LOG_FILE.write(
        f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
    )
    _SERVER_LOG_FILE.flush()

    _SERVER_PROCESS = subprocess.Popen(
        cmd,
        cwd=IRODORI_TTS_DIR,
        stdout=_SERVER_LOG_FILE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_server_env(),
    )

    deadline = time.time() + IRODORI_TTS_START_TIMEOUT_SEC
    while time.time() < deadline:
        try:
            _request_json("/health", timeout=2.0)
            print_step("  Irodori-TTS-Server 起動完了")
            _SERVER_READY = True
            return
        except Exception:
            pass
        if _SERVER_PROCESS.poll() is not None:
            raise PipelineError(
                "Irodori-TTS-Server が起動前に終了しました。\n"
                f"  log: {log_path}\n{_tail_server_log()}"
            )
        time.sleep(1.0)

    raise PipelineError(
        "Irodori-TTS-Server の起動がタイムアウトしました。\n"
        f"  URL: {IRODORI_TTS_BASE_URL}\n"
        f"  log: {log_path}\n{_tail_server_log()}"
    )


def irodori_tts_synthesize(
    text: str,
    out_audio: Path,
    ref_audio_path: Path,
) -> None:
    """Irodori-TTS-Server で音声を生成する。"""
    ensure_dir(out_audio.parent)
    ensure_irodori_tts_server()

    payload = {
        "model": IRODORI_TTS_MODEL,
        "input": text,
        "response_format": IRODORI_TTS_RESPONSE_FORMAT,
        "speed": IRODORI_TTS_SPEED,
        "irodori": {
            "ref_wav": str(ref_audio_path),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": "audio/*",
        "Content-Type": "application/json",
    }
    headers.update(_auth_headers())
    req = urllib.request.Request(
        _api_url("/v1/audio/speech"),
        data=body,
        headers=headers,
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            req, timeout=IRODORI_TTS_REQUEST_TIMEOUT_SEC
        ) as res:
            audio = res.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PipelineError(
            f"Irodori-TTS 生成エラー: HTTP {exc.code}\n{detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise PipelineError(f"Irodori-TTS 接続エラー: {exc}") from exc

    if len(audio) <= 100:
        raise PipelineError("Irodori-TTS: 生成音声が空です。")

    out_audio.write_bytes(audio)


def _convert_to_flac(in_audio: Path, out_flac: Path) -> None:
    """Irodori の出力音声をプロジェクト標準の FLAC に変換する。"""
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


def generate_segment_tts_irodori(
    seg: Segment,
    out_audio_stub: Path,
    ref_cache: SpeakerReferenceCache,
    segno: int = 0,
) -> Optional[TtsMeta]:
    """Irodori-TTS でセグメントのボイスクローン音声を生成する。"""
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

    seg_ref_path = ref_cache.get_omnivoice_segment_reference_path(segno)
    if seg_ref_path is not None:
        reference_speech = seg_ref_path
    else:
        reference_speech = ref_cache.get_omnivoice_reference_path(seg.speaker_id)

    if reference_speech is None:
        print_step(
            f"    警告: 話者 {seg.speaker_id} の Irodori リファレンスがありません。"
        )
        return None

    suffix = "." + IRODORI_TTS_RESPONSE_FORMAT.lstrip(".")
    tmp_audio = out_audio_stub.with_suffix(f".irodori{suffix}")

    try:
        irodori_tts_synthesize(
            text=text,
            out_audio=tmp_audio,
            ref_audio_path=reference_speech,
        )
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
