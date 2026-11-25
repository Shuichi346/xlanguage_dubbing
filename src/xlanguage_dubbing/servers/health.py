#!/usr/bin/env python3
"""
サーバーヘルスチェック・起動スクリプト生成。
"""

from __future__ import annotations

import shlex
import tempfile
from ipaddress import ip_address
from pathlib import Path

from xlanguage_dubbing.config import (
    IRODORI_CODEC_DEVICE,
    IRODORI_CODEC_REPO,
    IRODORI_HF_CHECKPOINT,
    IRODORI_MODEL_DEVICE,
    IRODORI_TTS_API_KEY,
    IRODORI_TTS_DIR,
    IRODORI_TTS_NUM_STEPS,
    IRODORI_TTS_SERVER_HOST,
    IRODORI_TTS_SERVER_PORT,
    TTS_ENGINE,
)
from xlanguage_dubbing.utils import PipelineError, print_step


def validate_irodori_bind(host: str, api_key: str) -> None:
    """Require authentication whenever a server may accept non-local clients."""
    normalized = host.strip().strip("[]").lower()
    try:
        loopback = ip_address(normalized).is_loopback
    except ValueError:
        loopback = normalized == "localhost"
    if not loopback and not api_key.strip():
        raise PipelineError(
            "Irodori を localhost 以外で起動するには IRODORI_TTS_API_KEY が必要です。"
            " ローカル利用では IRODORI_TTS_SERVER_HOST=127.0.0.1 を指定してください。"
        )


def _is_irodori_tts() -> bool:
    return TTS_ENGINE in {"irodori", "irodori-tts", "irodori_tts"}


def preflight_server_checks() -> None:
    """TTS サーバーのヘルスチェックを実行する。"""
    if _is_irodori_tts():
        print_step("  TTS エンジン: Irodori-TTS（ローカルAPIサーバー使用）")
    else:
        print_step("  TTS エンジン: プロセス内推論（サーバー不要）")


def generate_start_script(output_path: Path) -> None:
    """サーバー起動用シェルスクリプトを生成する。"""
    if _is_irodori_tts():
        validate_irodori_bind(IRODORI_TTS_SERVER_HOST, IRODORI_TTS_API_KEY)
        start_command = (
            "exec uv run --no-sync python -m irodori_openai_tts "
            f"--host {shlex.quote(IRODORI_TTS_SERVER_HOST)} "
            f"--port {IRODORI_TTS_SERVER_PORT}"
        )
        script = f"""#!/bin/bash
set -euo pipefail

# === xlanguage-dubbing Irodori-TTS-Server 起動スクリプト ===
# Setup:
#   bash scripts/setup_irodori.sh {shlex.quote(str(IRODORI_TTS_DIR))}
# The helper pins the nonWatermark server fork and its locked inference library.

export IRODORI_MODEL_DEVICE={shlex.quote(IRODORI_MODEL_DEVICE)}
export IRODORI_CODEC_DEVICE={shlex.quote(IRODORI_CODEC_DEVICE)}
export IRODORI_CODEC_REPO={shlex.quote(IRODORI_CODEC_REPO)}
export IRODORI_CODEC_PRECISION=fp32
export IRODORI_CODEC_DETERMINISTIC_ENCODE=true
export IRODORI_HF_CHECKPOINT={shlex.quote(IRODORI_HF_CHECKPOINT)}
export IRODORI_CHECKPOINT=''
export IRODORI_DEFAULT_NUM_STEPS={IRODORI_TTS_NUM_STEPS}
export IRODORI_API_KEY={shlex.quote(IRODORI_TTS_API_KEY)}

cd {shlex.quote(str(IRODORI_TTS_DIR))}
{start_command}
"""
        # The script contains the configured credential; never expose an old
        # world-readable launcher while publishing its replacement.
        with tempfile.TemporaryDirectory(prefix=".launcher-", dir=output_path.parent) as temporary:
            staged = Path(temporary) / output_path.name
            staged.write_text(script, encoding="utf-8")
            staged.chmod(0o700)
            staged.replace(output_path)
        print_step(f"  起動スクリプト生成: {output_path}")
        return

    script = """#!/bin/bash
# === xlanguage-dubbing サーバー起動スクリプト ===
# 翻訳: PLaMo 3 Translate 8B Q4_K_M（専用 llama-server を自動起動・終了）
# TTS:  OmniVoice / VoxCPM2（プロセス内推論・サーバー不要）

echo "=== xlanguage-dubbing ==="
echo ""
echo "翻訳の llama-server は自動起動します。手動での起動は不要です。"
echo "直接 'uv run xlanguage-dubbing' を実行してください。"
"""

    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    print_step(f"  起動スクリプト生成: {output_path}")
