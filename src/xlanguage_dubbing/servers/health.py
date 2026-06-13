#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サーバーヘルスチェック・起動スクリプト生成。
"""

from __future__ import annotations

import shlex
from pathlib import Path

from xlanguage_dubbing.config import (
    IRODORI_CODEC_DEVICE,
    IRODORI_MODEL_DEVICE,
    IRODORI_TTS_DIR,
    IRODORI_TTS_SERVER_HOST,
    IRODORI_TTS_SERVER_PORT,
    TTS_ENGINE,
)
from xlanguage_dubbing.utils import print_step


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
        start_command = (
            "exec uv run python -m irodori_openai_tts "
            f'--host "{IRODORI_TTS_SERVER_HOST}" '
            f'--port "{IRODORI_TTS_SERVER_PORT}"'
        )
        script = f"""#!/bin/bash
set -euo pipefail

# === xlanguage-dubbing Irodori-TTS-Server 起動スクリプト ===
# Setup:
#   git clone https://github.com/Aratako/Irodori-TTS-Server.git
#   cd Irodori-TTS-Server
#   uv sync --extra cpu

export IRODORI_MODEL_DEVICE={shlex.quote(IRODORI_MODEL_DEVICE)}
export IRODORI_CODEC_DEVICE={shlex.quote(IRODORI_CODEC_DEVICE)}

cd "{IRODORI_TTS_DIR}"
{start_command}
"""
        output_path.write_text(script, encoding="utf-8")
        output_path.chmod(0o755)
        print_step(f"  起動スクリプト生成: {output_path}")
        return

    script = """#!/bin/bash
# === xlanguage-dubbing サーバー起動スクリプト ===
# 翻訳: CAT-Translate-7b / TranslateGemma-12b-it（プロセス内推論・サーバー不要）
# TTS:  OmniVoice / VoxCPM2（プロセス内推論・サーバー不要）

echo "=== xlanguage-dubbing ==="
echo ""
echo "外部サーバーの起動は不要です。"
echo "直接 'uv run xlanguage-dubbing' を実行してください。"
"""

    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    print_step(f"  起動スクリプト生成: {output_path}")
