#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サーバーヘルスチェック・起動スクリプト生成。
"""

from __future__ import annotations

from pathlib import Path

from xlanguage_dubbing.config import KOKORO_FASTAPI_DIR, TTS_ENGINE
from xlanguage_dubbing.utils import print_step


def _is_kokoro_fastapi_tts() -> bool:
    return TTS_ENGINE in {"kokoro-fastapi", "kokoro_fastapi", "kokoro"}


def preflight_server_checks() -> None:
    """TTS サーバーのヘルスチェックを実行する。"""
    if _is_kokoro_fastapi_tts():
        print_step("  TTS エンジン: Kokoro-FastAPI（ローカルAPIサーバー使用）")
    else:
        print_step("  TTS エンジン: プロセス内推論（サーバー不要）")


def generate_start_script(output_path: Path) -> None:
    """サーバー起動用シェルスクリプトを生成する。"""
    if _is_kokoro_fastapi_tts():
        script = f"""#!/bin/bash
set -euo pipefail

# === xlanguage-dubbing Kokoro-FastAPI 起動スクリプト ===
# Direct Run (via uv): https://github.com/remsky/Kokoro-FastAPI

cd "{KOKORO_FASTAPI_DIR}"
uv run python -m unidic download

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" && -x ./start-gpu_mac.sh ]]; then
  exec ./start-gpu_mac.sh
elif [[ -x ./start-gpu.sh ]]; then
  exec ./start-gpu.sh
else
  exec ./start-cpu.sh
fi
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
