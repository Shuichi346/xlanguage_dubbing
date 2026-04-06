#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サーバーヘルスチェック・起動スクリプト生成。
翻訳は CAT-Translate (llama-cpp-python) でプロセス内推論するためサーバー不要。
OmniVoice / Kokoro はプロセス内推論のため外部サーバー不要。
"""

from __future__ import annotations

from pathlib import Path

from ja_dubbing.config import TTS_ENGINE
from ja_dubbing.utils import print_step


def _get_tts_engine() -> str:
    """現在選択されている TTS エンジン名を返す。"""
    return TTS_ENGINE.strip().lower()


def preflight_server_checks() -> None:
    """TTS サーバーのヘルスチェックを実行する。"""
    tts_engine = _get_tts_engine()

    if tts_engine == "kokoro":
        print_step("  TTS エンジン: Kokoro（サーバー不要）")
        return

    if tts_engine == "omnivoice":
        print_step("  TTS エンジン: OmniVoice（サーバー不要）")
        return


def generate_start_script(output_path: Path) -> None:
    """サーバー起動用シェルスクリプトを生成する。"""
    tts_engine = _get_tts_engine()

    if tts_engine == "kokoro":
        script = """#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（Kokoro TTSモード） ===
# 翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）
# TTS:  Kokoro TTS（プロセス内推論・サーバー不要）

echo "=== ja-dubbing（Kokoro TTSモード） ==="
echo ""
echo "外部サーバーの起動は不要です。"
echo "直接 'uv run ja-dubbing' を実行してください。"
"""
    else:
        script = """#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（OmniVoice モード） ===
# 翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）
# TTS:  OmniVoice（プロセス内推論・サーバー不要）

echo "=== ja-dubbing（OmniVoice モード） ==="
echo ""
echo "外部サーバーの起動は不要です。"
echo "直接 'uv run ja-dubbing' を実行してください。"
"""

    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    print_step(f"  起動スクリプト生成: {output_path}")
