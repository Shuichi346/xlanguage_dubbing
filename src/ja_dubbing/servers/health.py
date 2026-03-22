#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
サーバーヘルスチェック・起動スクリプト生成。
翻訳は CAT-Translate (llama-cpp-python) でプロセス内推論するためサーバー不要。
MioTTS / GPT-SoVITS のときのみ API サーバーチェックを行う。
Kokoro / T5Gemma-TTS はプロセス内推論のため外部サーバー不要。
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from ja_dubbing.config import (
    GPTSOVITS_API_URL,
    GPTSOVITS_CONDA_ENV,
    GPTSOVITS_DIR,
    MIOTTS_API_URL,
    TTS_ENGINE,
)
from ja_dubbing.utils import PipelineError, print_step


def check_health(url: str, timeout: float = 5.0) -> bool:
    """URLにGETリクエストを送り応答があるか確認する。"""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _check_gptsovits_health() -> bool:
    """GPT-SoVITS API サーバーの疎通を確認する（/tts エンドポイントへの空リクエストでなく、接続可能性のみ確認）。"""
    parsed = urlparse(GPTSOVITS_API_URL)
    import socket
    try:
        sock = socket.create_connection(
            (parsed.hostname or "127.0.0.1", parsed.port or 9880),
            timeout=5.0,
        )
        sock.close()
        return True
    except Exception:
        return False


def _get_tts_engine() -> str:
    """現在選択されている TTS エンジン名を返す。"""
    return TTS_ENGINE.strip().lower()


def preflight_server_checks() -> None:
    """TTS サーバーのヘルスチェックを実行する。"""
    tts_engine = _get_tts_engine()

    if tts_engine == "kokoro":
        print_step("  TTS エンジン: Kokoro（サーバー不要）")
        return

    if tts_engine == "t5gemma":
        print_step("  TTS エンジン: T5Gemma-TTS（サーバー不要）")
        return

    if tts_engine == "gptsovits":
        if not _check_gptsovits_health():
            parsed = urlparse(GPTSOVITS_API_URL)
            port = parsed.port or 9880
            raise PipelineError(
                f"GPT-SoVITS API サーバーに接続できません: {GPTSOVITS_API_URL}\n"
                f"以下の手順でサーバーを起動してください:\n"
                f"  conda activate {GPTSOVITS_CONDA_ENV}\n"
                f"  cd {GPTSOVITS_DIR}\n"
                f"  python api_v2.py -a 127.0.0.1 -p {port}"
                f" -c GPT_SoVITS/configs/tts_infer.yaml\n"
                f"\nまたは: uv run ja-dubbing --generate-script で起動スクリプトを生成\n"
            )
        print_step(f"  GPT-SoVITS API サーバー: 接続OK ({GPTSOVITS_API_URL})")
        return

    # MioTTS の場合
    miotts_health = f"{MIOTTS_API_URL.rstrip('/')}/health"
    if not check_health(miotts_health):
        raise PipelineError(
            f"MioTTS APIサーバーに接続できません: {MIOTTS_API_URL}\n"
            "  MioTTS-Inference の run_server.py を起動してください。"
        )


def generate_start_script(output_path: Path) -> None:
    """サーバー起動用シェルスクリプトを生成する。"""
    from ja_dubbing.config import (
        MIOTTS_CODEC_MODEL,
        MIOTTS_DEVICE,
        MIOTTS_INFERENCE_DIR,
        MIOTTS_LLM_MODEL,
        MIOTTS_LLM_PORT,
        MIOTTS_MAX_TEXT_LENGTH,
    )

    parsed = urlparse(MIOTTS_API_URL)
    miotts_api_port = parsed.port or 8001

    tts_engine = _get_tts_engine()

    if tts_engine == "kokoro":
        script = """#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（Kokoro TTSモード） ===
# 翻訳は CAT-Translate-7b（プロセス内推論・サーバー不要）
# Kokoro TTS もプロセス内で動作するため、外部サーバーは不要です。

echo "=== ja-dubbing（Kokoro TTSモード） ==="
echo ""
echo "翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）"
echo "TTS:  Kokoro TTS（プロセス内推論・サーバー不要）"
echo ""
echo "外部サーバーの起動は不要です。"
echo "直接 'uv run ja-dubbing' を実行してください。"
"""
    elif tts_engine == "t5gemma":
        script = """#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（T5Gemma-TTS モード） ===
# 翻訳は CAT-Translate-7b（プロセス内推論・サーバー不要）
# T5Gemma-TTS もプロセス内で動作するため、外部サーバーは不要です。

echo "=== ja-dubbing（T5Gemma-TTS モード） ==="
echo ""
echo "翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）"
echo "TTS:  T5Gemma-TTS（プロセス内推論・サーバー不要）"
echo ""
echo "外部サーバーの起動は不要です。"
echo "直接 'uv run ja-dubbing' を実行してください。"
"""
    elif tts_engine == "gptsovits":
        gs_parsed = urlparse(GPTSOVITS_API_URL)
        gs_port = gs_parsed.port or 9880
        gs_host = gs_parsed.hostname or "127.0.0.1"

        # GPT-SoVITS の絶対パスを取得（スクリプト生成時に解決）
        gs_dir_resolved = str(Path(GPTSOVITS_DIR).resolve())

        script = f"""#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（GPT-SoVITS モード） ===
# 翻訳は CAT-Translate-7b（プロセス内推論・サーバー不要）
# GPT-SoVITS API サーバーのみ起動します。
# GPT-SoVITS は独立した conda 環境で動作します。

echo "=== ja-dubbing サーバー起動（GPT-SoVITS V2ProPlus モード） ==="

# conda 初期化
eval "$(conda shell.bash hook)"

cleanup() {{
    echo ""
    echo "=== GPT-SoVITS サーバーを停止します ==="
    if [ -n "$GPTSOVITS_PID" ] && kill -0 "$GPTSOVITS_PID" 2>/dev/null; then
        kill "$GPTSOVITS_PID" 2>/dev/null
        echo "  停止: PID=$GPTSOVITS_PID"
    fi
    wait 2>/dev/null
    echo "=== 停止完了 ==="
    exit 0
}}

trap cleanup INT TERM

echo "1/1: GPT-SoVITS API サーバー起動（ポート {gs_port}）"
echo "  conda 環境: {GPTSOVITS_CONDA_ENV}"
echo "  ディレクトリ: {gs_dir_resolved}"

conda activate {GPTSOVITS_CONDA_ENV}

cd "{gs_dir_resolved}" || {{
    echo "エラー: {gs_dir_resolved} が見つかりません"
    echo "scripts/setup_gptsovits.sh を先に実行してください"
    exit 1
}}

python api_v2.py \\
    -a {gs_host} \\
    -p {gs_port} \\
    -c GPT_SoVITS/configs/tts_infer.yaml &
GPTSOVITS_PID=$!

echo "  GPT-SoVITS API 起動待機中（15秒）..."
sleep 15

if ! kill -0 "$GPTSOVITS_PID" 2>/dev/null; then
    echo "エラー: GPT-SoVITS API サーバーが起動に失敗しました"
    exit 1
fi

echo ""
echo "=== サーバー起動完了 ==="
echo "  翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）"
echo "  GPT-SoVITS API: PID=$GPTSOVITS_PID (ポート {gs_port})"
echo "  モデル: V2ProPlus（ゼロショットボイスクローン）"
echo ""
echo "停止するには: Ctrl+C"
wait
"""
    else:
        # MioTTS モード
        script = f"""#!/bin/bash
# === ja-dubbing サーバー起動スクリプト（MioTTS モード） ===
# 翻訳は CAT-Translate-7b（プロセス内推論・サーバー不要）
# MioTTS の Ollama + API サーバーのみ起動します。

echo "=== ja-dubbing サーバー起動（MioTTS モード） ==="

PIDS=()

cleanup() {{
    echo ""
    echo "=== 全サーバーを停止します ==="
    for pid in "${{PIDS[@]}}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  停止: PID=$pid"
        fi
    done
    wait 2>/dev/null
    echo "=== 停止完了 ==="
    exit 1
}}

trap cleanup INT TERM

check_process() {{
    local pid=$1
    local name=$2
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "エラー: $name が起動に失敗しました (PID=$pid)"
        cleanup
    fi
}}

echo "1/2: MioTTS LLMバックエンド起動 (ポート{MIOTTS_LLM_PORT})"
OLLAMA_HOST=localhost:{MIOTTS_LLM_PORT} ollama serve &
MIOTTS_LLM_PID=$!
PIDS+=("$MIOTTS_LLM_PID")

echo "  Ollama 起動待機中（5秒）..."
sleep 5
check_process "$MIOTTS_LLM_PID" "Ollama (MioTTS LLM)"

echo "  MioTTS LLMモデルをプル中..."
if ! OLLAMA_HOST=localhost:{MIOTTS_LLM_PORT} ollama pull {MIOTTS_LLM_MODEL}; then
    echo "エラー: MioTTS LLMモデルのプルに失敗しました"
    cleanup
fi
echo "  MioTTS LLMモデル準備完了: {MIOTTS_LLM_MODEL}"

echo "2/2: MioTTS APIサーバー起動 (ポート{miotts_api_port})"
cd {MIOTTS_INFERENCE_DIR} || {{ echo "エラー: {MIOTTS_INFERENCE_DIR} ディレクトリが見つかりません"; cleanup; }}
uv run python run_server.py \\
    --llm-base-url http://localhost:{MIOTTS_LLM_PORT}/v1 \\
    --device {MIOTTS_DEVICE} \\
    --codec-model {MIOTTS_CODEC_MODEL} \\
    --max-text-length {MIOTTS_MAX_TEXT_LENGTH} \\
    --port {miotts_api_port} &
MIOTTS_API_PID=$!
PIDS+=("$MIOTTS_API_PID")
cd -

echo "  MioTTS API 起動待機中（10秒）..."
sleep 10
check_process "$MIOTTS_API_PID" "MioTTS API"

echo ""
echo "=== サーバー起動完了 ==="
echo "  翻訳: CAT-Translate-7b（プロセス内推論・サーバー不要）"
echo "  MioTTS LLM:  PID=$MIOTTS_LLM_PID (ポート{MIOTTS_LLM_PORT})"
echo "  MioTTS API:  PID=$MIOTTS_API_PID (ポート{miotts_api_port})"
echo ""
echo "停止するには: Ctrl+C"
echo "または: pkill -f 'ollama serve' && pkill -f 'run_server.py'"
wait
"""
    output_path.write_text(script, encoding="utf-8")
    output_path.chmod(0o755)
    print_step(f"  起動スクリプト生成: {output_path}")
