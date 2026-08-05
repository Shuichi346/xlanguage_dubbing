#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数。
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

# macOS 環境で ffmpeg/ffprobe 等を探索するフォールバックパス
_HOMEBREW_PATHS = [
    "/opt/homebrew/bin",   # Apple Silicon Homebrew
    "/usr/local/bin",      # Intel Mac Homebrew
    "/usr/bin",
    "/bin",
]


class PipelineError(Exception):
    """パイプライン処理の例外。"""


def print_step(msg: str) -> None:
    """タイムスタンプ付きでメッセージを出力する。"""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def resolve_executable(bin_name: str) -> str:
    """外部コマンドのフルパスを解決する。"""
    found = shutil.which(bin_name)
    if found:
        return found

    for dir_path in _HOMEBREW_PATHS:
        candidate = Path(dir_path) / bin_name
        if candidate.is_file() and candidate.stat().st_mode & 0o111:
            return str(candidate)

    raise PipelineError(f"必要なコマンドが見つかりません: {bin_name}")


def which_or_raise(bin_name: str) -> str:
    """コマンドのパスを取得する。見つからない場合は例外。"""
    return resolve_executable(bin_name)


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """外部コマンドを実行する。"""
    resolved_cmd = list(cmd)
    if resolved_cmd and not Path(resolved_cmd[0]).is_absolute():
        try:
            resolved_cmd[0] = resolve_executable(resolved_cmd[0])
        except PipelineError:
            pass

    proc = subprocess.run(
        resolved_cmd,
        capture_output=True,
        text=True,
    )
    if check and proc.returncode != 0:
        raise PipelineError(
            "コマンド失敗\n"
            f"  cmd: {' '.join(resolved_cmd)}\n"
            f"  code: {proc.returncode}\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}\n"
        )
    return proc


def ensure_dir(p: Path) -> None:
    """ディレクトリを作成する（存在していてもOK）。"""
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """テキストをアトミックに書き込む。"""
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding)
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    """JSONをアトミックに書き込む。"""
    atomic_write_text(
        path, json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_json_if_exists(path: Path) -> Optional[Any]:
    """JSONファイルを読み込む。存在しない場合はNone。"""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def normalize_spaces(text: str) -> str:
    """テキストの空白を正規化する。"""
    t = (text or "").strip()
    return re.sub(r"\s+", " ", t)


def sanitize_text_for_tts(text: str) -> str:
    """TTS用にテキストを整形する。"""
    return normalize_spaces(text)


def ffmpeg_concat_quote(path_str: str) -> str:
    """ffmpeg concat demuxer用にパスをクォートする。"""
    return path_str.replace("'", r"'\''")


def force_memory_cleanup() -> None:
    """Python GC と PyTorch MPS キャッシュを強制クリーンアップする。"""
    gc.collect()

    try:
        import torch
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except ImportError:
        pass

    try:
        import mlx.core as mx
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    gc.collect()
