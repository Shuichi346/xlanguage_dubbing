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


class PipelineError(Exception):
    """パイプライン処理の例外。"""


def print_step(msg: str) -> None:
    """タイムスタンプ付きでメッセージを出力する。"""
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    """外部コマンドを実行する。"""
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        raise PipelineError(
            "コマンド失敗\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  code: {proc.returncode}\n"
            f"  stdout:\n{proc.stdout}\n"
            f"  stderr:\n{proc.stderr}\n"
        )
    return proc


def ensure_dir(p: Path) -> None:
    """ディレクトリを作成する（存在していてもOK）。"""
    p.mkdir(parents=True, exist_ok=True)


def which_or_raise(bin_name: str) -> str:
    """コマンドのパスを取得する。見つからない場合は例外。"""
    p = shutil.which(bin_name)
    if not p:
        raise PipelineError(f"必要なコマンドが見つかりません: {bin_name}")
    return p


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
        mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass

    gc.collect()
