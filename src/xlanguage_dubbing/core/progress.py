#!/usr/bin/env python3
"""
進行状況（チェックポイント）管理。
"""

from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from xlanguage_dubbing.utils import atomic_write_json, load_json_if_exists, print_step


@lru_cache(maxsize=128)
def _content_hash(path: str, size: int, mtime_ns: int, ctime_ns: int) -> str:
    """Stream large inputs once per unchanged filesystem identity."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def video_signature(video_path: Path) -> dict[str, Any]:
    """動画ファイルのシグネチャを取得する。"""
    resolved = video_path.resolve()
    if not resolved.is_file():
        # Naming helpers also support paths before an input has been created.
        return {"path": str(resolved), "missing": True}
    st = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
        "mtime_ns": st.st_mtime_ns,
        "sha256": _content_hash(str(resolved), st.st_size, st.st_mtime_ns, st.st_ctime_ns),
    }


class ProgressStore:
    """進行状況を管理するクラス。"""

    def __init__(self, path: Path, video_path: Path) -> None:
        from xlanguage_dubbing.core.cache import analysis_settings

        self.path = path
        self.video_path = video_path
        self.data: dict[str, Any] = {
            "version": 9,
            "video": video_signature(video_path),
            "analysis": analysis_settings(),
            "steps": {
                "probe_done": False,
                "asr_done": False,
                "diarization_done": False,
                "translate_done": False,
                "tts": {"done_count": 0, "total": 0},
                "retime_done": False,
                "mux_done": False,
            },
            "artifacts": {},
            "updated_at": "",
        }

    def load(self) -> bool:
        """進行状況を読み込む。"""
        obj = load_json_if_exists(self.path)
        if not isinstance(obj, dict):
            return False
        v = obj.get("video") or {}
        same = v == self.data["video"] and obj.get("analysis") == self.data["analysis"]
        if same:
            self.data = obj
            return True
        else:
            print_step(
                "progress.json はありますが動画が変更された可能性があるため"
                "無視して最初から実行します。"
            )
        return False

    def save(self) -> None:
        """進行状況を保存する。"""
        self.data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(self.path, self.data)

    def step(self, key: str) -> Any:
        """指定したステップの状態を取得する。"""
        return (self.data.get("steps") or {}).get(key)

    def set_step(self, key: str, value: Any) -> None:
        """指定したステップの状態を設定する。"""
        self.data.setdefault("steps", {})
        self.data["steps"][key] = value

    def set_artifact(self, key: str, value: Any) -> None:
        """アーティファクト情報を設定する。"""
        self.data.setdefault("artifacts", {})
        self.data["artifacts"][key] = value
