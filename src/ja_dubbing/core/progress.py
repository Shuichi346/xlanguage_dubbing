#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
進行状況（チェックポイント）管理。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from ja_dubbing.utils import atomic_write_json, load_json_if_exists, print_step


def video_signature(video_path: Path) -> Dict[str, Any]:
    """動画ファイルのシグネチャを取得する。"""
    st = video_path.stat()
    return {
        "path": str(video_path),
        "size": int(st.st_size),
        "mtime": float(st.st_mtime),
    }


class ProgressStore:
    """進行状況を管理するクラス。"""

    def __init__(self, path: Path, video_path: Path) -> None:
        self.path = path
        self.video_path = video_path
        self.data: Dict[str, Any] = {
            "version": 8,
            "video": video_signature(video_path),
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

    def load(self) -> None:
        """進行状況を読み込む。"""
        obj = load_json_if_exists(self.path)
        if not isinstance(obj, dict):
            return
        v = obj.get("video") or {}
        same = (
            isinstance(v, dict)
            and v.get("size") == self.data["video"]["size"]
            and float(v.get("mtime", -1)) == float(self.data["video"]["mtime"])
        )
        if same:
            self.data = obj
        else:
            print_step(
                "progress.json はありますが動画が変更された可能性があるため"
                "無視して最初から実行します。"
            )

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
