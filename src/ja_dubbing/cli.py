#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIエントリーポイント。
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

from ja_dubbing.config import TEMP_ROOT, TTS_ENGINE, VIDEO_EXTS, VIDEO_FOLDER
from ja_dubbing.core.pipeline import process_one_video
from ja_dubbing.segments.spacy_split import initialize_spacy
from ja_dubbing.servers.health import generate_start_script
from ja_dubbing.translation.cat_translate import CatTranslateClient
from ja_dubbing.tts.reference import SpeakerReferenceCache
from ja_dubbing.utils import (
    PipelineError,
    ensure_dir,
    force_memory_cleanup,
    print_step,
    which_or_raise,
)


def preflight_checks() -> None:
    """事前チェック。"""
    which_or_raise("ffmpeg")
    which_or_raise("ffprobe")
    ensure_dir(TEMP_ROOT)
    if not VIDEO_FOLDER.exists():
        raise PipelineError(f"VIDEO_FOLDER が存在しません: {VIDEO_FOLDER}")

    tts_engine = TTS_ENGINE.strip().lower()

    # Kokoro TTS 使用時は unidic 辞書の存在を確認する
    if tts_engine == "kokoro":
        from ja_dubbing.tts.kokoro_tts import ensure_unidic_downloaded
        ensure_unidic_downloaded()

    if tts_engine == "omnivoice":
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print_step("  OmniVoice 推論デバイス: mps")
            else:
                print_step("  OmniVoice 推論デバイス: cpu")
        except ImportError as exc:
            raise PipelineError(
                "torch がインストールされていません。\n"
                "  uv sync を実行してください。"
            ) from exc


def list_videos(folder: Path) -> list[Path]:
    """対象動画ファイルのリストを取得する。"""
    if not folder.exists():
        raise PipelineError(f"動画フォルダーが存在しません: {folder}")
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def main() -> int:
    """メインエントリーポイント。"""
    # macOS + MLX 環境で multiprocessing を安全に使うため spawn を設定する
    try:
        multiprocessing.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    if "--generate-script" in sys.argv:
        script_path = Path("start_servers.sh")
        generate_start_script(script_path)
        print_step(f"起動スクリプトを生成しました: {script_path}")
        return 0

    initialize_spacy()

    try:
        preflight_checks()
        videos = list_videos(VIDEO_FOLDER)
        if not videos:
            print_step(f"対象動画が見つかりません: {VIDEO_FOLDER}")
            return 0

        print_step(f"処理対象動画数: {len(videos)}")
        client = CatTranslateClient()

        failed: list[Path] = []
        for idx, v in enumerate(videos, start=1):
            print_step(f"\n[{idx}/{len(videos)}] 処理開始: {v.name}")
            work_dir = TEMP_ROOT / v.stem
            ensure_dir(work_dir)
            ref_cache = SpeakerReferenceCache(work_dir / "speaker_refs")
            try:
                process_one_video(v, client, ref_cache)
            except Exception as exc:
                print_step(f"エラー: {v}\n  {exc}")
                failed.append(v)
            finally:
                del ref_cache
                force_memory_cleanup()
                print_step(f"[{idx}/{len(videos)}] メモリクリーンアップ完了")

        if failed:
            print_step(f"\n失敗: {len(failed)}/{len(videos)}")
            for f in failed:
                print_step(f"  - {f}")
            return 2

        print_step("\n全て完了しました。")
        return 0

    except Exception as exc:
        print_step(f"致命的エラー: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
