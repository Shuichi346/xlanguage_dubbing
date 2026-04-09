#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLIエントリーポイント。
VIDEO_FOLDER 内に動画があればフォルダー一括処理、
フォルダーが存在しないか動画が0本の場合はターミナルから動画パスを1本入力して処理する。
"""

from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

from ja_dubbing.config import TEMP_ROOT, VIDEO_EXTS, VIDEO_FOLDER
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


# =========================================================
# macOS パス文字列の正規化
# =========================================================


def _normalize_user_path(raw: str) -> Path:
    """ユーザー入力のパス文字列を正規化する。

    macOS のターミナルや Finder からコピーしたパスに含まれる
    シングルクォート・ダブルクォートやバックスラッシュエスケープを除去する。
    """
    s = raw.strip()
    s = s.strip("'\"")
    s = s.replace("\\ ", " ")
    return Path(s).expanduser().resolve()


# =========================================================
# ターミナル入力による動画パス取得
# =========================================================


def _prompt_video_path() -> Path:
    """ターミナルから動画のパスを入力させて検証する。"""
    print_step("VIDEO_FOLDER に処理対象の動画がありません。")
    print_step("処理したい動画ファイルのパスを直接入力してください。")

    while True:
        try:
            raw_input = input("\n動画のパスを入力: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not raw_input:
            print("パスが空です。もう一度入力してください。")
            continue

        video_path = _normalize_user_path(raw_input)

        if not video_path.exists():
            print(f"ファイルが見つかりません: {video_path}")
            continue

        if not video_path.is_file():
            print(f"ファイルではありません: {video_path}")
            continue

        if video_path.suffix.lower() not in VIDEO_EXTS:
            supported = ", ".join(sorted(VIDEO_EXTS))
            print(f"対応していない拡張子です: {video_path.suffix}")
            print(f"対応形式: {supported}")
            continue

        return video_path


# =========================================================
# 事前チェック
# =========================================================


def preflight_checks() -> None:
    """事前チェック（VIDEO_FOLDER の存在は問わない）。"""
    which_or_raise("ffmpeg")
    which_or_raise("ffprobe")
    ensure_dir(TEMP_ROOT)

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


# =========================================================
# 動画リスト取得
# =========================================================


def list_videos(folder: Path) -> list[Path]:
    """対象動画ファイルのリストを取得する。フォルダー未存在時は空リストを返す。"""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


# =========================================================
# 動画処理
# =========================================================


def _process_single_video(video_path: Path) -> int:
    """動画1本を処理して終了する。再開機能は動画名ベースで動作する。"""
    print_step(f"単体モード: {video_path.name}")
    client = CatTranslateClient()
    work_dir = TEMP_ROOT / video_path.stem
    ensure_dir(work_dir)
    ref_cache = SpeakerReferenceCache(work_dir / "speaker_refs")
    try:
        process_one_video(video_path, client, ref_cache)
    except Exception as exc:
        print_step(f"エラー: {video_path}\n  {exc}")
        return 2
    finally:
        del ref_cache
        force_memory_cleanup()
        print_step("メモリクリーンアップ完了")

    print_step("\n処理完了しました。")
    return 0


def _process_folder_videos(videos: list[Path]) -> int:
    """VIDEO_FOLDER 内の動画を順次処理する。"""
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


# =========================================================
# メインエントリーポイント
# =========================================================


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

        # VIDEO_FOLDER 内の動画を検索する
        videos = list_videos(VIDEO_FOLDER)

        if videos:
            # フォルダーモード: 動画が見つかった場合は一括処理
            return _process_folder_videos(videos)
        else:
            # 単体モード: フォルダー未存在 or 動画0本 → ターミナル入力
            video_path = _prompt_video_path()
            return _process_single_video(video_path)

    except Exception as exc:
        print_step(f"致命的エラー: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
