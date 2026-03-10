#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plamo-translate-cli (MCP) による翻訳処理。
サーバーモードの plamo-translate に MCP クライアントで接続して翻訳する。
イベントループを使い回し、セグメント間のオーバーヘッドを抑制する。
"""

from __future__ import annotations

import asyncio
import gc
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

from ja_dubbing.config import (
    GLITCH_MIN_REPEAT,
    GLITCH_PHRASE,
    INPUT_REPEAT_THRESHOLD,
    INPUT_UNIQUE_RATIO_THRESHOLD,
    PLAMO_TRANSLATE_RETRIES,
    PLAMO_TRANSLATE_RETRY_BACKOFF_SEC,
    TRANSLATE_TIMEOUT_SEC,
)
from ja_dubbing.core.models import Segment
from ja_dubbing.core.progress import ProgressStore
from ja_dubbing.audio.segment_io import load_segments_json, save_segments_json_atomic
from ja_dubbing.utils import PipelineError, print_step


class PlamoTranslateError(Exception):
    """plamo-translate翻訳エラー。"""


# =========================================================
# 翻訳「出力」（日本語）の異常検出
# =========================================================


def is_translation_glitch(text_ja: str) -> bool:
    """翻訳結果が異常（繰り返し）かどうかを判定する。"""
    t = (text_ja or "").strip()
    if not t:
        return False

    compact = re.sub(r"\s+", "", t)
    escaped = re.escape(GLITCH_PHRASE)
    pat = re.compile(
        rf"(?:{escaped}.{{0,20}}){{{GLITCH_MIN_REPEAT},}}"
    )
    return bool(pat.search(compact))


# =========================================================
# 翻訳「入力」（英語）の繰り返し検出
# =========================================================


def _is_repetitive_input(text_en: str) -> bool:
    """入力テキストが繰り返しで翻訳不要かを判定する。"""
    t = (text_en or "").strip()
    if not t:
        return True

    # 引用符で囲まれたフレーズを抽出
    phrases = re.findall(r'"([^"]+)"', t)
    if not phrases:
        # 引用符がない場合は文単位で分割
        phrases = re.split(r"(?<=[\.\!\?])\s+", t)
        phrases = [p.strip() for p in phrases if p.strip()]

    if len(phrases) <= 2:
        return False

    # ユニーク率が閾値未満なら繰り返しとみなす
    unique_phrases = set(p.strip().lower() for p in phrases)
    unique_ratio = len(unique_phrases) / len(phrases)
    if unique_ratio < INPUT_UNIQUE_RATIO_THRESHOLD:
        return True

    # 同一フレーズがN回以上出現しているかチェック
    counter = Counter(p.strip().lower() for p in phrases)
    most_common_count = counter.most_common(1)[0][1]
    if most_common_count >= INPUT_REPEAT_THRESHOLD:
        return True

    return False


# =========================================================
# MCP非同期翻訳
# =========================================================


async def _translate_async(text: str) -> str:
    """MCPクライアント経由で非同期翻訳を実行する。"""
    try:
        from plamo_translate.clients.translate import MCPClient
    except ImportError as exc:
        raise PipelineError(
            "plamo-translate がインストールされていません。\n"
            "以下を実行してください:\n"
            "  uv pip install plamo-translate\n"
        ) from exc

    client = MCPClient(stream=False)
    messages = [
        {"role": "user", "content": f"input lang=English\n{text}"},
        {"role": "user", "content": "output lang=Japanese\n"},
    ]
    result_parts: list[str] = []
    try:
        async for chunk in client.translate(messages):
            result_parts.append(chunk)
    finally:
        # MCPClient の非同期リソースを確実にクローズする
        if hasattr(client, "close"):
            try:
                await client.close()
            except Exception:
                pass
        elif hasattr(client, "aclose"):
            try:
                await client.aclose()
            except Exception:
                pass
        elif hasattr(client, "__aexit__"):
            try:
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        del client

    return "".join(result_parts).strip()


async def _translate_with_timeout(text: str) -> str:
    """タイムアウト付きで翻訳を実行する。"""
    try:
        return await asyncio.wait_for(
            _translate_async(text), timeout=TRANSLATE_TIMEOUT_SEC
        )
    except asyncio.TimeoutError:
        raise PlamoTranslateError(
            f"翻訳タイムアウト（{TRANSLATE_TIMEOUT_SEC:.0f}秒）: "
            f"テキスト先頭='{text[:80]}...'"
        )


def _get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """既存のイベントループを取得するか、新規作成して返す。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# =========================================================
# 翻訳クライアント
# =========================================================


class PlamoTranslateClient:
    """plamo-translate-cli MCPクライアントラッパー。"""

    def __init__(self) -> None:
        self._loop = _get_or_create_event_loop()
        self._call_count = 0
        # GCを実行する翻訳回数の間隔
        self._gc_interval = 50

    def translate(
        self,
        text: str,
        retries: int = PLAMO_TRANSLATE_RETRIES,
        retry_backoff_sec: float = PLAMO_TRANSLATE_RETRY_BACKOFF_SEC,
    ) -> tuple[str, str]:
        """テキストを英日翻訳する。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                result = self._loop.run_until_complete(
                    _translate_with_timeout(text)
                )
                self._call_count += 1
                # 一定回数ごとにGCを実行してメモリ蓄積を防ぐ
                if self._call_count % self._gc_interval == 0:
                    gc.collect()
                return result, "stop"
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    wait_sec = retry_backoff_sec * attempt
                    print_step(
                        f"    翻訳リトライ {attempt}/{retries}: "
                        f"{wait_sec:.1f}秒待機... ({exc})"
                    )
                    time.sleep(wait_sec)

        raise PlamoTranslateError(f"翻訳失敗(リトライ枯渇): {last_err}")


# =========================================================
# セグメント翻訳
# =========================================================


def _new_segment_en_only(seg: Segment) -> Segment:
    """英語テキストのみを保持したセグメントを生成する（翻訳再開用）。"""
    return replace(seg, text_ja="")


def translate_segment_safely(client: PlamoTranslateClient, text_en: str) -> str:
    """セグメントを安全に翻訳する。"""
    t = (text_en or "").strip()
    if not t:
        return ""

    # 翻訳「入力」の繰り返し検出: サーバーハングを未然に防ぐ
    if _is_repetitive_input(t):
        print_step("    繰り返し入力を検出 → スキップ")
        return "（繰り返し音声）"

    if len(t) <= 2000:
        out, _ = client.translate(t)
        ja = out.strip()
        if is_translation_glitch(ja):
            return "翻訳エラー"
        return ja

    # 長いテキストは文単位で分割して翻訳
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    outs: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if _is_repetitive_input(p):
            outs.append("繰り返し音声エラー")
            continue
        out, _ = client.translate(p)
        ja = out.strip()
        if is_translation_glitch(ja):
            outs.append("翻訳エラー")
        else:
            outs.append(ja)

    joined = " ".join([o for o in outs if o])
    if is_translation_glitch(joined):
        return "翻訳エラー"
    return joined


def translate_segments_resumable(
    client: PlamoTranslateClient,
    segments_en: list[Segment],
    seg_json_enja: Path,
    progress: ProgressStore,
) -> list[Segment]:
    """セグメントを再開可能な形式で翻訳する。"""
    if seg_json_enja.exists():
        try:
            loaded = load_segments_json(seg_json_enja)
            if len(loaded) == len(segments_en):
                segments = loaded
            else:
                segments = [_new_segment_en_only(s) for s in segments_en]
        except Exception:
            segments = [_new_segment_en_only(s) for s in segments_en]
    else:
        segments = [_new_segment_en_only(s) for s in segments_en]

    total = len(segments)
    for segno, seg in enumerate(segments, start=1):
        if (seg.text_ja or "").strip():
            if is_translation_glitch(seg.text_ja):
                segments[segno - 1] = replace(seg, text_ja="翻訳エラー")
                save_segments_json_atomic(segments, seg_json_enja)
                progress.set_step("translate_done", False)
                progress.set_artifact("segments_en_ja_json", str(seg_json_enja))
                progress.save()
            continue

        print_step(
            f"  翻訳 {segno}/{total}: {seg.start:.3f}-{seg.end:.3f}  "
            f"'{seg.text_en[:60]}'"
        )
        ja = translate_segment_safely(client, seg.text_en)
        segments[segno - 1] = replace(seg, text_ja=ja)
        save_segments_json_atomic(segments, seg_json_enja)
        progress.set_step("translate_done", False)
        progress.set_artifact("segments_en_ja_json", str(seg_json_enja))
        progress.save()

    progress.set_step("translate_done", True)
    progress.save()
    return segments
