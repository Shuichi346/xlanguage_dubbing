#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
翻訳処理の統合エントリポイント。

言語ペアに応じて最適なエンジンを自動選択する:
  - 日英/英日 → CAT-Translate-7b（高精度日英特化モデル）
  - その他 → TranslateGemma-12b-it（55言語対応）

CAT-Translate-7b と TranslateGemma-12b-it は同時にメモリに載せると
24GB では不足するため、エンジン切替時にモデルを解放する。
"""

from __future__ import annotations

import gc
import re
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Optional

from xlanguage_dubbing.config import (
    CAT_TRANSLATE_FILE,
    CAT_TRANSLATE_N_CTX,
    CAT_TRANSLATE_N_GPU_LAYERS,
    CAT_TRANSLATE_REPEAT_PENALTY,
    CAT_TRANSLATE_REPO,
    CAT_TRANSLATE_RETRIES,
    CAT_TRANSLATE_RETRY_BACKOFF_SEC,
    INPUT_REPEAT_THRESHOLD,
    INPUT_UNIQUE_RATIO_THRESHOLD,
    OUTPUT_LANG,
    OUTPUT_REPEAT_THRESHOLD,
    TRANSLATEGEMMA_RETRIES,
    TRANSLATEGEMMA_RETRY_BACKOFF_SEC,
)
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.core.progress import ProgressStore
from xlanguage_dubbing.audio.segment_io import load_segments_json, save_segments_json_atomic
from xlanguage_dubbing.lang_utils import (
    detect_language_from_text,
    is_ja_en_pair,
    normalize_lang_code,
    select_translation_engine,
)
from xlanguage_dubbing.utils import PipelineError, force_memory_cleanup, print_step


class CatTranslateError(Exception):
    """翻訳エラー。"""


# CAT-Translate モデルの遅延ロード用グローバルキャッシュ
_CAT_MODEL = None
# 現在ロード中のエンジン名
_CURRENT_ENGINE: str = ""


def _get_model_path() -> str:
    """CAT-Translate GGUF モデルファイルのパスを取得する。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PipelineError(
            "huggingface-hub がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    print_step(f"  モデルを取得中: {CAT_TRANSLATE_REPO}/{CAT_TRANSLATE_FILE}")
    path = hf_hub_download(
        repo_id=CAT_TRANSLATE_REPO,
        filename=CAT_TRANSLATE_FILE,
    )
    print_step(f"  モデルパス: {path}")
    return path


def _get_cat_model():
    """CAT-Translate モデルを遅延ロードする。"""
    global _CAT_MODEL, _CURRENT_ENGINE

    # 別のエンジンがロードされていたら解放する
    if _CURRENT_ENGINE == "translategemma":
        from xlanguage_dubbing.translation.translategemma import release_gemma_model
        release_gemma_model()
        force_memory_cleanup()

    if _CAT_MODEL is not None:
        _CURRENT_ENGINE = "cat_translate"
        return _CAT_MODEL

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise PipelineError(
            "llama-cpp-python がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    model_path = _get_model_path()

    print_step("  CAT-Translate モデルをロード中...")
    _CAT_MODEL = Llama(
        model_path=model_path,
        n_gpu_layers=CAT_TRANSLATE_N_GPU_LAYERS,
        n_ctx=CAT_TRANSLATE_N_CTX,
        verbose=False,
    )
    _CURRENT_ENGINE = "cat_translate"
    print_step("  CAT-Translate モデルのロード完了")
    return _CAT_MODEL


def _release_cat_model() -> None:
    """CAT-Translate モデルを解放する。"""
    global _CAT_MODEL, _CURRENT_ENGINE
    if _CAT_MODEL is not None:
        del _CAT_MODEL
        _CAT_MODEL = None
        if _CURRENT_ENGINE == "cat_translate":
            _CURRENT_ENGINE = ""
        gc.collect()
        print_step("  CAT-Translate モデルを解放しました")


def _ensure_engine(engine_name: str) -> None:
    """必要なエンジンがロードされていることを保証する。メモリ節約のため排他的。"""
    global _CURRENT_ENGINE
    if _CURRENT_ENGINE == engine_name:
        return

    # 現在のエンジンを解放する
    if _CURRENT_ENGINE == "cat_translate":
        _release_cat_model()
        force_memory_cleanup()
    elif _CURRENT_ENGINE == "translategemma":
        from xlanguage_dubbing.translation.translategemma import release_gemma_model
        release_gemma_model()
        force_memory_cleanup()

    _CURRENT_ENGINE = engine_name


def release_all_translation_models() -> None:
    """全翻訳モデルを解放する。TTS 開始前などに呼び出す。"""
    global _CURRENT_ENGINE
    _release_cat_model()

    try:
        from xlanguage_dubbing.translation.translategemma import release_gemma_model
        release_gemma_model()
    except ImportError:
        pass

    _CURRENT_ENGINE = ""
    force_memory_cleanup()
    print_step("  全翻訳モデルを解放しました")


# =========================================================
# 翻訳異常検出
# =========================================================


def _detect_repeated_phrases(text: str) -> bool:
    """翻訳結果に同一フレーズがN回以上繰り返されているか検査する。"""
    t = (text or "").strip()
    if not t:
        return False

    phrases = re.split(r"[。、！？!?,.\s]+", t)
    phrases = [p.strip() for p in phrases if len(p.strip()) >= 2]

    if len(phrases) < OUTPUT_REPEAT_THRESHOLD:
        return False

    counter = Counter(phrases)
    most_common_count = counter.most_common(1)[0][1]
    if most_common_count >= OUTPUT_REPEAT_THRESHOLD:
        return True

    consecutive = 1
    for i in range(1, len(phrases)):
        if phrases[i] == phrases[i - 1]:
            consecutive += 1
            if consecutive >= OUTPUT_REPEAT_THRESHOLD:
                return True
        else:
            consecutive = 1

    return False


def is_translation_glitch(text: str) -> bool:
    """翻訳結果が異常かどうかを判定する。"""
    return _detect_repeated_phrases(text)


def _is_repetitive_input(text: str) -> bool:
    """入力テキストが繰り返しで翻訳不要かを判定する。"""
    t = (text or "").strip()
    if not t:
        return True

    phrases = re.findall(r'"([^"]+)"', t)
    if not phrases:
        phrases = re.split(r"(?<=[\.\!\?])\s+", t)
        phrases = [p.strip() for p in phrases if p.strip()]

    if len(phrases) <= 2:
        return False

    unique_phrases = set(p.strip().lower() for p in phrases)
    unique_ratio = len(unique_phrases) / len(phrases)
    if unique_ratio < INPUT_UNIQUE_RATIO_THRESHOLD:
        return True

    counter = Counter(p.strip().lower() for p in phrases)
    most_common_count = counter.most_common(1)[0][1]
    if most_common_count >= INPUT_REPEAT_THRESHOLD:
        return True

    return False


# =========================================================
# 統合翻訳実行
# =========================================================


def _translate_with_cat(text: str, source_lang: str, target_lang: str) -> str:
    """CAT-Translate で日英翻訳する。"""
    _ensure_engine("cat_translate")
    model = _get_cat_model()

    src = normalize_lang_code(source_lang)
    tgt = normalize_lang_code(target_lang)

    if src == "en" and tgt == "ja":
        prompt_content = (
            f"Translate the following English text into Japanese.\n\n{text}"
        )
    elif src == "ja" and tgt == "en":
        prompt_content = (
            f"Translate the following Japanese text into English.\n\n{text}"
        )
    else:
        prompt_content = (
            f"Translate the following text.\n\n{text}"
        )

    messages = [
        {"role": "user", "content": prompt_content},
    ]

    response = model.create_chat_completion(
        messages=messages,
        max_tokens=CAT_TRANSLATE_N_CTX // 2,
        temperature=0.0,
        top_p=1.0,
        repeat_penalty=CAT_TRANSLATE_REPEAT_PENALTY,
    )

    choices = response.get("choices", [])
    if not choices:
        return ""

    result = choices[0].get("message", {}).get("content", "")
    return result.strip()


def _translate_with_gemma(text: str, source_lang: str, target_lang: str) -> str:
    """TranslateGemma で多言語翻訳する。"""
    _ensure_engine("translategemma")

    from xlanguage_dubbing.translation.translategemma import translate_text_gemma

    return translate_text_gemma(
        text,
        source_lang_code=normalize_lang_code(source_lang),
        target_lang_code=normalize_lang_code(target_lang),
    )


def _translate_text(
    text: str, source_lang: str, target_lang: str
) -> str:
    """言語ペアに応じた最適なエンジンで翻訳する。"""
    engine = select_translation_engine(source_lang, target_lang)

    if engine == "cat_translate":
        return _translate_with_cat(text, source_lang, target_lang)
    else:
        return _translate_with_gemma(text, source_lang, target_lang)


# =========================================================
# 翻訳クライアント
# =========================================================


class CatTranslateClient:
    """統合翻訳クライアント。

    名前は後方互換性のために CatTranslateClient のまま。
    実際には言語ペアに応じて CAT-Translate と TranslateGemma を切り替える。
    """

    def __init__(self) -> None:
        self._call_count = 0
        self._gc_interval = 50

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "",
        retries: int = CAT_TRANSLATE_RETRIES,
        retry_backoff_sec: float = CAT_TRANSLATE_RETRY_BACKOFF_SEC,
    ) -> tuple[str, str]:
        """テキストを翻訳する。"""
        tgt = target_lang if target_lang else OUTPUT_LANG
        engine = select_translation_engine(source_lang, tgt)

        # TranslateGemma 用のリトライ設定
        if engine == "translategemma":
            retries = TRANSLATEGEMMA_RETRIES
            retry_backoff_sec = TRANSLATEGEMMA_RETRY_BACKOFF_SEC

        last_err: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                result = _translate_text(text, source_lang, tgt)
                self._call_count += 1
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

        raise CatTranslateError(f"翻訳失敗(リトライ枯渇): {last_err}")


# =========================================================
# セグメント翻訳
# =========================================================


def _new_segment_src_only(seg: Segment) -> Segment:
    """元テキストのみを保持したセグメントを生成する。"""
    return replace(seg, text_tgt="")


def translate_segment_safely(
    client: CatTranslateClient,
    text_src: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """セグメントを安全に翻訳する。"""
    t = (text_src or "").strip()
    if not t:
        return ""

    if _is_repetitive_input(t):
        print_step("    繰り返し入力を検出 → スキップ")
        return "（繰り返し音声）"

    if len(t) <= 2000:
        out, _ = client.translate(
            t, source_lang=source_lang, target_lang=target_lang
        )
        tgt = out.strip()
        if is_translation_glitch(tgt):
            return "翻訳エラー"
        return tgt

    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    outs: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if _is_repetitive_input(p):
            outs.append("繰り返し音声エラー")
            continue
        out, _ = client.translate(
            p, source_lang=source_lang, target_lang=target_lang
        )
        tgt = out.strip()
        if is_translation_glitch(tgt):
            outs.append("翻訳エラー")
        else:
            outs.append(tgt)

    joined = " ".join([o for o in outs if o])
    if is_translation_glitch(joined):
        return "翻訳エラー"
    return joined


def translate_segments_resumable(
    client: CatTranslateClient,
    segments_src: list[Segment],
    seg_json_path: Path,
    progress: ProgressStore,
    detected_lang: str = "",
) -> list[Segment]:
    """セグメントを再開可能な形式で翻訳する。

    各セグメントの detected_lang を翻訳の source_lang として使用する。
    detected_lang が空の場合はセグメントのテキストから自動検出する。
    """
    target_lang = normalize_lang_code(OUTPUT_LANG)

    if seg_json_path.exists():
        try:
            loaded = load_segments_json(seg_json_path)
            if len(loaded) == len(segments_src):
                segments = loaded
            else:
                segments = [_new_segment_src_only(s) for s in segments_src]
        except Exception:
            segments = [_new_segment_src_only(s) for s in segments_src]
    else:
        segments = [_new_segment_src_only(s) for s in segments_src]

    total = len(segments)
    prev_engine = ""

    for segno, seg in enumerate(segments, start=1):
        if (seg.text_tgt or "").strip():
            if is_translation_glitch(seg.text_tgt):
                segments[segno - 1] = replace(seg, text_tgt="翻訳エラー")
                save_segments_json_atomic(segments, seg_json_path)
                progress.set_step("translate_done", False)
                progress.set_artifact(
                    "segments_translated_json", str(seg_json_path)
                )
                progress.save()
            continue

        # セグメント単位の元言語を決定する
        seg_source_lang = (
            seg.detected_lang
            or detected_lang
            or detect_language_from_text(seg.text_src)
            or "en"
        )

        # 翻訳エンジンの表示（切り替え時のみ）
        engine = select_translation_engine(seg_source_lang, target_lang)
        if engine != prev_engine:
            engine_name = (
                "CAT-Translate" if engine == "cat_translate" else "TranslateGemma"
            )
            print_step(
                f"  翻訳エンジン: {engine_name} "
                f"({seg_source_lang} → {target_lang})"
            )
            prev_engine = engine

        print_step(
            f"  翻訳 {segno}/{total}: {seg.start:.3f}-{seg.end:.3f} "
            f"[{seg_source_lang}→{target_lang}] "
            f"'{seg.text_src[:60]}'"
        )
        tgt = translate_segment_safely(
            client, seg.text_src, seg_source_lang, target_lang
        )
        segments[segno - 1] = replace(seg, text_tgt=tgt)
        save_segments_json_atomic(segments, seg_json_path)
        progress.set_step("translate_done", False)
        progress.set_artifact("segments_translated_json", str(seg_json_path))
        progress.save()

    progress.set_step("translate_done", True)
    progress.save()
    return segments
