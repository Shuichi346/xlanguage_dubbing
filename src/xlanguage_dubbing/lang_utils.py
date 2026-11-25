#!/usr/bin/env python3
"""
言語検出・言語コード変換ユーティリティ。
ISO 639-1 コードの正規化、テキストからの言語検出、
PLaMo 翻訳の対応言語検証を提供する。
"""

from __future__ import annotations

from xlanguage_dubbing.utils import PipelineError, print_step

# Official PLaMo 3 Translate language codes; Chinese scripts count separately.
LANG_CODE_TO_NAME: dict[str, str] = {
    "ja": "Japanese", "en": "English",
    "zh-hans": "Chinese (Simplified)", "zh-hant": "Chinese (Traditional)",
    "ko": "Korean", "ar": "Arabic", "az": "Azerbaijani", "be": "Belarusian",
    "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "ceb": "Cebuano",
    "de": "German", "es": "Spanish", "et": "Estonian", "eu": "Basque",
    "fa": "Persian", "fr": "French", "gu": "Gujarati", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "it": "Italian", "jv": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "lt": "Lithuanian", "lv": "Latvian",
    "ml": "Malayalam", "mr": "Marathi", "ms": "Malay", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "pa": "Punjabi", "pl": "Polish",
    "ru": "Russian", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian",
    "sr": "Serbian", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "th": "Thai", "tl": "Tagalog", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
}


def detect_language_from_text(text: str) -> str:
    """テキストから言語を検出して ISO 639-1 コードで返す。

    検出失敗時は空文字列を返す。
    """
    t = (text or "").strip()
    if not t or len(t) < 3:
        return ""

    try:
        from langdetect import detect
        lang_code = detect(t)
        # langdetect は "zh-cn" のような形式を返す場合がある
        return normalize_lang_code(lang_code)
    except Exception:
        return ""


def get_lang_name(lang_code: str) -> str:
    """ISO 639-1 コードから英語の言語名を取得する。"""
    code = normalize_lang_code(lang_code)
    return LANG_CODE_TO_NAME.get(code, code.title() if code else "Unknown")


def normalize_lang_code(lang_code: str) -> str:
    """Normalize region aliases while preserving Chinese script distinctions."""
    code = (lang_code or "").strip().lower().replace("_", "-")
    if code == "zh" or code.startswith("zh-"):
        parts = code.split("-")[1:]
        if "hant" in parts or ("hans" not in parts and any(p in {"tw", "hk", "mo"} for p in parts)):
            return "zh-hant"
        return "zh-hans"
    return code.split("-")[0]


def speech_lang_code(lang_code: str) -> str:
    """Use the base Chinese code expected by speech backends, preserving PFML3 scripts elsewhere."""
    return normalize_lang_code(lang_code).split("-")[0]


def validate_translation_language(lang_code: str) -> str:
    """Reject languages outside the official PLaMo 3 Translate support list."""
    code = normalize_lang_code(lang_code)
    if code not in LANG_CODE_TO_NAME:
        raise PipelineError(
            f"PLaMo 3 Translate 8B 非対応言語: {lang_code!r}。"
            "対応言語コードは README の翻訳セクションを参照してください。"
        )
    return code


def validate_translation_languages(source_lang: str, target_lang: str) -> tuple[str, str]:
    """Validate both sides before inference or checkpoint reuse."""
    return validate_translation_language(source_lang), validate_translation_language(target_lang)


def detect_segments_language(segments: list, input_lang: str) -> str:
    """セグメント群から主要言語を検出する。

    input_lang が "auto" の場合、全セグメントのテキストから多数決で言語を判定する。
    input_lang が明示されている場合はそのまま返す。
    """
    lang = normalize_lang_code(input_lang)
    if lang and lang != "auto":
        return lang

    # セグメントのテキストから言語検出を行う
    lang_counts: dict[str, int] = {}
    for seg in segments:
        text = getattr(seg, "text_src", "") or getattr(seg, "text_en", "") or ""
        detected = detect_language_from_text(text)
        if detected:
            lang_counts[detected] = lang_counts.get(detected, 0) + 1

    if not lang_counts:
        print_step("  警告: 言語自動検出に失敗しました。en をフォールバックとして使用します。")
        return "en"

    # 最も多く検出された言語を返す
    majority_lang = max(lang_counts, key=lambda k: lang_counts[k])
    total = sum(lang_counts.values())
    print_step(
        f"  言語自動検出結果: {majority_lang} "
        f"({lang_counts[majority_lang]}/{total} セグメント, "
        f"{get_lang_name(majority_lang)})"
    )
    if len(lang_counts) > 1:
        details = ", ".join(
            f"{k}={v}" for k, v in sorted(
                lang_counts.items(), key=lambda x: -x[1]
            )
        )
        print_step(f"  検出言語内訳: {details}")

    return majority_lang
