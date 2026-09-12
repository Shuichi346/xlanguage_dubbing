#!/usr/bin/env python3
"""
言語検出・言語コード変換ユーティリティ。
ISO 639-1 コードの正規化、テキストからの言語検出、
翻訳エンジン選択ロジックを提供する。
"""

from __future__ import annotations

from xlanguage_dubbing.utils import print_step

# ISO 639-1 コードと英語名の対応表（TranslateGemma 対応言語）
LANG_CODE_TO_NAME: dict[str, str] = {
    "aa": "Afar", "ab": "Abkhazian", "af": "Afrikaans", "ak": "Akan",
    "am": "Amharic", "an": "Aragonese", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bm": "Bambara", "bn": "Bengali", "bo": "Tibetan", "br": "Breton",
    "bs": "Bosnian", "ca": "Catalan", "ce": "Chechen", "co": "Corsican",
    "cs": "Czech", "cv": "Chuvash", "cy": "Welsh", "da": "Danish",
    "de": "German", "dv": "Divehi", "dz": "Dzongkha", "ee": "Ewe",
    "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "ff": "Fulah",
    "fi": "Finnish", "fo": "Faroese", "fr": "French", "fy": "Western Frisian",
    "ga": "Irish", "gd": "Scottish Gaelic", "gl": "Galician", "gn": "Guarani",
    "gu": "Gujarati", "gv": "Manx", "ha": "Hausa", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian", "hu": "Hungarian",
    "hy": "Armenian", "ia": "Interlingua", "id": "Indonesian", "ie": "Interlingue",
    "ig": "Igbo", "ii": "Sichuan Yi", "ik": "Inupiaq", "io": "Ido",
    "is": "Icelandic", "it": "Italian", "iu": "Inuktitut", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "ki": "Kikuyu", "kk": "Kazakh",
    "kl": "Kalaallisut", "km": "Central Khmer", "kn": "Kannada", "ko": "Korean",
    "ks": "Kashmiri", "ku": "Kurdish", "kw": "Cornish", "ky": "Kyrgyz",
    "la": "Latin", "lb": "Luxembourgish", "lg": "Ganda", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "lu": "Luba-Katanga", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "nb": "Norwegian Bokmal", "nd": "North Ndebele",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian",
    "nr": "South Ndebele", "nv": "Navajo", "ny": "Chichewa", "oc": "Occitan",
    "om": "Oromo", "or": "Oriya", "os": "Ossetian", "pa": "Punjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "qu": "Quechua",
    "rm": "Romansh", "rn": "Rundi", "ro": "Romanian", "ru": "Russian",
    "rw": "Kinyarwanda", "sa": "Sanskrit", "sc": "Sardinian", "sd": "Sindhi",
    "se": "Northern Sami", "sg": "Sango", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "ss": "Swati", "st": "Southern Sotho", "su": "Sundanese",
    "sv": "Swedish", "sw": "Swahili", "ta": "Tamil", "te": "Telugu",
    "tg": "Tajik", "th": "Thai", "ti": "Tigrinya", "tk": "Turkmen",
    "tl": "Tagalog", "tn": "Tswana", "to": "Tonga", "tr": "Turkish",
    "ts": "Tsonga", "tt": "Tatar", "ug": "Uyghur", "uk": "Ukrainian",
    "ur": "Urdu", "uz": "Uzbek", "ve": "Venda", "vi": "Vietnamese",
    "vo": "Volapuk", "wa": "Walloon", "wo": "Wolof", "xh": "Xhosa",
    "yi": "Yiddish", "yo": "Yoruba", "za": "Zhuang", "zh": "Chinese",
    "zu": "Zulu",
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
        return lang_code.split("-")[0].lower()
    except Exception:
        return ""


def get_lang_name(lang_code: str) -> str:
    """ISO 639-1 コードから英語の言語名を取得する。"""
    code = (lang_code or "").strip().lower().split("-")[0]
    return LANG_CODE_TO_NAME.get(code, code.title() if code else "Unknown")


def normalize_lang_code(lang_code: str) -> str:
    """言語コードを正規化する（小文字、ハイフン前の基本コードのみ取得）。"""
    return (lang_code or "").strip().lower().split("-")[0]


def is_ja_en_pair(source_lang: str, target_lang: str) -> bool:
    """翻訳が日英ペア（CAT-Translate が最適）かどうかを判定する。"""
    src = normalize_lang_code(source_lang)
    tgt = normalize_lang_code(target_lang)
    return (src == "ja" and tgt == "en") or (src == "en" and tgt == "ja")


def select_translation_engine(source_lang: str, target_lang: str) -> str:
    """翻訳エンジンを選択する。

    日英/英日ペア → "cat_translate"
    それ以外 → "translategemma"
    """
    if is_ja_en_pair(source_lang, target_lang):
        return "cat_translate"
    return "translategemma"


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
