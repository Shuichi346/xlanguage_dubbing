#!/usr/bin/env python3
"""
whisper.cpp CLI（+ Silero VAD）による音声認識処理。
--language auto 対応: 自動言語判定結果を JSON から取得する。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from xlanguage_dubbing.config import (
    INPUT_LANG,
    VAD_MODEL,
    WHISPER_CPP_DIR,
    WHISPER_LANG,
    WHISPER_MODEL,
    WHISPER_SAMPLE_RATE,
)
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.utils import PipelineError, print_step, run_cmd, which_or_raise


def _resolve_whisper_cli() -> str:
    """whisper-cli バイナリのパスを解決する。"""
    candidate = WHISPER_CPP_DIR / "build" / "bin" / "whisper-cli"
    if candidate.exists():
        return str(candidate)

    import shutil
    path_bin = shutil.which("whisper-cli")
    if path_bin:
        return path_bin

    raise PipelineError(
        "whisper-cli が見つかりません。\n"
        "以下を実行して whisper.cpp をセットアップしてください:\n"
        "  chmod +x scripts/setup_whisper.sh\n"
        "  ./scripts/setup_whisper.sh\n"
    )


def _resolve_whisper_model() -> str:
    """Whisper モデルファイルのパスを解決する。"""
    model_file = WHISPER_CPP_DIR / "models" / f"ggml-{WHISPER_MODEL}.bin"
    if model_file.exists():
        return str(model_file)

    raise PipelineError(
        f"Whisper モデルが見つかりません: {model_file}\n"
        "  ./scripts/setup_whisper.sh を実行してください。\n"
    )


def _resolve_vad_model() -> str:
    """VAD モデルファイルのパスを解決する。"""
    vad_file = WHISPER_CPP_DIR / "models" / f"ggml-{VAD_MODEL}.bin"
    if vad_file.exists():
        return str(vad_file)

    raise PipelineError(
        f"VAD モデルが見つかりません: {vad_file}\n"
        "  ./scripts/setup_whisper.sh を実行してください。\n"
    )


def extract_wav_for_whisper(video_path: Path, wav_path: Path) -> None:
    """動画から16kHz mono WAVを抽出する。"""
    which_or_raise("ffmpeg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(WHISPER_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    run_cmd(cmd)


def extract_wav_for_vibevoice(video_path: Path, wav_path: Path) -> None:
    """動画から24kHz mono WAVを抽出する。"""
    from xlanguage_dubbing.config import VIBEVOICE_SAMPLE_RATE

    which_or_raise("ffmpeg")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(VIBEVOICE_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]
    run_cmd(cmd)


def whisper_transcribe(wav_path: Path) -> tuple[list[Segment], str]:
    """whisper.cpp CLI + VAD で音声を文字起こしする。

    戻り値: (セグメントリスト, 検出言語コード)
    """
    whisper_cli = _resolve_whisper_cli()
    model_path = _resolve_whisper_model()
    vad_model_path = _resolve_vad_model()

    n_threads = max(1, (os.cpu_count() or 8) - 2)

    output_base = wav_path.parent / wav_path.stem
    json_path = Path(f"{output_base}.json")

    # INPUT_LANG に基づいて whisper の言語を決定する
    lang = WHISPER_LANG if WHISPER_LANG else "auto"
    if INPUT_LANG and INPUT_LANG != "auto":
        lang = INPUT_LANG

    cmd = [
        whisper_cli,
        "--model", model_path,
        "--file", str(wav_path),
        "--language", lang,
        "--threads", str(n_threads),
        "--vad",
        "--vad-model", vad_model_path,
        "--output-json",
        "--output-file", str(output_base),
        "--no-prints",
    ]

    print_step(f"  whisper-cli 実行中: {wav_path.name}")
    print_step(f"    モデル: {WHISPER_MODEL}")
    print_step(f"    言語: {lang}")
    print_step(f"    VAD: {VAD_MODEL}")
    print_step(f"    スレッド数: {n_threads}")

    run_cmd(cmd)

    if not json_path.exists():
        raise PipelineError(
            f"whisper-cli の JSON 出力が見つかりません: {json_path}"
        )

    segments, detected_lang = _parse_whisper_json(json_path)

    if not segments:
        raise PipelineError("whisper.cpp: 文字起こし結果が空です。")

    print_step(f"  whisper.cpp 完了: {len(segments)} セグメント")
    if detected_lang:
        print_step(f"  検出言語: {detected_lang}")

    return segments, detected_lang


def _parse_whisper_json(json_path: Path) -> tuple[list[Segment], str]:
    """whisper.cpp の JSON 出力をパースする。

    戻り値: (セグメントリスト, 検出言語コード)
    """
    raw = json_path.read_text(encoding="utf-8")
    data = json.loads(raw)

    # whisper.cpp は result.language にフルネーム（"english"）を出力する
    detected_lang = ""
    result = data.get("result", {})
    if isinstance(result, dict):
        lang_full = result.get("language", "")
        if lang_full:
            detected_lang = _whisper_lang_name_to_code(lang_full)

    transcription = data.get("transcription", [])
    segments: list[Segment] = []

    for entry in transcription:
        offsets = entry.get("offsets", {})
        start_ms = int(offsets.get("from", 0))
        end_ms = int(offsets.get("to", 0))
        text = (entry.get("text", "") or "").strip()

        if not text:
            continue

        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0

        if end_sec <= start_sec:
            continue

        segments.append(
            Segment(
                idx=len(segments),
                start=start_sec,
                end=end_sec,
                text_src=text,
                detected_lang=detected_lang,
            )
        )

    return segments, detected_lang


# whisper.cpp が返す言語フルネーム → ISO 639-1 コードの対応表
_WHISPER_LANG_MAP: dict[str, str] = {
    "english": "en", "chinese": "zh", "german": "de", "spanish": "es",
    "russian": "ru", "korean": "ko", "french": "fr", "japanese": "ja",
    "portuguese": "pt", "turkish": "tr", "polish": "pl", "catalan": "ca",
    "dutch": "nl", "arabic": "ar", "swedish": "sv", "italian": "it",
    "indonesian": "id", "hindi": "hi", "finnish": "fi", "vietnamese": "vi",
    "hebrew": "he", "ukrainian": "uk", "greek": "el", "malay": "ms",
    "czech": "cs", "romanian": "ro", "danish": "da", "hungarian": "hu",
    "tamil": "ta", "norwegian": "no", "thai": "th", "urdu": "ur",
    "croatian": "hr", "bulgarian": "bg", "lithuanian": "lt", "latin": "la",
    "maori": "mi", "malayalam": "ml", "welsh": "cy", "slovak": "sk",
    "telugu": "te", "persian": "fa", "latvian": "lv", "bengali": "bn",
    "serbian": "sr", "azerbaijani": "az", "slovenian": "sl", "kannada": "kn",
    "estonian": "et", "macedonian": "mk", "breton": "br", "basque": "eu",
    "icelandic": "is", "armenian": "hy", "nepali": "ne", "mongolian": "mn",
    "bosnian": "bs", "kazakh": "kk", "albanian": "sq", "swahili": "sw",
    "galician": "gl", "marathi": "mr", "punjabi": "pa", "sinhala": "si",
    "khmer": "km", "shona": "sn", "yoruba": "yo", "somali": "so",
    "afrikaans": "af", "occitan": "oc", "georgian": "ka", "belarusian": "be",
    "tajik": "tg", "sindhi": "sd", "gujarati": "gu", "amharic": "am",
    "yiddish": "yi", "lao": "lo", "uzbek": "uz", "faroese": "fo",
    "haitian creole": "ht", "pashto": "ps", "turkmen": "tk", "nynorsk": "nn",
    "maltese": "mt", "sanskrit": "sa", "luxembourgish": "lb", "myanmar": "my",
    "tibetan": "bo", "tagalog": "tl", "malagasy": "mg", "assamese": "as",
    "tatar": "tt", "hawaiian": "haw", "lingala": "ln", "hausa": "ha",
    "bashkir": "ba", "javanese": "jv", "sundanese": "su",
}


def _whisper_lang_name_to_code(lang_name: str) -> str:
    """whisper.cpp の言語フルネームを ISO 639-1 コードに変換する。"""
    name = (lang_name or "").strip().lower()
    return _WHISPER_LANG_MAP.get(name, name[:2] if len(name) >= 2 else "")


def transcribe_short_audio(wav_path: Path, language: str = "") -> str:
    """短い音声ファイルを whisper.cpp で文字起こしする。"""
    if not wav_path.exists():
        return ""

    tmp_wav = wav_path.parent / f"{wav_path.stem}_w16k.wav"
    try:
        which_or_raise("ffmpeg")
        run_cmd([
            "ffmpeg", "-y",
            "-i", str(wav_path),
            "-vn", "-ac", "1", "-ar", str(WHISPER_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(tmp_wav),
        ])
    except Exception:
        return ""

    try:
        whisper_cli = _resolve_whisper_cli()
        model_path = _resolve_whisper_model()
    except PipelineError:
        tmp_wav.unlink(missing_ok=True)
        return ""

    lang = language if language else (INPUT_LANG if INPUT_LANG != "auto" else "auto")
    output_base = tmp_wav.parent / tmp_wav.stem
    json_path = Path(f"{output_base}.json")

    cmd = [
        whisper_cli,
        "--model", model_path,
        "--file", str(tmp_wav),
        "--language", lang,
        "--threads", "4",
        "--output-json",
        "--output-file", str(output_base),
        "--no-prints",
    ]

    try:
        run_cmd(cmd)
    except Exception:
        tmp_wav.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
        return ""

    text = ""
    if json_path.exists():
        try:
            segments, _ = _parse_whisper_json(json_path)
            text = " ".join(
                s.text_src.strip() for s in segments if s.text_src.strip()
            )
        except Exception:
            text = ""

    tmp_wav.unlink(missing_ok=True)
    json_path.unlink(missing_ok=True)

    return text.strip()


def release_whisper_model() -> None:
    """whisper.cpp CLI 方式ではモデルの明示的解放は不要。"""
    pass
