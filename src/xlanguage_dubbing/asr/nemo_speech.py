"""NeMo-Speech.cpp transcription with word-level speaker tags."""

from __future__ import annotations

import json
import math
import shutil
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

from xlanguage_dubbing.config import (
    INPUT_LANG,
    MERGE_GAP_SEC,
    MERGE_MAX_CHARS,
    MERGE_MAX_SEC,
    NEMO_SPEECH_BIN,
    NEMO_SPEECH_DEVICE,
    NEMO_SPEECH_DIAR_MODEL,
    NEMO_SPEECH_MAX_SPEAKERS,
    NEMO_SPEECH_MODEL,
)
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.lang_utils import normalize_lang_code
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    normalize_spaces,
    print_step,
    run_cmd,
    which_or_raise,
)


def resolve_nemo_speech_cli() -> str:
    """Find the user-installed CLI even when ~/.local/bin is absent from PATH."""
    binary = str(Path(NEMO_SPEECH_BIN).expanduser())
    if binary == "nemo-speech" and not shutil.which(binary):
        installed = Path.home() / ".local" / "bin" / binary
        if installed.is_file():
            binary = str(installed)
    try:
        return which_or_raise(binary)
    except PipelineError as exc:
        raise PipelineError(
            "nemo-speech が見つかりません。NeMo-Speech.cpp をインストールするか、"
            " NEMO_SPEECH_BIN に実行ファイルのパスを設定してください。"
        ) from exc


def _run_transcription(wav_path: Path, *, diarize: bool, language: str) -> dict[str, Any]:
    if not wav_path.is_file():
        raise PipelineError(f"NeMo-Speech 入力音声が見つかりません: {wav_path}")
    command = [
        resolve_nemo_speech_cli(), "transcribe", str(wav_path.resolve()),
        "--model", NEMO_SPEECH_MODEL, "--device", NEMO_SPEECH_DEVICE, "--json",
    ]
    if diarize:
        command.extend(["--diarize", "--diar-model", NEMO_SPEECH_DIAR_MODEL])
    if language and language != "auto":
        command.extend(["--language", language])
    # The CLI owns its models; exiting releases ASR and diarization memory.
    result = run_cmd(command)
    try:
        data = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise PipelineError("NeMo-Speech の JSON 出力を読み込めません。") from exc
    if not isinstance(data, dict):
        raise PipelineError("NeMo-Speech の JSON 出力はオブジェクトである必要があります。")
    return data


def _parse_nemo_json(data: dict[str, Any]) -> tuple[list[Segment], list[DiarizationSegment]]:
    """Preserve speaker changes and pauses while grouping timed words into turns."""
    words = data.get("words")
    if not isinstance(words, list) or not words:
        raise PipelineError("NeMo-Speech の単語タイムスタンプが空です。")
    languages = data.get("languages", [])
    language = (
        normalize_lang_code(languages[0])
        if isinstance(languages, list) and len(languages) == 1 and isinstance(languages[0], str)
        else ""
    )
    segments: list[Segment] = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise PipelineError("NeMo-Speech の単語データが不正です。")
        text = normalize_spaces(word["word"])
        if not text:
            continue
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError("NeMo-Speech の単語時刻が不正です。") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise PipelineError("NeMo-Speech の単語時刻が不正です。")
        tag = word.get("speaker", 0)
        if type(tag) is not int or not 0 <= tag <= NEMO_SPEECH_MAX_SPEAKERS:
            raise PipelineError("NeMo-Speech の話者番号は1〜8（未割り当ては0）に限られます。")
        speaker = f"SPEAKER_{tag - 1:02d}" if tag else ""
        punctuation = all(unicodedata.category(char).startswith("P") for char in text)
        previous = segments[-1] if segments else None
        if (
            punctuation and previous is not None
            and speaker in {"", previous.speaker_id}
            and start - previous.end <= MERGE_GAP_SEC
        ):
            # Standalone punctuation can have zero duration and no speaker tag.
            segments[-1] = replace(
                previous, text_src=previous.text_src + text, end=max(previous.end, end)
            )
            continue
        if end == start:
            raise PipelineError("NeMo-Speech の発話単語に長さ0の時刻があります。")
        if (
            previous is not None and previous.speaker_id == speaker
            and start - previous.end <= MERGE_GAP_SEC
            and max(previous.end, end) - previous.start <= MERGE_MAX_SEC
            and len(previous.text_src) + len(text) + 1 <= MERGE_MAX_CHARS
        ):
            segments[-1] = replace(
                previous, text_src=previous.text_src + " " + text, end=max(previous.end, end)
            )
        else:
            segments.append(Segment(
                idx=len(segments), start=start, end=end, text_src=text,
                speaker_id=speaker, detected_lang=language,
            ))
    diarization = [
        DiarizationSegment(start=seg.start, end=seg.end, speaker=seg.speaker_id)
        for seg in segments if seg.speaker_id
    ]
    if not segments or not diarization:
        raise PipelineError("NeMo-Speech の文字起こし・話者分離結果が空です。")
    return segments, diarization


def nemo_speech_transcribe(wav_path: Path) -> tuple[list[Segment], list[DiarizationSegment]]:
    """Run the selected ASR and diarizer in one short-lived CLI process."""
    print_step(f"  NeMo-Speech 実行中: {wav_path.name} ({NEMO_SPEECH_DEVICE})")
    print_step(f"    ASR: {NEMO_SPEECH_MODEL}")
    print_step(f"    話者分離: {NEMO_SPEECH_DIAR_MODEL}（最大8人）")
    data = _run_transcription(wav_path, diarize=True, language=INPUT_LANG)
    segments, diarization = _parse_nemo_json(data)
    atomic_write_json(wav_path.parent / "nemo_speech.json", data)
    print_step(
        f"  NeMo-Speech 完了: {len(segments)} セグメント、"
        f"{len({seg.speaker for seg in diarization})} 話者"
    )
    return segments, diarization


def transcribe_short_audio_nemo(wav_path: Path, language: str = "") -> str:
    """Transcribe reference clips without loading an unnecessary diarizer."""
    data = _run_transcription(wav_path, diarize=False, language=language or INPUT_LANG)
    text = data.get("text", "")
    if not isinstance(text, str):
        raise PipelineError("NeMo-Speech の参照音声テキストが不正です。")
    return normalize_spaces(text)
