#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spaCyによるセグメント文分割処理。
"""

from __future__ import annotations

import re
from typing import List, Optional

from xlanguage_dubbing.config import SPACY_MIN_WEIGHT, SPACY_MODEL
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.utils import PipelineError, normalize_spaces

_SPACY_NLP = None


def _get_spacy_nlp():
    """spaCyのNLPを遅延ロードする。"""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
    except Exception as exc:
        raise PipelineError(
            "spaCy がインストールされていません。\n"
            "  uv sync を実行してください。\n"
        ) from exc

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    except Exception as exc:
        raise PipelineError(
            f"spaCyモデル '{SPACY_MODEL}' が読み込めません。\n"
            f"  uv run python -m spacy download {SPACY_MODEL}\n"
        ) from exc

    if not (
        nlp.has_pipe("parser")
        or nlp.has_pipe("senter")
        or nlp.has_pipe("sentencizer")
    ):
        nlp.add_pipe("sentencizer")

    _SPACY_NLP = nlp
    return _SPACY_NLP


def _weight_for_time_allocation(text: str) -> int:
    """タイムコード配分の重みを計算する。"""
    t = re.sub(r"\s+", "", (text or ""))
    return max(SPACY_MIN_WEIGHT, len(t))


def chunk_segments_for_spacy(
    segments: List[Segment],
    *,
    max_sec: float,
    max_chars: int,
    max_gap_sec: float,
) -> List[Segment]:
    """spaCy処理前にセグメントをチャンク化する。"""
    if not segments:
        return []

    out: List[Segment] = []
    buf: Optional[Segment] = None

    def flush() -> None:
        nonlocal buf
        if buf is not None and (buf.text_src or "").strip():
            out.append(buf)
        buf = None

    for s in segments:
        text = (s.text_src or "").strip()
        if not text:
            continue

        if buf is None:
            buf = Segment(
                idx=len(out), start=s.start, end=s.end,
                text_src=text, speaker_id=s.speaker_id,
                detected_lang=s.detected_lang,
            )
            continue

        gap = max(0.0, s.start - buf.end)
        new_text = (buf.text_src + " " + text).strip()
        new_dur = max(0.0, s.end - buf.start)

        can_merge = (
            gap <= max_gap_sec
            and new_dur <= max_sec
            and len(new_text) <= max_chars
            and buf.speaker_id == s.speaker_id
        )

        if can_merge:
            buf = Segment(
                idx=buf.idx, start=buf.start, end=s.end,
                text_src=new_text, speaker_id=buf.speaker_id,
                detected_lang=buf.detected_lang or s.detected_lang,
            )
        else:
            flush()
            buf = Segment(
                idx=len(out), start=s.start, end=s.end,
                text_src=text, speaker_id=s.speaker_id,
                detected_lang=s.detected_lang,
            )

    flush()
    return [
        Segment(
            idx=i, start=s.start, end=s.end,
            text_src=s.text_src, speaker_id=s.speaker_id,
            detected_lang=s.detected_lang,
        )
        for i, s in enumerate(out)
    ]


def split_segments_by_spacy_sentences(segments: List[Segment]) -> List[Segment]:
    """セグメントをspaCyで文単位に分割する。"""
    if not segments:
        return []

    nlp = _get_spacy_nlp()
    out: List[Segment] = []

    for seg in segments:
        text = (seg.text_src or "").strip()
        if not text:
            continue

        try:
            doc = nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text and s.text.strip()]
        except Exception:
            sents = re.split(r"(?<=[\.\!\?])\s+", text)
            sents = [s.strip() for s in sents if s.strip()]

        if len(sents) <= 1 or seg.duration <= 0.0:
            out.append(
                Segment(
                    idx=len(out),
                    start=seg.start,
                    end=seg.end,
                    text_src=normalize_spaces(text),
                    speaker_id=seg.speaker_id,
                    detected_lang=seg.detected_lang,
                )
            )
            continue

        weights = [_weight_for_time_allocation(s) for s in sents]
        total_w = float(sum(weights)) if weights else 0.0
        if total_w <= 0.0:
            out.append(
                Segment(
                    idx=len(out),
                    start=seg.start,
                    end=seg.end,
                    text_src=normalize_spaces(text),
                    speaker_id=seg.speaker_id,
                    detected_lang=seg.detected_lang,
                )
            )
            continue

        start = seg.start
        dur = seg.duration
        acc = 0.0

        for i, (sent, w) in enumerate(zip(sents, weights, strict=True)):
            if i == len(sents) - 1:
                sent_start = start + dur * (acc / total_w)
                sent_end = seg.end
            else:
                sent_start = start + dur * (acc / total_w)
                acc += float(w)
                sent_end = start + dur * (acc / total_w)

            sent_start = max(seg.start, min(sent_start, seg.end))
            sent_end = max(sent_start, min(sent_end, seg.end))

            out.append(
                Segment(
                    idx=len(out),
                    start=float(sent_start),
                    end=float(sent_end),
                    text_src=normalize_spaces(sent),
                    speaker_id=seg.speaker_id,
                    detected_lang=seg.detected_lang,
                )
            )

    return out


def initialize_spacy() -> None:
    """spaCyを初期化する。"""
    _get_spacy_nlp()
