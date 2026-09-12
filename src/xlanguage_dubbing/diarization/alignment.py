#!/usr/bin/env python3
"""
Whisperセグメントと話者分離結果の突合処理。
"""

from __future__ import annotations

from xlanguage_dubbing.core.models import DiarizationSegment, Segment


def assign_speakers(
    segments: list[Segment],
    diarization: list[DiarizationSegment],
) -> list[Segment]:
    """各Whisperセグメントに最も重複時間が長い話者IDを割り当てる。"""
    result: list[Segment] = []
    prev_speaker = ""

    for seg in segments:
        overlap: dict[str, float] = {}
        for dia in diarization:
            ov_start = max(seg.start, dia.start)
            ov_end = min(seg.end, dia.end)
            if ov_end > ov_start:
                overlap[dia.speaker] = (
                    overlap.get(dia.speaker, 0.0) + (ov_end - ov_start)
                )

        if overlap:
            best_speaker = max(overlap, key=lambda k: overlap[k])
        else:
            best_speaker = prev_speaker if prev_speaker else ""

            if not best_speaker and diarization:
                seg_mid = (seg.start + seg.end) / 2.0
                closest = min(
                    diarization,
                    key=lambda d: min(
                        abs(d.start - seg_mid), abs(d.end - seg_mid)
                    ),
                )
                best_speaker = closest.speaker

            if not best_speaker:
                best_speaker = "UNKNOWN"

        prev_speaker = best_speaker

        result.append(Segment(
            idx=seg.idx,
            start=seg.start,
            end=seg.end,
            text_src=seg.text_src,
            text_tgt=seg.text_tgt,
            speaker_id=best_speaker,
            detected_lang=seg.detected_lang,
        ))
    return result
