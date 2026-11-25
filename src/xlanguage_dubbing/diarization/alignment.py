#!/usr/bin/env python3
"""
Whisperセグメントと話者分離結果の突合処理。
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace

from xlanguage_dubbing.core.models import DiarizationSegment, Segment


@dataclass(slots=True)
class _IntervalNode:
    index: int
    turn: DiarizationSegment
    max_end: float
    left: _IntervalNode | None = None
    right: _IntervalNode | None = None


def _build_index(turns: list[tuple[int, DiarizationSegment]]) -> _IntervalNode | None:
    if not turns:
        return None
    middle = len(turns) // 2
    index, turn = turns[middle]
    left = _build_index(turns[:middle])
    right = _build_index(turns[middle + 1 :])
    return _IntervalNode(
        index,
        turn,
        max(
            turn.end,
            left.max_end if left else float("-inf"),
            right.max_end if right else float("-inf"),
        ),
        left,
        right,
    )


def _find_overlaps(
    node: _IntervalNode | None,
    start: float,
    end: float,
    found: list[tuple[int, DiarizationSegment]],
) -> None:
    if node is None or node.max_end <= start:
        return
    _find_overlaps(node.left, start, end, found)
    if node.turn.start < end:
        if node.turn.end > start:
            found.append((node.index, node.turn))
        _find_overlaps(node.right, start, end, found)


def assign_speakers(
    segments: list[Segment],
    diarization: list[DiarizationSegment],
) -> list[Segment]:
    """各Whisperセグメントに最も重複時間が長い話者IDを割り当てる。"""
    result: list[Segment] = []
    prev_speaker = ""
    tree = _build_index(sorted(enumerate(diarization), key=lambda item: item[1].start))
    # Nearest-endpoint fallback preserves original input order on equal distances.
    endpoint_indices: dict[float, int] = {}
    for index, turn in enumerate(diarization):
        endpoint_indices.setdefault(turn.start, index)
        endpoint_indices.setdefault(turn.end, index)
    endpoints = sorted(endpoint_indices)

    for seg in segments:
        overlap: dict[str, float] = {}
        candidates: list[tuple[int, DiarizationSegment]] = []
        _find_overlaps(tree, seg.start, seg.end, candidates)
        for _, dia in sorted(candidates):
            ov_start = max(seg.start, dia.start)
            ov_end = min(seg.end, dia.end)
            if ov_end > ov_start:
                overlap[dia.speaker] = overlap.get(dia.speaker, 0.0) + (ov_end - ov_start)

        if overlap:
            best_speaker = max(overlap, key=lambda k: overlap[k])
        else:
            best_speaker = prev_speaker if prev_speaker else ""

            if not best_speaker and diarization:
                seg_mid = (seg.start + seg.end) / 2.0
                position = bisect_left(endpoints, seg_mid)
                nearby = endpoints[max(0, position - 1) : position + 1]
                endpoint = min(
                    nearby, key=lambda value: (abs(value - seg_mid), endpoint_indices[value])
                )
                closest = diarization[endpoint_indices[endpoint]]
                best_speaker = closest.speaker

            if not best_speaker:
                best_speaker = "UNKNOWN"

        prev_speaker = best_speaker

        result.append(replace(seg, speaker_id=best_speaker))
    return result
