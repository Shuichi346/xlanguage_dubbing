#!/usr/bin/env python3
"""
リタイム関連処理。
音声に合わせて動画速度を変更する。
"""

from __future__ import annotations

from xlanguage_dubbing.core.models import RetimePart, Segment, TtsMeta

_MIN_CHUNK_SEC = 0.15


def build_retime_parts(
    segments: list[Segment],
    tts_meta: dict[int, TtsMeta],
    video_duration_sec: float,
) -> tuple[list[RetimePart], float]:
    """元動画をギャップ・発話・末尾に分解する。"""
    raw_parts: list[RetimePart] = []
    cursor = 0.0
    coverage_end = 0.0
    vd = max(0.0, float(video_duration_sec))
    ordered = sorted(enumerate(segments, start=1), key=lambda item: item[1].start)

    for position, (segno, seg) in enumerate(ordered):
        start = min(vd, max(cursor, float(seg.start), 0.0))
        coverage_end = max(coverage_end, min(vd, max(start, float(seg.end))))
        end = coverage_end
        if position + 1 < len(ordered):
            # Each source interval belongs to only one turn. A contained or
            # simultaneous turn still receives its TTS duration on the timeline.
            next_start = min(vd, max(start, float(ordered[position + 1][1].start)))
            end = min(end, next_start)

        if start > cursor + 1e-6:
            raw_parts.append(RetimePart("gap", cursor, start, start - cursor, 1.0))

        orig_dur = end - start
        meta = tts_meta.get(segno)
        out_dur = float(meta.duration_sec) if meta and meta.duration_sec > 0 else orig_dur
        if out_dur > 0:
            # A zero-width interval is a held boundary frame, with silence in
            # the background track; it must not replay an overlapping interval.
            speed = orig_dur / out_dur if orig_dur > 0 else 0.0
            raw_parts.append(RetimePart("speech", start, end, out_dur, speed, segno))
        cursor = end

    if vd > cursor + 1e-6:
        tail_dur = vd - cursor
        raw_parts.append(
            RetimePart(
                kind="tail",
                orig_start=cursor,
                orig_end=vd,
                out_duration=tail_dur,
                speed=1.0,
            )
        )

    parts = _merge_tiny_parts(raw_parts)
    total_out = sum(p.out_duration for p in parts)
    return parts, float(total_out)


def _merge_tiny_parts(raw: list[RetimePart]) -> list[RetimePart]:
    """短すぎるパートを隣接パートに吸収する。"""
    if not raw:
        return []

    result: list[RetimePart] = []
    i = 0
    while i < len(raw):
        part = raw[i]
        orig_dur = max(0.0, part.orig_end - part.orig_start)

        if part.kind == "speech":
            result.append(part)
            i += 1
            continue

        if orig_dur >= _MIN_CHUNK_SEC:
            result.append(part)
            i += 1
            continue

        if i + 1 < len(raw) and raw[i + 1].kind == "speech":
            nxt = raw[i + 1]
            new_orig_start = part.orig_start
            new_orig_dur = max(0.001, nxt.orig_end - new_orig_start)
            new_speed = new_orig_dur / max(0.001, nxt.out_duration)
            result.append(
                RetimePart(
                    kind=nxt.kind,
                    orig_start=new_orig_start,
                    orig_end=nxt.orig_end,
                    out_duration=nxt.out_duration,
                    speed=new_speed,
                    segno=nxt.segno,
                )
            )
            i += 2
            continue

        if result and result[-1].kind == "speech":
            prev = result[-1]
            new_orig_end = part.orig_end
            new_orig_dur = max(0.001, new_orig_end - prev.orig_start)
            new_speed = new_orig_dur / max(0.001, prev.out_duration)
            result[-1] = RetimePart(
                kind=prev.kind,
                orig_start=prev.orig_start,
                orig_end=new_orig_end,
                out_duration=prev.out_duration,
                speed=new_speed,
                segno=prev.segno,
            )
            i += 1
            continue

        result.append(part)
        i += 1

    return result
