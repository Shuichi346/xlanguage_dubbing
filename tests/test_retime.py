"""Regression coverage for preserving and sequencing overlapping turns."""

import unittest

from xlanguage_dubbing.core.models import Segment, TtsMeta
from xlanguage_dubbing.core.retime import build_retime_parts


class RetimeOverlapTests(unittest.TestCase):
    def _parts(self, ranges, *, length=10):
        segments = [
            Segment(i, start, end, f"turn {i}", speaker_id=str(i))
            for i, (start, end) in enumerate(ranges)
        ]
        metadata = {
            i: TtsMeta(i, "unused.flac", max(0.2, end - start))
            for i, (start, end) in enumerate(ranges, 1)
        }
        parts, duration = build_retime_parts(segments, metadata, length)
        speech = [part for part in parts if part.kind == "speech"]
        self.assertCountEqual([part.segno for part in speech], metadata)
        self.assertTrue(
            all(
                left.orig_end <= right.orig_start
                for left, right in zip(parts, parts[1:], strict=False)
            )
        )
        self.assertAlmostEqual(sum(part.orig_end - part.orig_start for part in parts), length)
        return parts, duration

    def test_contained_turn_preserves_every_utterance_without_replay(self):
        parts, duration = self._parts([(0, 5), (1, 2)])
        self.assertEqual(
            [(part.orig_start, part.orig_end) for part in parts], [(0, 1), (1, 5), (5, 10)]
        )
        self.assertEqual(duration, 11)

    def test_partial_overlap_has_monotonic_source_intervals(self):
        self._parts([(0, 5), (4, 8)])

    def test_simultaneous_and_end_of_media_turns_hold_boundary_frames(self):
        parts, _ = self._parts([(0, 5), (0, 5), (10, 10)])
        self.assertEqual(parts[0].orig_start, parts[0].orig_end)
        self.assertEqual(parts[0].speed, 0)
        self.assertEqual(parts[-1].orig_start, parts[-1].orig_end)

    def test_unsorted_turns_keep_original_metadata_numbers(self):
        parts, _ = self._parts([(4, 8), (0, 5)])
        self.assertEqual([part.segno for part in parts if part.kind == "speech"], [2, 1])

    def test_multiple_contained_turns_cover_the_full_union(self):
        self._parts([(0, 8), (1, 2), (3, 4), (5, 6)])


if __name__ == "__main__":
    unittest.main()
