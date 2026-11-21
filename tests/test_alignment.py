"""Compare indexed alignment against the established overlap/tie semantics."""

import random
import unittest

from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.diarization.alignment import assign_speakers


def baseline(segments, turns):
    speakers = []
    previous = ""
    for segment in segments:
        scores = {}
        for turn in turns:
            overlap = min(segment.end, turn.end) - max(segment.start, turn.start)
            if overlap > 0:
                scores[turn.speaker] = scores.get(turn.speaker, 0) + overlap
        if scores:
            speaker = max(scores, key=lambda name: scores[name])
        elif previous:
            speaker = previous
        elif turns:
            midpoint = (segment.start + segment.end) / 2
            speaker = min(
                turns, key=lambda turn: min(abs(turn.start - midpoint), abs(turn.end - midpoint))
            ).speaker
        else:
            speaker = "UNKNOWN"
        speakers.append(speaker)
        previous = speaker
    return speakers


class AlignmentIndexTests(unittest.TestCase):
    def test_matches_baseline_with_unsorted_nested_turns_and_ties(self):
        generator = random.Random(42)
        turns = [
            DiarizationSegment(start, start + generator.randint(1, 12), str(i % 4))
            for i in range(120)
            for start in [generator.randint(0, 90)]
        ]
        segments = [
            Segment(i, start, start + generator.randint(0, 20), "text")
            for i in range(100)
            for start in [generator.randint(-10, 110)]
        ]
        self.assertEqual(
            [s.speaker_id for s in assign_speakers(segments, turns)], baseline(segments, turns)
        )

    def test_empty_and_equal_distance_nearest_endpoint_fallback(self):
        segments = [Segment(0, 4, 6, "text"), Segment(1, 20, 21, "text")]
        turns = [DiarizationSegment(8, 10, "first"), DiarizationSegment(0, 2, "second")]
        self.assertEqual(
            [s.speaker_id for s in assign_speakers(segments, turns)], ["first", "first"]
        )
        self.assertEqual(assign_speakers(segments, [])[0].speaker_id, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
