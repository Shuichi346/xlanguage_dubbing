from __future__ import annotations

import runpy
import unittest
from collections import Counter
from pathlib import Path


class ConfigMatrixTests(unittest.TestCase):
    def test_supported_engines_cover_the_eighteen_case_matrix(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        namespace = runpy.run_path(
            str(repo_root / "scripts" / "run_config_matrix.py"),
            run_name="config_matrix_test",
        )
        cases = namespace["build_cases"]()

        self.assertEqual(len(cases), 18)
        self.assertEqual(
            Counter(case.asr_engine for case in cases),
            Counter({"vibevoice": 6, "whisper": 6, "nemo-speech": 6}),
        )
        self.assertEqual(
            Counter(case.tts_engine for case in cases),
            Counter({"omnivoice": 6, "voxcpm2": 6, "irodori": 6}),
        )


if __name__ == "__main__":
    unittest.main()
