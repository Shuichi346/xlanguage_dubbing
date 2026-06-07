#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the dubbing pipeline across the supported engine configuration matrix."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ASR_ENGINES = ("vibevoice", "whisper")
AUDIO_SEPARATION_VALUES = ("true", "false")
TTS_ENGINES = ("omnivoice", "voxcpm2", "irodori")

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_TEST_VIDEO = REPO_ROOT / "input_videos" / "test.mp4"
RUN_ROOT = REPO_ROOT / "temp" / "config_matrix"


@dataclass(frozen=True)
class MatrixCase:
    asr_engine: str
    enable_audio_separation: str
    tts_engine: str

    @property
    def case_id(self) -> str:
        audio_mode = "separated" if self.enable_audio_separation == "true" else "rawaudio"
        return f"asr-{self.asr_engine}__audio-{audio_mode}__tts-{self.tts_engine}"


@dataclass
class CaseResult:
    case_id: str
    asr_engine: str
    enable_audio_separation: str
    tts_engine: str
    returncode: int | None
    status: str
    log_path: str
    output_path: str
    started_at: str | None = None
    finished_at: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run xlanguage-dubbing against input_videos/test.mp4 for every "
            "ASR_ENGINE x ENABLE_AUDIO_SEPARATION x TTS_ENGINE combination."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cases without running the pipeline.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed combination.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove the matrix run directory before starting.",
    )
    return parser.parse_args()


def build_cases() -> list[MatrixCase]:
    return [
        MatrixCase(asr_engine, audio_separation, tts_engine)
        for asr_engine, audio_separation, tts_engine in itertools.product(
            ASR_ENGINES,
            AUDIO_SEPARATION_VALUES,
            TTS_ENGINES,
        )
    ]


def ensure_test_video() -> None:
    if FIXED_TEST_VIDEO.is_file():
        return
    raise SystemExit(f"Fixed test video was not found: {FIXED_TEST_VIDEO}")


def prepare_case_input(case_dir: Path) -> Path:
    input_dir = case_dir / "input_videos"
    input_dir.mkdir(parents=True, exist_ok=True)
    test_link = input_dir / "test.mp4"
    if test_link.exists() or test_link.is_symlink():
        test_link.unlink()
    test_link.symlink_to(FIXED_TEST_VIDEO)
    return input_dir


def run_case(case: MatrixCase) -> CaseResult:
    case_dir = RUN_ROOT / case.case_id
    log_dir = RUN_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    input_dir = prepare_case_input(case_dir)
    temp_root = case_dir / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)

    output_suffix = f"_{case.case_id}_xlDub.mp4"
    output_path = input_dir / f"test{output_suffix}"
    log_path = log_dir / f"{case.case_id}.log"
    started_at = timestamp()

    env = os.environ.copy()
    env.update(
        {
            "VIDEO_FOLDER": str(input_dir),
            "TEMP_ROOT": str(temp_root),
            "INPUT_LANG": "en",
            "OUTPUT_LANG": "ja",
            "ASR_ENGINE": case.asr_engine,
            "ENABLE_AUDIO_SEPARATION": case.enable_audio_separation,
            "TTS_ENGINE": case.tts_engine,
            "OUTPUT_SUFFIX": output_suffix,
            "KEEP_TEMP": "true",
        }
    )

    command = ("uv", "run", "xlanguage-dubbing")
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"case_id={case.case_id}\n")
        log_file.write(f"command={' '.join(command)}\n")
        log_file.write(f"VIDEO_FOLDER={input_dir}\n")
        log_file.write(f"TEMP_ROOT={temp_root}\n")
        log_file.write(f"OUTPUT_SUFFIX={output_suffix}\n")
        log_file.write("\n")
        log_file.flush()

        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    finished_at = timestamp()
    status = "passed" if completed.returncode == 0 and output_path.exists() else "failed"
    return CaseResult(
        case_id=case.case_id,
        asr_engine=case.asr_engine,
        enable_audio_separation=case.enable_audio_separation,
        tts_engine=case.tts_engine,
        returncode=completed.returncode,
        status=status,
        log_path=str(log_path),
        output_path=str(output_path),
        started_at=started_at,
        finished_at=finished_at,
    )


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_summary(results: list[CaseResult]) -> Path:
    summary_path = RUN_ROOT / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fixed_test_video": str(FIXED_TEST_VIDEO),
        "generated_at": timestamp(),
        "total": len(results),
        "passed": sum(1 for result in results if result.status == "passed"),
        "failed": sum(1 for result in results if result.status == "failed"),
        "results": [asdict(result) for result in results],
    }
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def main() -> int:
    args = parse_args()
    ensure_test_video()

    cases = build_cases()
    print(f"Fixed test video: {FIXED_TEST_VIDEO}")
    print(f"Matrix cases: {len(cases)}")

    if args.clean and RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)

    if args.dry_run:
        for case in cases:
            print(
                f"{case.case_id}: "
                f"INPUT_LANG=en OUTPUT_LANG=ja "
                f"ASR_ENGINE={case.asr_engine} "
                f"ENABLE_AUDIO_SEPARATION={case.enable_audio_separation} "
                f"TTS_ENGINE={case.tts_engine}"
            )
        return 0

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    results: list[CaseResult] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.case_id}")
        result = run_case(case)
        results.append(result)
        print(f"  {result.status}: {result.log_path}")
        if args.fail_fast and result.status != "passed":
            break

    summary_path = write_summary(results)
    failed = [result for result in results if result.status != "passed"]
    print(f"Summary: {summary_path}")
    if failed:
        print("Failed cases:")
        for result in failed:
            print(f"  - {result.case_id}: {result.log_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
