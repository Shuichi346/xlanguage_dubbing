from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.tts import irodori_reference_worker as worker
from xlanguage_dubbing.tts import reference


def _segments(speakers: str) -> list[Segment]:
    return [
        Segment(
            idx=index,
            start=float(index - 1),
            end=float(index) - 0.1,
            text_src=f"line {index}",
            speaker_id=speaker,
        )
        for index, speaker in enumerate(speakers, start=1)
    ]


class IrodoriReferenceCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache_dir = self.root / "speaker_refs"
        self.server_dir = self.root / "Irodori-TTS-Server"
        self.server_dir.mkdir()
        (self.server_dir / "uv.lock").write_text("locked", encoding="utf-8")
        self.media_path = self.root / "voice.wav"
        self.media_path.write_bytes(b"source-audio")
        self.worker_manifests: list[dict] = []

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _fake_extract(self, _media, output, **_kwargs) -> None:
        Path(output).write_bytes(b"RIFF" + (b"\0" * 256))

    def _fake_worker(self, manifest_path: Path, *, server_dir: Path) -> None:
        self.assertEqual(server_dir, self.server_dir.resolve())
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.worker_manifests.append(manifest)
        for speaker_index, speaker in enumerate(manifest["speakers"], start=1):
            latent = torch.full((10, 32), float(speaker_index), dtype=torch.float32)
            torch.save(latent, speaker["output_path"])

    def _patch_generation(self):
        return (
            mock.patch.object(reference, "IRODORI_TTS_DIR", self.server_dir),
            mock.patch.object(
                reference,
                "extract_audio_segment",
                side_effect=self._fake_extract,
            ),
            mock.patch.object(
                reference,
                "_run_irodori_reference_worker",
                side_effect=self._fake_worker,
            ),
        )

    def test_ababcabc_builds_one_latent_per_speaker_and_reuses_it(self) -> None:
        patches = self._patch_generation()
        with patches[0], patches[1] as extract_mock, patches[2] as worker_mock:
            cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                tts_engine="irodori",
            )
            cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABABCABC"),
            )

            self.assertEqual(worker_mock.call_count, 1)
            self.assertGreater(extract_mock.call_count, 3)
            manifest = self.worker_manifests[0]
            self.assertEqual(
                [item["speaker_id"] for item in manifest["speakers"]],
                ["A", "B", "C"],
            )
            self.assertEqual(len(list(self.cache_dir.glob("*.pt"))), 3)
            first_paths = {
                speaker: cache.get_irodori_speaker_reference_latent_path(speaker)
                for speaker in "ABC"
            }

        self.worker_manifests.clear()
        patches = self._patch_generation()
        with patches[0], patches[1] as extract_mock, patches[2] as worker_mock:
            resumed_cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                tts_engine="irodori",
            )
            resumed_cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABABCABC"),
            )

            worker_mock.assert_not_called()
            extract_mock.assert_not_called()
            for speaker in "ABC":
                self.assertEqual(
                    resumed_cache.get_irodori_speaker_reference_latent_path(speaker),
                    first_paths[speaker],
                )

    def test_corrupt_cache_regenerates_only_the_affected_speaker(self) -> None:
        patches = self._patch_generation()
        with patches[0], patches[1], patches[2]:
            cache = reference.SpeakerReferenceCache(self.cache_dir, "irodori")
            cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABABCABC"),
            )

        corrupt_path = cache.get_irodori_speaker_reference_latent_path("B")
        self.assertIsNotNone(corrupt_path)
        corrupt_path.write_bytes(b"corrupt")
        self.worker_manifests.clear()

        patches = self._patch_generation()
        with patches[0], patches[1], patches[2] as worker_mock:
            resumed_cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                "irodori",
            )
            resumed_cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABABCABC"),
            )

        self.assertEqual(worker_mock.call_count, 1)
        self.assertEqual(
            [item["speaker_id"] for item in self.worker_manifests[0]["speakers"]],
            ["B"],
        )

    def test_partial_worker_failure_reuses_completed_speaker_on_resume(self) -> None:
        def partially_failing_worker(
            manifest_path: Path,
            *,
            server_dir: Path,
        ) -> None:
            self.assertEqual(server_dir, self.server_dir.resolve())
            manifest = json.loads(
                Path(manifest_path).read_text(encoding="utf-8")
            )
            speaker = manifest["speakers"][0]
            output_path = Path(speaker["output_path"])
            latent = torch.ones((10, 32), dtype=torch.float32)
            torch.save(latent, output_path)
            worker._update_cache_metadata(
                manifest,
                speaker,
                latent,
                output_path,
            )
            raise reference.PipelineError("speaker B failed")

        with (
            mock.patch.object(reference, "IRODORI_TTS_DIR", self.server_dir),
            mock.patch.object(
                reference,
                "extract_audio_segment",
                side_effect=self._fake_extract,
            ),
            mock.patch.object(
                reference,
                "_run_irodori_reference_worker",
                side_effect=partially_failing_worker,
            ),
        ):
            cache = reference.SpeakerReferenceCache(self.cache_dir, "irodori")
            with self.assertRaisesRegex(reference.PipelineError, "speaker B failed"):
                cache.build_irodori_speaker_references(
                    self.media_path,
                    _segments("ABC"),
                )

        self.worker_manifests.clear()
        patches = self._patch_generation()
        with patches[0], patches[1], patches[2]:
            resumed_cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                "irodori",
            )
            resumed_cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABC"),
            )

        self.assertEqual(
            [item["speaker_id"] for item in self.worker_manifests[0]["speakers"]],
            ["B", "C"],
        )

    def test_source_or_codec_change_invalidates_without_touching_other_caches(self) -> None:
        sentinel = self.cache_dir / "ovref_A.wav"
        self.cache_dir.mkdir(parents=True)
        sentinel.write_bytes(b"omnivoice-cache")

        patches = self._patch_generation()
        with patches[0], patches[1], patches[2]:
            cache = reference.SpeakerReferenceCache(self.cache_dir, "irodori")
            cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABC"),
            )

        original_stat = self.media_path.stat()
        self.media_path.write_bytes(b"changed-data")
        self.assertEqual(self.media_path.stat().st_size, original_stat.st_size)
        os.utime(
            self.media_path,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        self.worker_manifests.clear()
        patches = self._patch_generation()
        with patches[0], patches[1], patches[2]:
            changed_source_cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                "irodori",
            )
            changed_source_cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABC"),
            )
        self.assertEqual(len(self.worker_manifests[0]["speakers"]), 3)

        self.worker_manifests.clear()
        patches = self._patch_generation()
        with (
            patches[0],
            patches[1],
            patches[2],
            mock.patch.object(reference, "IRODORI_CODEC_REPO", "changed/codec"),
        ):
            changed_codec_cache = reference.SpeakerReferenceCache(
                self.cache_dir,
                "irodori",
            )
            changed_codec_cache.build_irodori_speaker_references(
                self.media_path,
                _segments("ABC"),
            )
        self.assertEqual(len(self.worker_manifests[0]["speakers"]), 3)
        self.assertEqual(sentinel.read_bytes(), b"omnivoice-cache")

    def test_reference_ranges_remain_chronological_and_capped(self) -> None:
        segments = [
            Segment(
                idx=index,
                start=float(index * 60),
                end=float(index * 60 + 50),
                text_src="line",
                speaker_id="A",
            )
            for index in range(3)
        ]
        ranges = reference._select_irodori_reference_ranges(
            segments,
            max_sec=120.0,
        )
        self.assertEqual(ranges, [(0.0, 50.0), (60.0, 110.0), (120.0, 140.0)])
        self.assertEqual(sum(end - start for start, end in ranges), 120.0)

    def test_non_irodori_reference_builders_never_start_latent_worker(self) -> None:
        with mock.patch.object(reference, "_run_irodori_reference_worker") as worker_mock:
            for engine in ("omnivoice", "voxcpm2"):
                cache = reference.SpeakerReferenceCache(
                    self.cache_dir / engine,
                    engine,
                )
                cache.build_omnivoice_references(self.media_path, [], [])
                cache.build_omnivoice_segment_references(self.media_path, [])

        worker_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
