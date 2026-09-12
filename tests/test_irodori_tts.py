from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xlanguage_dubbing import irodori_tts
from xlanguage_dubbing.core import pipeline
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.servers import health
from xlanguage_dubbing.tts.reference import SpeakerReferenceCache


class _AudioResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b"RIFF" + (b"\0" * 256)


class _Progress:
    def __init__(self) -> None:
        self.steps = []
        self.artifacts = []

    def set_step(self, *args) -> None:
        self.steps.append(args)

    def set_artifact(self, *args) -> None:
        self.artifacts.append(args)

    def save(self) -> None:
        pass


class _ReferenceCache(SpeakerReferenceCache):
    def __init__(self, root: Path) -> None:
        super().__init__(root, "irodori")
        self.paths = {}
        for speaker in "ABC":
            path = root / f"{speaker}.pt"
            path.write_bytes(b"latent")
            self.paths[speaker] = path.resolve()

    def get_irodori_speaker_reference_latent_path(self, speaker_id: str):
        return self.paths.get(speaker_id)

    def get_irodori_speaker_reference_duration(self, speaker_id: str) -> float:
        return 120.0


def _segments(speakers: str) -> list[Segment]:
    return [
        Segment(
            idx=index,
            start=float(index - 1),
            end=float(index),
            text_src=f"source {index}",
            text_tgt=f"セリフ {index}",
            speaker_id=speaker,
        )
        for index, speaker in enumerate(speakers, start=1)
    ]


class IrodoriTtsTests(unittest.TestCase):
    def test_segment_generation_passes_the_cached_speaker_latent(self) -> None:
        synthesize_calls = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference_cache = _ReferenceCache(root)

            def fake_synthesize(**kwargs):
                synthesize_calls.append(kwargs)
                kwargs["out_audio"].write_bytes(b"RIFF" + (b"\0" * 256))

            def fake_convert(_input_path, output_path):
                output_path.write_bytes(b"flac")

            with (
                mock.patch.object(
                    irodori_tts,
                    "irodori_tts_synthesize",
                    side_effect=fake_synthesize,
                ),
                mock.patch.object(
                    irodori_tts,
                    "_convert_to_flac",
                    side_effect=fake_convert,
                ),
                mock.patch.object(
                    irodori_tts,
                    "ffprobe_duration_sec",
                    return_value=1.0,
                ),
            ):
                for segno, speaker in enumerate("AAB", start=1):
                    irodori_tts.generate_segment_tts_irodori(
                        Segment(
                            idx=segno,
                            start=float(segno - 1),
                            end=float(segno),
                            text_src="source",
                            text_tgt="セリフ",
                            speaker_id=speaker,
                        ),
                        root / f"seg_{segno:05d}",
                        reference_cache,
                        segno=segno,
                    )

        self.assertEqual(
            [call["ref_latent_path"] for call in synthesize_calls],
            [
                reference_cache.paths["A"],
                reference_cache.paths["A"],
                reference_cache.paths["B"],
            ],
        )

    def test_generated_server_script_uses_the_same_fp32_codec_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "start_servers.sh"
            with (
                mock.patch.object(health, "_is_irodori_tts", return_value=True),
                mock.patch.object(
                    health,
                    "IRODORI_CODEC_REPO",
                    "Aratako/Semantic-DACVAE-Japanese-32dim",
                ),
            ):
                health.generate_start_script(output_path)

            script = output_path.read_text(encoding="utf-8")

        self.assertIn(
            "export IRODORI_CODEC_REPO=Aratako/Semantic-DACVAE-Japanese-32dim",
            script,
        )
        self.assertIn("export IRODORI_CODEC_PRECISION=fp32", script)
        self.assertIn("export IRODORI_CODEC_DETERMINISTIC_ENCODE=true", script)

    def test_speech_payload_uses_ref_latent_exclusively(self) -> None:
        captured_requests = []

        def fake_urlopen(request, *, timeout):
            captured_requests.append((request, timeout))
            return _AudioResponse()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_path = root / "output.wav"
            latent_path = root / "speaker.pt"
            latent_path.write_bytes(b"latent")

            with (
                mock.patch.object(irodori_tts, "ensure_irodori_tts_server"),
                mock.patch.object(
                    irodori_tts.urllib.request,
                    "urlopen",
                    side_effect=fake_urlopen,
                ),
                mock.patch.object(irodori_tts, "IRODORI_TTS_NUM_STEPS", 8),
                mock.patch.object(
                    irodori_tts,
                    "IRODORI_TTS_T_SCHEDULE_MODE",
                    "sway",
                ),
                mock.patch.object(irodori_tts, "IRODORI_TTS_SWAY_COEFF", -1.0),
            ):
                irodori_tts.irodori_tts_synthesize(
                    text="テストです。",
                    out_audio=output_path,
                    ref_latent_path=latent_path,
                )

        self.assertEqual(len(captured_requests), 1)
        payload = json.loads(captured_requests[0][0].data.decode("utf-8"))
        options = payload["irodori"]
        self.assertEqual(options["ref_latent"], str(latent_path))
        for forbidden_key in (
            "ref_wav",
            "ref_wavs",
            "ref_latents",
            "caption",
            "style_prompt",
            "seconds",
        ):
            self.assertNotIn(forbidden_key, options)
            self.assertNotIn(forbidden_key, payload)
        self.assertEqual(options["num_steps"], 8)
        self.assertEqual(options["t_schedule_mode"], "sway")
        self.assertEqual(options["sway_coeff"], -1.0)

    def test_ababcabc_generation_order_and_speaker_latent_reuse(self) -> None:
        calls = []

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            segment_audio_dir = root / "seg_audio"
            segment_audio_dir.mkdir()
            work_dir = root / "work"
            work_dir.mkdir()
            reference_cache = _ReferenceCache(root)

            def fake_generate(seg, output_stub, cache, *, segno):
                calls.append(
                    (
                        segno,
                        seg.speaker_id,
                        output_stub.name,
                        cache.get_irodori_speaker_reference_latent_path(
                            seg.speaker_id
                        ),
                    )
                )
                return None

            with (
                mock.patch.object(irodori_tts, "ensure_irodori_tts_server"),
                mock.patch.object(
                    irodori_tts,
                    "generate_segment_tts_irodori",
                    side_effect=fake_generate,
                ),
            ):
                pipeline._run_tts_irodori(
                    _segments("ABABCABC"),
                    segment_audio_dir,
                    work_dir,
                    _Progress(),
                    reference_cache,
                )

        self.assertEqual([item[1] for item in calls], list("ABABCABC"))
        self.assertEqual([item[0] for item in calls], list(range(1, 9)))
        self.assertEqual(
            [item[2] for item in calls],
            [f"seg_{index:05d}" for index in range(1, 9)],
        )
        for speaker in "ABC":
            paths = {item[3] for item in calls if item[1] == speaker}
            self.assertEqual(paths, {reference_cache.paths[speaker]})


if __name__ == "__main__":
    unittest.main()
