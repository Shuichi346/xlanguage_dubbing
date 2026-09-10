import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing.audio.ffmpeg import (
    concat_audio_to_flac,
    create_silence_flac,
    encode_original_audio_chunk_flac,
)
from xlanguage_dubbing.config import TTS_CHANNELS, TTS_SAMPLE_RATE
from xlanguage_dubbing.utils import PipelineError


class EncodeOriginalAudioChunkFlacTests(unittest.TestCase):
    @mock.patch("xlanguage_dubbing.audio.ffmpeg.run_cmd")
    @mock.patch("xlanguage_dubbing.audio.ffmpeg.which_or_raise")
    def test_buffers_tiny_initial_frame_without_padding(
        self,
        _which_or_raise: mock.Mock,
        run_cmd: mock.Mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_flac = Path(temp_dir) / "chunk.flac"
            encode_original_audio_chunk_flac(
                Path("input.mp4"),
                out_flac,
                start=1533.759714,
                end=1548.160000,
                speed=0.978280,
            )

        command = run_cmd.call_args.args[0]
        audio_filter = command[command.index("-af") + 1]
        self.assertIn("aresample=async=1:first_pts=0", audio_filter)
        self.assertTrue(audio_filter.endswith("asetnsamples=n=4096:p=0"))


class ConcatAudioTests(unittest.TestCase):
    def test_mixed_flac_block_sizes_preserve_samples_and_repair_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            silence = root / "silence.flac"
            speech = root / "speech's audio.flac"
            output = root / "joined.flac"
            create_silence_flac(silence, 0.2)
            samples = np.random.default_rng(42).integers(
                -20000, 20000, size=(12001, TTS_CHANNELS), dtype=np.int16,
            )
            sf.write(str(speech), samples, TTS_SAMPLE_RATE, subtype="PCM_16")
            # Simulate the valid header and incomplete track left by old concat.
            sf.write(str(output), samples[:100], TTS_SAMPLE_RATE, subtype="PCM_16")
            concat_audio_to_flac([silence, speech, silence], output, root / "list.txt")
            actual, rate = sf.read(str(output), dtype="int16", always_2d=True)
            quiet, _ = sf.read(str(silence), dtype="int16", always_2d=True)
            np.testing.assert_array_equal(actual, np.concatenate([quiet, samples, quiet]))
            self.assertEqual(rate, TTS_SAMPLE_RATE)
            mtime = output.stat().st_mtime_ns
            concat_audio_to_flac([silence, speech, silence], output, root / "list.txt")
            self.assertEqual(output.stat().st_mtime_ns, mtime)

    def test_failed_decode_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.flac"
            output = root / "joined.flac"
            sf.write(str(source), np.zeros((100, TTS_CHANNELS)), TTS_SAMPLE_RATE)
            with mock.patch.object(sf.SoundFile, "blocks", side_effect=RuntimeError("decode")):
                with self.assertRaisesRegex(RuntimeError, "decode"):
                    concat_audio_to_flac([source], output, root / "list.txt")
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".concat-*")))

    def test_rejects_mismatched_sample_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.flac"
            sf.write(str(source), np.zeros((100, TTS_CHANNELS)), TTS_SAMPLE_RATE // 2)
            with self.assertRaises(PipelineError):
                concat_audio_to_flac([source], root / "joined.flac", root / "list.txt")


if __name__ == "__main__":
    unittest.main()
