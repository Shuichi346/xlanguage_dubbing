import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xlanguage_dubbing.audio.ffmpeg import encode_original_audio_chunk_flac


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


if __name__ == "__main__":
    unittest.main()
