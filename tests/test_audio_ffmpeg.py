import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from xlanguage_dubbing.audio.ffmpeg import (
    _run_atomic_audio,
    _run_atomic_media,
    concat_audio_to_flac,
    concat_ts_files,
    create_silence_flac,
    encode_original_audio_chunk_flac,
    encode_video_chunk_ts,
    ffprobe_duration_sec,
    mux_retimed_video_with_tracks,
    remux_ts_to_mp4,
    validate_audio_file,
    validate_media_file,
)
from xlanguage_dubbing.config import TTS_CHANNELS, TTS_SAMPLE_RATE
from xlanguage_dubbing.utils import PipelineError


class EncodeOriginalAudioChunkFlacTests(unittest.TestCase):
    @mock.patch("xlanguage_dubbing.audio.ffmpeg._run_atomic_audio")
    @mock.patch("xlanguage_dubbing.audio.ffmpeg.which_or_raise")
    def test_buffers_tiny_initial_frame_without_padding(
        self,
        _which_or_raise: mock.Mock,
        run_atomic_audio: mock.Mock,
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

        command = run_atomic_audio.call_args.args[0]
        audio_filter = command[command.index("-af") + 1]
        self.assertIn(f"aresample={TTS_SAMPLE_RATE}:async=1:first_pts=0", audio_filter)
        self.assertIn("apad=whole_len=", audio_filter)
        self.assertIn("atrim=end_sample=", audio_filter)
        self.assertTrue(audio_filter.endswith("asetnsamples=n=4096:p=0"))


class AtomicPublicationTests(unittest.TestCase):
    def test_failed_media_validation_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.mp4"
            output.write_bytes(b"previous completed video")

            def fake_encode(command):
                Path(command[-1]).write_bytes(b"interrupted new video")

            with (
                mock.patch("xlanguage_dubbing.audio.ffmpeg.run_cmd", side_effect=fake_encode),
                mock.patch("xlanguage_dubbing.audio.ffmpeg.validate_media_file", return_value=False),
                self.assertRaises(PipelineError),
            ):
                _run_atomic_media(["ffmpeg", str(output)], output, video=True)
            self.assertEqual(output.read_bytes(), b"previous completed video")
            self.assertFalse(list(output.parent.glob(".media-*")))

    def test_failed_audio_encode_does_not_publish_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.flac"

            def fake_encode(command):
                Path(command[-1]).write_bytes(b"partial audio")
                raise PipelineError("interrupted")

            with (
                mock.patch("xlanguage_dubbing.audio.ffmpeg.run_cmd", side_effect=fake_encode),
                self.assertRaisesRegex(PipelineError, "interrupted"),
            ):
                _run_atomic_audio(["ffmpeg", str(output)], output, 100)
            self.assertFalse(output.exists())
            self.assertFalse(list(output.parent.glob(".audio-*")))


class NativeTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.source = cls.root / "source.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=160x90:rate=30:duration=3", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=48000:duration=3", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.source),
        ], check=True, capture_output=True, timeout=30)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_ten_chunks_follow_cumulative_frames_and_samples(self):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            root = Path(directory)
            videos = []
            background = []
            elapsed = 0.0
            frames_used = 0
            samples_used = 0
            with mock.patch("xlanguage_dubbing.audio.ffmpeg.OUTPUT_SIZE", 96):
                for index in range(10):
                    elapsed += 2 / 3
                    frames = round(elapsed * 30) - frames_used
                    samples = round(elapsed * TTS_SAMPLE_RATE) - samples_used
                    video = root / f"video_{index}.ts"
                    audio = root / f"audio_{index}.flac"
                    encode_video_chunk_ts(
                        self.source, video, start=0.25, end=1.25, speed=1.5,
                        target_frames=frames, frame_rate=30,
                    )
                    if index % 2:
                        create_silence_flac(audio, samples / TTS_SAMPLE_RATE, target_samples=samples)
                    else:
                        encode_original_audio_chunk_flac(
                            self.source, audio, start=0.25, end=1.25, speed=1.5,
                            target_samples=samples,
                        )
                    self.assertTrue(validate_audio_file(audio, expected_samples=samples))
                    videos.append(video)
                    background.append(audio)
                    frames_used += frames
                    samples_used += samples
            joined_ts = root / "joined.ts"
            joined_video = root / "joined.mp4"
            joined_audio = root / "joined.flac"
            final = root / "final.mp4"
            concat_ts_files(videos, joined_ts, root / "video.txt")
            remux_ts_to_mp4(joined_ts, joined_video)
            concat_audio_to_flac(background, joined_audio, root / "audio.txt")
            self.assertEqual(sf.info(joined_audio).frames, 320000)
            self.assertLessEqual(abs(ffprobe_duration_sec(joined_video) - 20 / 3), 1 / 30)
            probe = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "json", str(joined_video),
            ], check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"]), 200)
            mux_retimed_video_with_tracks(
                joined_video, joined_audio, final, original_flac=joined_audio, original_volume=0.1,
            )
            self.assertTrue(validate_media_file(
                final, require_video=True, require_audio=True,
                expected_duration=20 / 3, duration_tolerance=1 / 30,
            ))

    def test_short_slow_background_has_exact_samples(self):
        output = self.root / "slow.flac"
        expected = round(0.2 / 0.3 * TTS_SAMPLE_RATE)
        encode_original_audio_chunk_flac(
            self.source, output, start=0.25, end=0.45, speed=0.3, target_samples=expected,
        )
        self.assertTrue(validate_audio_file(output, expected_samples=expected))

    def test_zero_width_turn_freezes_last_frame_and_silences_background(self):
        video = self.root / "frozen.ts"
        audio = self.root / "frozen.flac"
        with mock.patch("xlanguage_dubbing.audio.ffmpeg.OUTPUT_SIZE", 96):
            encode_video_chunk_ts(
                self.source, video, start=3, end=3, speed=0,
                target_frames=15, frame_rate=30,
            )
        encode_original_audio_chunk_flac(
            self.source, audio, start=3, end=3, speed=0,
            target_samples=TTS_SAMPLE_RATE // 2,
        )
        data, _ = sf.read(audio)
        self.assertFalse(np.any(data))
        self.assertTrue(validate_media_file(
            video, require_video=True, expected_duration=0.5, duration_tolerance=1 / 30,
        ))


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
