import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

from xlanguage_dubbing.diarization.speaker import _load_audio_waveform


class AudioWaveformTests(unittest.TestCase):
    def test_torchcodec_uses_get_all_samples_without_fallback(self):
        waveform = torch.zeros((2, 32), dtype=torch.float32)
        decoder = mock.Mock(spec=["get_all_samples"])
        decoder.get_all_samples.return_value = types.SimpleNamespace(
            data=waveform, sample_rate=48000,
        )
        constructor = mock.Mock(return_value=decoder)
        decoder_module = types.ModuleType("torchcodec.decoders")
        decoder_module.__dict__["AudioDecoder"] = constructor
        with mock.patch.dict(sys.modules, {"torchcodec.decoders": decoder_module}):
            actual, rate = _load_audio_waveform(Path("input.wav"))
        constructor.assert_called_once_with("input.wav")
        decoder.get_all_samples.assert_called_once_with()
        self.assertIs(actual, waveform)
        self.assertEqual(rate, 48000)

    def test_soundfile_fallback_preserves_channels_and_samples(self):
        decoder_module = types.ModuleType("torchcodec.decoders")
        decoder_module.__dict__["AudioDecoder"] = mock.Mock(side_effect=RuntimeError("native"))
        torchaudio_module = types.ModuleType("torchaudio")
        torchaudio_module.__dict__["load"] = mock.Mock(side_effect=RuntimeError("backend"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.wav"
            samples = np.array([[0.125, -0.25], [0.5, -0.5]], dtype=np.float32)
            sf.write(str(path), samples, 16000, subtype="FLOAT")
            with mock.patch.dict(sys.modules, {
                "torchcodec.decoders": decoder_module,
                "torchaudio": torchaudio_module,
            }):
                actual, rate = _load_audio_waveform(path)
        np.testing.assert_array_equal(actual.numpy(), samples.T)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(rate, 16000)
