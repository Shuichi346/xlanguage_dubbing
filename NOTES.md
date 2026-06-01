# Notes

## 2026-05-26

- Added Demucs `htdemucs_ft` separation so ASR, diarization, and TTS references can use the `vocals.wav` stem while final mixing uses `no_vocals.wav` as the background bed.
- Added `ENABLE_AUDIO_SEPARATION=false` as a fallback path that uses raw source audio and writes to `temp/<video>_rawaudio` to avoid reusing separated-audio checkpoints.
- Kept `ORIGINAL_VOLUME` only for raw-audio mode; separated `no_vocals.wav` background audio is mixed at `1.00`.

## 2026-06-01

- Reintroduced Kokoro-FastAPI as `TTS_ENGINE=kokoro-fastapi` for speed-priority English to Japanese dubbing with fixed voice `jf_alpha`.
- Kokoro-FastAPI mode starts/reuses the local API server, calls `/v1/audio/speech`, and skips pyannote speaker identification plus TTS reference extraction because no voice cloning is performed.
