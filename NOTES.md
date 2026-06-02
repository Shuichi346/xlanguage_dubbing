# Notes

## 2026-06-02

- Added a configuration-matrix runner that isolates each case under `temp/config_matrix/`, symlinks the fixed sample video into a per-case input directory, and logs every ASR/audio-separation/TTS combination separately.

## 2026-06-01

- Reintroduced Kokoro-FastAPI as `TTS_ENGINE=kokoro-fastapi` for speed-priority English to Japanese dubbing with fixed voice `jf_alpha`.
- Kokoro-FastAPI mode starts/reuses the local API server, calls `/v1/audio/speech`, and skips pyannote speaker identification plus TTS reference extraction because no voice cloning is performed.
- Fixed Kokoro-FastAPI startup on Apple Silicon by launching it without the parent project `VIRTUAL_ENV`, setting the server default voice to `jf_alpha`, and passing `lang_code=j` for synthesis requests.
- Patched the local `Kokoro-FastAPI` checkout so non-English chunk sizing does not call the English eSpeak phonemizer path that fails on the packaged `espeak-ng-data/phontab` lookup.

## 2026-05-26

- Added Demucs `htdemucs_ft` separation so ASR, diarization, and TTS references can use the `vocals.wav` stem while final mixing uses `no_vocals.wav` as the background bed.
- Added `ENABLE_AUDIO_SEPARATION=false` as a fallback path that uses raw source audio and writes to `temp/<video>_rawaudio` to avoid reusing separated-audio checkpoints.
- Kept `ORIGINAL_VOLUME` only for raw-audio mode; separated `no_vocals.wav` background audio is mixed at `1.00`.
