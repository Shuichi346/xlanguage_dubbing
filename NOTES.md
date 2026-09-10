# Notes

## 2026-09-10

- Installed Ruff 0.16.6 and ty 0.0.80 from the project `dev` dependency group using `uv sync --only-group dev --inexact` to preserve runtime packages. Initial checks reported 113 Ruff errors and 19 ty diagnostics; broad code cleanup remains separate from tool setup. The local `uv.lock` was refreshed and remains ignored under the existing repository policy.

- Validation passed all 17 tests, recovered all 948 cached audio parts, decoded the complete joined speech track without errors, and muxed a 5319.087-second H.264/AAC verification MP4 under the job’s `retime` directory using the current `.env`.

- Reproduced FFmpeg 9.0.1 concat decoding failures (`blocksize 4064 > 1024`) with the cached Steve Jobs OmniVoice/raw-audio job. Silence and speech FLACs had compatible rates/channels but different STREAMINFO block sizes.
- Replaced compressed-packet concatenation with independent SoundFile decoders streaming PCM into one FLAC writer. Inputs remain normalized to the configured TTS rate/channels; 24-bit output preserves both 16- and 24-bit source precision.
- Replaced existence-only joined-audio cache checks with total-frame/rate/channel checks and atomic publication, allowing the existing truncated `dubbed_full.flac` to recover without rerunning TTS.

## 2026-08-07

- Confirmed from the Irodori v4 model and server implementations that multiple short utterances should be encoded separately, concatenated in order, and capped at 120 seconds (3000 DACVAE steps at 48 kHz).
- Moved Irodori reference encoding into a short-lived Irodori-TTS-Server process and cached one raw FP32 tensor per speaker; API requests now use `irodori.ref_latent`, while OmniVoice and VoxCPM2 behavior remains unchanged.
- Added cache fingerprints and tensor/hash validation so source audio, selected ranges, codec settings, or cache corruption trigger atomic regeneration of only the affected speaker latents.
- Did not add DeepFilterNet3 because DeepFilterNet 0.5.6 requires `numpy<2` while pyannote-audio 4.x requires `numpy>=2`; the project dependency set cannot resolve both safely.

## 2026-08-05

- Updated Irodori-TTS-Server auto-start and generated launch scripts to use `Aratako/Irodori-TTS-v4-Small` with `uv run --no-sync`.
- Preserved per-segment `irodori.ref_wav` requests and the low-latency `num_steps=8`, `t_schedule_mode=sway`, `sway_coeff=-1.0` sampling profile.
- Resolved the actionable Ruff findings in VibeVoice type annotations, spaCy sentence allocation, CAT-Translate imports, and subprocess output capture; targeted lint, compilation, and behavior checks passed.

## 2026-06-13

- Confirmed that the CAT-Translate-7b instruction and chat wrapper match CyberAgent's official 7B template, then fixed the GGUF completion path to pass pre-tokenized input so `llama-cpp-python` does not prepend a second BOS token.
- Confirmed that the TranslateGemma-12b-it prompt matches Google's structured text-translation template embedded in the GGUF, including its language codes, three newlines before source text, turn markers, and single automatically prepended BOS token; no prompt change was needed.
- TranslateGemma Q6 inference failed with `llama_decode returned -3` when all layers were offloaded under current Metal memory pressure, while a 32-layer offload completed successfully.

## 2026-06-08

- Added Irodori-TTS-Server Sway Sampling request options with defaults: `num_steps=8`, `t_schedule_mode=sway`, and `sway_coeff=-1.0`.
- Changed the default Demucs model to `htdemucs` and added `DEMUCS_DEVICE` so Apple Silicon runs can use `mps` while retaining `cpu` as an explicit fallback.

## 2026-06-07

- Changed VoxCPM2 to Controllable Cloning: generation uses per-segment `reference_wav_path` only, while speaker/segment reference cache artifacts use engine-specific `voxcpm2_*` names.
- Replaced supported Kokoro-FastAPI selection with `TTS_ENGINE=irodori`.
- Irodori uses the local `Irodori-TTS-Server` API, sends per-segment reference audio as `irodori.ref_wav`, and intentionally omits Caption / Style Prompt and fixed `seconds`.
- Added `IRODORI_MODEL_DEVICE` and `IRODORI_CODEC_DEVICE` for Irodori server auto-start; both default to `cpu`.

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
