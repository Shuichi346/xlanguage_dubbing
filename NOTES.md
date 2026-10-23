# Notes

## 2026-10-23

- MeanFlow requires updated inference code as well as new weights. Updated the separate Irodori server environment from Irodori-TTS commit `8ca3acb58ab4e19ad6d594aaed6bafe3e88f7f71` to `89f9d8fbd4d51ea019867ee1197725ede1df13c5` using `uv sync --extra cpu --upgrade-package irodori-tts --inexact`. The ignored server lock and installed package both resolved to the new commit. The official MF checkpoint resolved to Hub revision `ccc78f5d480b6e51b69b2d5042a14c4da04fea6e`.
- The active `.env`, example defaults, and generated `start_servers.sh` now select v4.1-Small-MF / four steps. `config.irodori_sampling_options()` omits RF-only Sway controls for the official MF checkpoint; explicitly selected RF checkpoints retain those controls. MeanFlow learns CFG during distillation, so cloning/design requests do not send CFG overrides. Startup clears `IRODORI_CHECKPOINT` so a server-local path cannot override the selected Hub model, sets the server step default, and rejects a healthy server reporting another Hub checkpoint without stopping external processes.
- Original-voice mode previously reused existence-only segment/TTS/retime caches and skipped an old final MP4. `tts/voice_design.py` exposes `irodori_synthesis_settings()` and `irodori_render_fingerprint()` for `core/pipeline.py`; these fingerprint effective Irodori settings for final output names and settings, translated lines, and reference-latent contents for original-mode render directories. ASR/translation paths remain reusable. Designed-voice namespaces use the same effective sampling settings, so changing inactive MF Sway values does not regenerate a voice. Existing old outputs are preserved.
- All 70 unit tests, Ruff, and ty passed. `test_irodori_tts.py` covers MF payloads, RF controls, launch exports, and refusal of an old running server; `test_voice_design_pipeline.py` also runs the pipeline against legacy MP4/TTS artifacts and verifies generation reaches the isolated render directory.
- Native CPU/FP32 verification completed caption-only design (5.76 seconds of 48 kHz mono audio), codec-only deterministic reference encoding (two independently encoded short utterances, 119 concatenated latent steps), and cloning through `irodori.ref_latent` (3.60 seconds of 48 kHz mono audio). Server logs confirmed `sample_meanflow` with four NFE. Cold design and cloning requests took about 14.4 and 8.0 seconds respectively; these are smoke-test timings, not a v4 speed comparison. The pipeline-owned servers were released after verification.
- Verification artifacts are under `temp/irodori_mf_verification/`, including `report.json`, `clone.wav`, and the unit-test log. Whisper large-v3-turbo with Silero VAD transcribed the cloned line as the requested Japanese sentence. Full-video rendering, cross-language similarity, and an old/new listening benchmark were not run.

## 2026-10-21

- Added `tts/voice_design.py` to create one target-language reference per detected speaker, preserving the fixed Japanese passage and its English translation. This avoids conditioning cross-language cloning on the original pronunciation; it does not fine-tune model weights or guarantee accent removal.
- Age/gender estimation uses the official `audeering/wav2vec2-large-robust-24-ft-age-gender` Hugging Face safetensors checkpoint at revision `bad2737f529168bc3d3fcd97edfaba9583c7d36d`, loaded on CPU with the provider's custom age/gender heads and audio normalization. Raw female/male/child logits require softmax exactly once, and normalized age is multiplied by 100. Adult ages below 40 map to young adult; the other supported adult group is middle-aged. Missing/uncertain attributes, children, invalid audio, and model failures use an adult male. The estimator is released before TTS loads. The provider's CC BY-NC-SA 4.0 model license requires separate commercial licensing for commercial use.
- Voice design is opt-in with `ENABLE_VOICE_DESIGN=true`. VoxCPM2 uses its parenthesized description syntax, Irodori uses Caption with no reference only for initial generation, and OmniVoice selects gender/age/moderate-pitch attributes. Cloning then receives the same designed reference for every line; original segment references are cleared. OmniVoice creates a reusable clone prompt per designed speaker and releases it with the model.
- Irodori v4 Small remains Japanese-only. After all designed WAVs exist, only a pipeline-owned design server is stopped, generated speech is split into short utterances, and the existing short-lived worker encodes 48 kHz mono FP32 clips with deterministic DACVAE / `normalize_db=-16.0`. Worker metadata must include `reference_latent` as well as fingerprint/hash/shape/dtype for validation. External servers remain externally managed.
- Designed checkpoints live beneath each separated/raw-audio and ASR directory's `voice_design/<engine>_<language>_<settings>/` path. Samples are hashed and atomically published individually; rendering has its own voice/translation fingerprint. Outputs include a voice-design marker. Valid fallback male voices deliberately persist after estimator installation; changing `VOICE_DESIGN_SEED` requests a new voice set.
- Regression coverage in `test_age_gender.py`, `test_voice_design_engines.py`, `test_voice_design_cache.py`, and `test_voice_design_pipeline.py` checks four-speaker reuse, uncertainty handling, cache repair, codec metadata, language restrictions, memory ordering, and both source-audio modes. All 58 unit tests, Ruff, ty, compilation, and the existing 18-case matrix dry-run passed. Native OmniVoice generated a 14.92-second Japanese reference and two separate lines at 24 kHz using one cached clone prompt. Local Whisper large-v3-turbo with Silero VAD transcribed the complete reference passage. TTS-specific reading errors are acceptable; no transcript-perfect regeneration loop was added.
- The initial Zenodo download did not respond. The follow-up fetched the user's specified official Hugging Face 24-layer model directly, verified its full SHA-256 against Hub metadata, and successfully inferred an eight-second real speech clip (male, estimated age about 58, gender confidence 0.986, no fallback). The official model card requires custom heads; loading only AutoModel would discard the trained age/gender predictions. `test_age_gender_torch.py` verifies safetensors loading, normalization, CPU FP32 inference, and rejection of missing heads using small local models.
- Persistent weights and source provenance live under `models/age-gender/hf24/`, outside `temp/`; `.gitignore` excludes all of `models/`, and no model files are tracked. Relative `AGE_GENDER_MODEL_DIR` paths resolve against the project root even when launched from a temporary working directory. Only generated reference WAVs/latents and verification artifacts use `temp/`. Optional audonnx supports explicit local ONNX exports or an existing six-layer cache; no ONNX weights were downloaded in this work and there is no automatic Zenodo or third-party mirror fallback.
- Verification reports are in `temp/voice_design_verification/`. Native VoxCPM2/Irodori generation and full video rendering were not rerun for the model-download follow-up. Tests explicitly select original/design modes independently of the user's `.env`; repaired FLOAT WAV comparisons use decoded samples because PEAK timestamps can change even when audio is identical.

## 2026-10-11

- Integrated the installed NeMo-Speech.cpp 0.2.0 CLI as `ASR_ENGINE=nemo-speech`, explicitly selecting `nvidia/nemotron-3.5-asr-streaming-0.6b` and `nvidia/Nemotron-3-Diarization`. Metal and CPU are configurable; no Python NeMo dependency or installation step was added.
- Converted JSON word times in seconds and 1-based speaker tags into bounded turns and reference intervals. Tags 1–8 map to `SPEAKER_00`–`SPEAKER_07`; out-of-range tags are rejected, and unassigned words are kept without inventing a speaker. Standalone zero-duration punctuation attaches to the preceding turn.
- Kept NeMo checkpoints in a `nemo-speech/` subdirectory within each existing separated/raw-audio work directory to avoid reusing another engine's segments, translations, or references. Reference transcription uses NeMo without loading the diarizer.
- Verified both requested cached models on Metal using a 25-second, two-speaker clip: the CLI returned timed speaker tags, the Python adapter produced nine turns, and reference transcription succeeded. All 31 unit tests, Ruff, ty, and the 18-case matrix dry-run passed. Full translation, TTS inference, and final video rendering were not rerun for this addition.

## 2026-09-10

- Reproduced FFmpeg 9.0.1 concat decoding failures (`blocksize 4064 > 1024`) with the cached Steve Jobs OmniVoice/raw-audio job. Silence and speech FLACs had compatible rates/channels but different STREAMINFO block sizes.
- Replaced compressed-packet concatenation with independent SoundFile decoders streaming PCM into one FLAC writer. Inputs remain normalized to the configured TTS rate/channels; 24-bit output preserves both 16- and 24-bit source precision.
- Replaced existence-only joined-audio cache checks with total-frame/rate/channel checks and atomic publication, allowing the existing truncated `dubbed_full.flac` to recover without rerunning TTS.
- Verified the concat fix with 17 tests, recovered all 948 cached audio parts, decoded the complete joined speech track without errors, and muxed a 5319.087-second H.264/AAC verification MP4 under the job’s `retime` directory using that run’s `.env`.
- Installed Ruff 0.16.6 and ty 0.0.80 from the project `dev` dependency group using `uv sync --only-group dev --inexact` to preserve runtime packages. The local `uv.lock` was refreshed and remained ignored under the existing repository policy.
- Addressed the initial 113 Ruff errors and 19 ty diagnostics by modernizing annotations/imports, typing pending Irodori reference records, narrowing optional values, and validating worker manifests. Both tools, compilation, and all 20 unit tests passed after this cleanup; full model inference was not rerun.
- Corrected the nonexistent TorchCodec `decode()` call to `get_all_samples()`. Native TorchCodec still failed to load in the installed environment; retained audio fallbacks and tested the primary API contract with a test double and SoundFile sample/channel preservation. Native TorchCodec decoding remains unverified.
- Kept one explained `ty: ignore[unresolved-import]` for the worker-only `irodori_tts.codec` import. Confirmed its source file existed in the separate Irodori server Python 3.10 environment; this was not a runtime import or model-inference test. Other type diagnostics were resolved without blanket rule suppression.

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
