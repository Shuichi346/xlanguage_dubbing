# Changelog

## Fixed

- Corrected the TorchCodec audio loader to use `get_all_samples()` instead of the nonexistent `decode()` method, preserving existing fallback loaders.
- Rejected malformed Irodori worker manifests with a pipeline error before subprocess launch.
- Fixed final audio concatenation failing on mixed silence/TTS FLAC block sizes by decoding each chunk independently with bounded memory. Incomplete joined tracks are rebuilt on resume, and replacements are published atomically.
- Fixed CAT-Translate-7b GGUF inference adding a duplicate BOS token, which could reduce translation quality and cause wrong-language or prompt-contaminated output.
- Stored VoxCPM2 reference metadata and segment clips under `voxcpm2_*` names instead of `omnivoice_*` names.

## Added

- Added `ASR_ENGINE=nemo-speech` for joint NeMo-Speech.cpp transcription and speaker diarization with Nemotron 3.5 ASR and Nemotron 3 Diarization, supporting up to eight speakers.
- Added configurable NeMo CLI/model/device settings, NeMo reference-clip transcription, isolated NeMo checkpoints within each audio mode, and six NeMo cases in the 18-case configuration matrix.
- Added Ruff and ty as project development dependencies with documented quick lint/type checks.
- Added optional Demucs voice/background separation before ASR and TTS reference extraction.
- Added `ENABLE_AUDIO_SEPARATION`, `DEMUCS_MODEL`, and `DEMUCS_DEVICE` environment settings; disabled mode keeps the previous raw-audio workflow.
- Added `TTS_ENGINE=irodori` for Japanese Irodori-TTS-v4-Small voice cloning through Irodori-TTS-Server.
- Added `IRODORI_MODEL_DEVICE` and `IRODORI_CODEC_DEVICE` settings, both defaulting to `cpu`.
- Added `IRODORI_CODEC_REPO` so reference precomputation and server decoding use the same DACVAE codec.
- Added Irodori Sway Sampling settings: `IRODORI_TTS_NUM_STEPS`, `IRODORI_TTS_T_SCHEDULE_MODE`, and `IRODORI_TTS_SWAY_COEFF`.
- Added `IRODORI_REFERENCE_MAX_SEC` for Irodori long-reference construction.
- Added `scripts/run_config_matrix.py` for developer verification across all supported `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, and `TTS_ENGINE` combinations on `input_videos/test.mp4`.

## Changed

- Updated the default Irodori checkpoint to `Aratako/Irodori-TTS-v4-Small` and preserved the `8`-step Sway Sampling profile.
- Changed Irodori cloning to encode each selected short utterance in a short-lived server-environment process, concatenate up to 120 seconds of latents, and atomically cache one validated `.pt` tensor per speaker.
- Changed the default Demucs model from `htdemucs_ft` to faster `htdemucs`.
- Changed `TTS_ENGINE=voxcpm2` synthesis from Ultimate Cloning to Controllable Cloning by using per-segment `reference_wav_path` without prompt audio/text.
- Separated background audio is now mixed at full volume; `ORIGINAL_VOLUME` only attenuates raw original audio when separation is disabled.
- Replaced supported `TTS_ENGINE=kokoro-fastapi` selection with `TTS_ENGINE=irodori`.
- Irodori mode now reuses the speaker-level cache via `irodori.ref_latent`, eliminating repeated DACVAE encoding in each API request. It never combines `ref_latent` with `ref_wav`; Sway Sampling remains enabled while caption/style prompt and fixed `seconds` stay omitted.

## [9.0.2] - 2026-04-16

### Fixed

- Fixed all OmniVoice TTS segments failing with `'numpy.ndarray' object has no attribute 'detach'`.
  - `model.generate()` returned `list[np.ndarray]`, while the old code assumed `torch.Tensor` and called `.detach().cpu().float().numpy()`.
  - Added a type-safe `_to_numpy()` helper for `torch.Tensor`, `np.ndarray`, and other inputs.
  - Replaced `torch.cat` with `np.concatenate` when joining waveforms.
  - Read the sample rate dynamically from the model's `sampling_rate` attribute.

## [9.0.1] - 2026-04-14

### Fixed

- Fixed all VoxCPM2 TTS segments failing.
  - VoxCPM v2.0.0's constructor did not accept `device`, causing `VoxCPM.__init__() got an unexpected keyword argument 'device'`.
  - Raised the minimum version to `voxcpm>=2.0.2`, which added `device` support.
  - Added a fallback for compatibility with older versions.
- Read the sample rate dynamically from the model object.
- Standardized the Python requirement on `>=3.13`.
  - `audioop-lts>=0.2.1` required Python 3.13 or later; Python 3.12 still included the standard-library `audioop`.
  - Added the required `audioop-lts` dependency for pydub's internal `audioop` use on Python 3.13.

### Changed

- Updated the version from 9.0.0 to 9.0.1.

## [9.0.0] - 2026-04-14

### Added

- Added the VoxCPM2 TTS engine with a 2B model, 30-language support, and native 48 kHz output.
  - Used Ultimate Cloning with reference audio and text.
  - Supported MPS and CPU on Apple Silicon Macs.
- Added `TTS_ENGINE` to select `omnivoice` or `voxcpm2`.
- Added `src/xlanguage_dubbing/voxcpm2_tts.py`.
- Added VoxCPM2 settings to `.env.example` and `config.py`.

### Changed

- Extended `pipeline.py` to switch TTS engines.
- Added TTS engine display to `cli.py`.
- Added `voxcpm` to the dependencies in `pyproject.toml`.
- Updated the version from 8.0.0 to 9.0.0.

## [8.0.0] - 2026-04-09

### Added

- Added multilingual audio dubbing.
  - Added `INPUT_LANG`, defaulting to `auto`, for ASR language detection.
  - Added `OUTPUT_LANG` to select the output language explicitly.
  - Added the `detected_lang` field to each segment.
- Added the TranslateGemma-12b-it GGUF translation engine with 55-language support.
  - Selected it automatically for language pairs other than Japanese-English or English-Japanese.
  - Retained CAT-Translate for Japanese-English and English-Japanese translation.
- Added `src/xlanguage_dubbing/lang_utils.py` for language detection and language-code conversion.
- Added `src/xlanguage_dubbing/translation/translategemma.py`.

### Changed

- Renamed the project from `ja-dubbing` to `xlanguage-dubbing`.
  - Renamed the package from `ja_dubbing` to `xlanguage_dubbing`.
  - Renamed the CLI command from `ja-dubbing` to `xlanguage-dubbing`.
  - Changed the output suffix from `_jaDub.mp4` to `_xlDub.mp4`.
- Changed the default ASR engine from `whisper` to `vibevoice` for code-switching and multilingual audio.
- Generalized volume settings: `ENGLISH_VOLUME` became `ORIGINAL_VOLUME`, and `JAPANESE_VOLUME` became `DUBBED_VOLUME`.
- Linked whisper.cpp's language setting to `INPUT_LANG`, including `auto` support.
- Added `detected_lang` to segment JSON.
- Updated the version from 7.0.0 to 8.0.0.

## [7.0.0] - 2026-04-09

### Changed

- Consolidated TTS on OmniVoice.
- Updated the version from 6.0.0 to 7.0.0.

### Removed

- Removed Kokoro TTS support completely.

## [6.0.0]

### Added

- Added dual OmniVoice and Kokoro TTS engines.
- Added local translation with CAT-Translate-7b.
- Added VibeVoice-ASR support.
