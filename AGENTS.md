# AGENTS.md

Project instructions for coding agents working in this repository.

- Preserve the `ENABLE_AUDIO_SEPARATION` fallback: when it is `false`, the pipeline must use the original media audio for ASR, reference extraction, and final background mixing.
- Keep audio-separation and raw-audio temporary outputs isolated. The raw-audio mode uses `temp/<video>_rawaudio`; separated mode uses `temp/<video>`.
- Apply `ORIGINAL_VOLUME` only to raw original audio. When `ENABLE_AUDIO_SEPARATION=true`, mix the separated background stem at full volume unless a new explicit background-volume setting is added.
- Do not remove the Demucs `--two-stems vocals` contract unless the pipeline is updated to consume a different voice/background stem layout.
- Keep Demucs model/device configurable through `DEMUCS_MODEL` and `DEMUCS_DEVICE`; do not hard-code `cpu` or `mps` in the separation command.
- Keep `scripts/run_config_matrix.py` aligned with supported values whenever `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, or `TTS_ENGINE` options change.
- Keep VoxCPM2 in Controllable Cloning mode: pass per-segment `reference_wav_path` only for synthesis, and keep VoxCPM2 reference cache artifacts under `voxcpm2_*` names.
- Keep Irodori-TTS-v4-Small in server API mode: use `Aratako/Irodori-TTS-v4-Small`, pass each speaker's cached reference latent as `irodori.ref_latent`, keep the `8`-step Sway Sampling profile, do not send Caption / Style Prompt, and do not set fixed `seconds`.
- Generate Irodori reference latents in a short-lived process using the Irodori-TTS-Server environment before starting the long-running TTS server. Load only the DACVAE codec, encode each selected short utterance separately as 48 kHz mono FP32 with deterministic encoding and `normalize_db=-16.0`, then concatenate the latents in utterance order.
- Limit Irodori reference latents to 120 seconds, compose them from multiple short utterances by the same speaker, and keep one validated, atomically-written `.pt` cache per speaker under `speaker_refs`.
- Keep the CAT-Translate-7b prompt aligned with CyberAgent's 7B-specific chat template. Pass the complete prompt as tokens created with `add_bos=False` and `special=True` because the template already contains `<s>`.
- Keep the TranslateGemma text prompt aligned with the structured template embedded in its GGUF: preserve source/target language codes, three newlines before source text, and Gemma turn markers. Its prompt omits a textual BOS because `llama-cpp-python` prepends one.

- Concatenate silence and TTS FLAC chunks through independent decoders; their STREAMINFO block sizes may differ. Validate joined-track sample counts before cache reuse and publish completed audio atomically.

- Install project lint/type tools with `uv sync --only-group dev --inexact`. Run `uv run --no-sync ruff check src tests scripts` and `uv run --no-sync ty check` for quick checks after Python edits; report existing diagnostics rather than silently suppressing them.
