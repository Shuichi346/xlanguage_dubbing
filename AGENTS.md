# AGENTS.md

Project instructions for coding agents working in this repository.

- Preserve the `ENABLE_AUDIO_SEPARATION` fallback: when it is `false`, the pipeline must use the original media audio for ASR, reference extraction, and final background mixing.
- Keep audio-separation and raw-audio temporary outputs isolated. The raw-audio mode uses `temp/<video>_rawaudio`; separated mode uses `temp/<video>`.
- Apply `ORIGINAL_VOLUME` only to raw original audio. When `ENABLE_AUDIO_SEPARATION=true`, mix the separated background stem at full volume unless a new explicit background-volume setting is added.
- Do not remove the Demucs `--two-stems vocals` contract unless the pipeline is updated to consume a different voice/background stem layout.
- Preserve Kokoro-FastAPI Japanese-mode safeguards: launch the server without the parent `VIRTUAL_ENV`, keep warmup/request `lang_code` aligned with `KOKORO_FASTAPI_VOICE`, and do not route Japanese chunk sizing through the English eSpeak phonemizer.
- Keep `scripts/run_config_matrix.py` aligned with supported values whenever `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, or `TTS_ENGINE` options change.
- Keep VoxCPM2 in Controllable Cloning mode: pass per-segment `reference_wav_path` only for synthesis, and keep VoxCPM2 reference cache artifacts under `voxcpm2_*` names.
