# AGENTS.md

Project instructions for coding agents working in this repository.

- Preserve the `ENABLE_AUDIO_SEPARATION` fallback: when it is `false`, the pipeline must use the original media audio for ASR, reference extraction, and final background mixing.
- Keep audio-separation and raw-audio temporary outputs isolated. The raw-audio mode uses `temp/<video>_rawaudio`; separated mode uses `temp/<video>`.
- Apply `ORIGINAL_VOLUME` only to raw original audio. When `ENABLE_AUDIO_SEPARATION=true`, mix the separated background stem at full volume unless a new explicit background-volume setting is added.
- Do not remove the Demucs `--two-stems vocals` contract unless the pipeline is updated to consume a different voice/background stem layout.
- Keep Demucs model/device configurable through `DEMUCS_MODEL` and `DEMUCS_DEVICE`; do not hard-code `cpu` or `mps` in the separation command.
- Keep `scripts/run_config_matrix.py` aligned with supported values whenever `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, or `TTS_ENGINE` options change.
- Keep VoxCPM2 in Controllable Cloning mode: pass per-segment `reference_wav_path` only for synthesis, and keep VoxCPM2 reference cache artifacts under `voxcpm2_*` names.
- Keep Irodori-TTS in server API mode: pass per-segment reference audio as `irodori.ref_wav`, do not send Caption / Style Prompt, and do not set fixed `seconds`.
