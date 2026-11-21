"""Stable identities for source analysis, synthesis, and completed outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from xlanguage_dubbing import config
from xlanguage_dubbing.utils import atomic_write_json, load_json_if_exists


def fingerprint(value: Any) -> str:
    """Hash JSON-compatible settings without depending on dictionary order."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def analysis_settings() -> dict:
    """Include only the selected recognizer and source segmentation settings."""
    engine = config.ASR_ENGINE.strip().lower()
    prefix = {"whisper": "WHISPER_", "vibevoice": "VIBEVOICE_", "nemo-speech": "NEMO_SPEECH_"}[
        engine
    ]
    prefixes = (prefix, "MERGE_", "SPACY_")
    settings = {
        key: value
        for key, value in vars(config).items()
        if key.startswith(prefixes) and not key.endswith("_BIN")
    }
    settings.update(
        schema=2,
        engine=engine,
        language=config.INPUT_LANG,
        separation=config.ENABLE_AUDIO_SEPARATION,
    )
    if config.ENABLE_AUDIO_SEPARATION:
        settings["demucs_model"] = config.DEMUCS_MODEL
    if engine == "whisper":
        settings["vad_model"] = config.VAD_MODEL
        settings["diarization_model"] = config.PYANNOTE_MODEL
    return json.loads(json.dumps(settings, default=str))


def input_namespace(video_path: Path) -> str:
    """Keep inputs and analysis settings isolated without deleting old artifacts."""
    from xlanguage_dubbing.core.progress import video_signature

    return (
        "input_"
        + fingerprint(
            {
                "video": video_signature(video_path),
                "analysis": analysis_settings(),
            }
        )[:20]
    )


def synthesis_settings() -> dict:
    """Include every effective engine setting while omitting inactive MF controls."""
    engine = config.TTS_ENGINE
    if engine.startswith("irodori"):
        from xlanguage_dubbing.tts.voice_design import irodori_synthesis_settings

        options = irodori_synthesis_settings()
        engine = "irodori"
    else:
        prefix = "VOXCPM2_" if engine == "voxcpm2" else "OMNIVOICE_"
        options = {key: value for key, value in vars(config).items() if key.startswith(prefix)}
    return {
        "schema": 2,
        "engine": engine,
        "options": options,
        "language": config.OUTPUT_LANG,
        "min_segment": config.MIN_SEGMENT_SEC,
        "rate": config.TTS_SAMPLE_RATE,
        "channels": config.TTS_CHANNELS,
    }


def output_settings() -> dict:
    """Identify synthesis and presentation choices independently of translation."""
    return {
        "synthesis": synthesis_settings(),
        "height": config.OUTPUT_SIZE,
        "separation": config.ENABLE_AUDIO_SEPARATION,
        "background_volume": 1.0 if config.ENABLE_AUDIO_SEPARATION else config.ORIGINAL_VOLUME,
        "dubbed_volume": config.DUBBED_VOLUME,
    }


def reference_identity(directory: Path) -> dict:
    """Fingerprint selected cached references with bounded streaming reads."""
    from xlanguage_dubbing.core.progress import video_signature

    engine = config.TTS_ENGINE
    prefixes = (
        ("irodori_",)
        if engine.startswith("irodori")
        else (("voxcpm2_",) if engine == "voxcpm2" else ("ovref_", "ovseg_ref_", "omnivoice_"))
    )
    return {
        str(path.relative_to(directory)): video_signature(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
        and path.suffix in {".wav", ".pt", ".json"}
        and (config.ENABLE_VOICE_DESIGN or path.name.startswith(prefixes))
    }


def completion_path(output: Path) -> Path:
    return output.with_suffix(".complete.json")


def completed_output_is_valid(output: Path, video: Path, references: Path) -> bool:
    """Trust only outputs published by this version with unchanged identities."""
    from xlanguage_dubbing.core.progress import video_signature

    manifest = load_json_if_exists(completion_path(output))
    if not isinstance(manifest, dict) or manifest.get("schema") != 2 or not output.is_file():
        return False
    if (
        manifest.get("input") != input_namespace(video)
        or manifest.get("settings") != fingerprint(output_settings())
        or manifest.get("output") != video_signature(output)
    ):
        return False
    return (
        not manifest.get("references_retained", True)
        or not references.exists()
        or manifest.get("references") == reference_identity(references)
    )


def publish_completion(output: Path, video: Path, references: Path) -> None:
    """Publish completion only after the media renderer has validated its output."""
    from xlanguage_dubbing.core.progress import video_signature

    atomic_write_json(
        completion_path(output),
        {
            "schema": 2,
            "input": input_namespace(video),
            "settings": fingerprint(output_settings()),
            "output": video_signature(output),
            "references": reference_identity(references),
            "references_retained": config.KEEP_TEMP,
        },
    )
