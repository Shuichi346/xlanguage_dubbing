#!/usr/bin/env python3
"""Build Irodori reference latents in the Irodori-TTS-Server environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def _load_audio(path: Path):
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)

    waveform = waveform.to(dtype=torch.float32)
    if waveform.ndim != 2:
        raise ValueError(f"Unsupported audio shape for {path}: {tuple(waveform.shape)}")
    if waveform.shape[0] != 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), int(sample_rate)


def _atomic_save_tensor(tensor, output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary_path.open("wb") as output_file:
            torch.save(tensor, output_file)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(value: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.name}.tmp.{os.getpid()}"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(value, output_file, ensure_ascii=False, indent=2)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_cache_metadata(
    manifest: dict[str, Any],
    speaker: dict[str, Any],
    latent,
    output_path: Path,
) -> None:
    metadata_path_value = manifest.get("metadata_path")
    metadata_value = speaker.get("metadata")
    if not metadata_path_value or not isinstance(metadata_value, dict):
        return

    metadata_path = Path(str(metadata_path_value))
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}

    speaker_metadata = dict(metadata_value)
    speaker_metadata.update(
        {
            "latent_sha256": _file_sha256(output_path),
            "latent_shape": [int(value) for value in latent.shape],
            "latent_dtype": str(latent.dtype),
        }
    )
    metadata[str(speaker["speaker_id"])] = speaker_metadata
    _atomic_write_json(metadata, metadata_path)


def build_reference_latents(manifest: dict[str, Any]) -> None:
    import torch

    # This worker runs in the separate Irodori-TTS-Server environment.
    from irodori_tts.codec import DACVAECodec  # ty: ignore[unresolved-import]

    codec_settings = manifest["codec"]
    if codec_settings.get("precision") != "fp32":
        raise ValueError("Irodori reference latent generation requires FP32.")
    if not codec_settings.get("deterministic_encode"):
        raise ValueError("Irodori reference latent generation must be deterministic.")

    codec = DACVAECodec.load(
        repo_id=str(codec_settings["repo_id"]),
        device=str(codec_settings["device"]),
        dtype=torch.float32,
        deterministic_encode=True,
        normalize_db=float(codec_settings["normalize_db"]),
    )
    expected_sample_rate = int(codec_settings["sample_rate"])
    if int(codec.sample_rate) != expected_sample_rate:
        raise ValueError(
            "Unexpected Irodori codec sample rate: "
            f"{codec.sample_rate} (expected {expected_sample_rate})"
        )

    max_steps = max(
        1,
        math.ceil(
            float(codec_settings["max_seconds"])
            * float(codec.sample_rate)
            / float(int(codec.model.hop_length))
        ),
    )
    expected_latent_dim = int(codec_settings["latent_dim"])

    for speaker in manifest["speakers"]:
        pieces = []
        current_steps = 0
        for clip_value in speaker["clips"]:
            clip_path = Path(str(clip_value))
            waveform, sample_rate = _load_audio(clip_path)
            if sample_rate != expected_sample_rate:
                raise ValueError(
                    f"Reference clip must be {expected_sample_rate} Hz: "
                    f"{clip_path} ({sample_rate} Hz)"
                )
            piece = codec.encode_waveform(
                waveform.unsqueeze(0),
                sample_rate=sample_rate,
                normalize_db=float(codec_settings["normalize_db"]),
                ensure_max=True,
            )[0].cpu().to(dtype=torch.float32).contiguous()
            if piece.ndim != 2 or piece.shape[0] == 0:
                raise ValueError(
                    f"Reference clip produced an invalid latent: {clip_path} "
                    f"shape={tuple(piece.shape)}"
                )
            if int(piece.shape[1]) != expected_latent_dim:
                raise ValueError(
                    f"Unexpected latent dimension for {clip_path}: "
                    f"{piece.shape[1]} (expected {expected_latent_dim})"
                )
            pieces.append(piece)
            current_steps += int(piece.shape[0])
            if current_steps >= max_steps:
                break

        if not pieces:
            raise ValueError(
                f"No reference clips for speaker {speaker['speaker_id']}"
            )

        combined = torch.cat(pieces, dim=0)[:max_steps]
        combined = combined.cpu().to(dtype=torch.float32).contiguous()
        if not torch.isfinite(combined).all():
            raise ValueError(
                f"Reference latent contains non-finite values: {speaker['speaker_id']}"
            )
        output_path = Path(str(speaker["output_path"]))
        _atomic_save_tensor(combined, output_path)
        _update_cache_metadata(manifest, speaker, combined, output_path)
        print(
            f"[irodori-reference] {speaker['speaker_id']}: "
            f"{len(pieces)} clips, {combined.shape[0]} latent steps",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    build_reference_latents(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
