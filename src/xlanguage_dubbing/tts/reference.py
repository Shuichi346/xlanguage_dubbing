#!/usr/bin/env python3
"""
話者別リファレンス音声の抽出・キャッシュ管理。
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from xlanguage_dubbing.audio.ffmpeg import extract_audio_segment
from xlanguage_dubbing.config import (
    IRODORI_CODEC_DEVICE,
    IRODORI_CODEC_REPO,
    IRODORI_HF_CHECKPOINT,
    IRODORI_REFERENCE_MAX_SEC,
    IRODORI_TTS_DIR,
    MIN_SEGMENT_SEC,
    OMNIVOICE_REFERENCE_MAX_SEC,
    OMNIVOICE_REFERENCE_MIN_SEC,
    OMNIVOICE_REFERENCE_TARGET_SEC,
    TTS_ENGINE,
)
from xlanguage_dubbing.core.models import DiarizationSegment, Segment
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    ensure_dir,
    load_json_if_exists,
    normalize_spaces,
    print_step,
    resolve_executable,
)

_IRODORI_CACHE_SCHEMA_VERSION = 1
_IRODORI_SAMPLE_RATE = 48000
_IRODORI_CHANNELS = 1
_IRODORI_PRECISION = "fp32"
_IRODORI_LATENT_DIM = 32
_IRODORI_NORMALIZE_DB = -16.0
_IRODORI_DETERMINISTIC_ENCODE = True


class SpeakerReferenceCache:
    """話者ごとのリファレンス音声をキャッシュ管理する。"""

    def __init__(self, cache_dir: Path, tts_engine: str | None = None) -> None:
        self._cache_dir = cache_dir
        self._tts_engine = _normalize_reference_engine(tts_engine or TTS_ENGINE)
        self._omnivoice_refs: dict[str, Path] = {}
        self._omnivoice_prompt_texts: dict[str, str] = {}
        self._omnivoice_segment_refs: dict[int, Path] = {}
        self._omnivoice_segment_prompt_texts: dict[int, str] = {}
        self._irodori_speaker_ref_latents: dict[str, Path] = {}
        self._irodori_speaker_ref_durations: dict[str, float] = {}
        ensure_dir(cache_dir)

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def segment_reference_dir(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_segment_refs"

    def get_omnivoice_reference_path(self, speaker_id: str) -> Path | None:
        path = self._omnivoice_refs.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_prompt_text(self, speaker_id: str) -> str:
        return self._omnivoice_prompt_texts.get(speaker_id, "")

    def get_omnivoice_segment_reference_path(self, segno: int) -> Path | None:
        path = self._omnivoice_segment_refs.get(segno)
        if path and path.exists():
            return path.resolve()
        return None

    def get_omnivoice_segment_prompt_text(self, segno: int) -> str:
        return self._omnivoice_segment_prompt_texts.get(segno, "")

    def get_irodori_speaker_reference_latent_path(
        self,
        speaker_id: str,
    ) -> Path | None:
        path = self._irodori_speaker_ref_latents.get(speaker_id)
        if path and path.exists():
            return path.resolve()
        return None

    def get_irodori_speaker_reference_duration(self, speaker_id: str) -> float:
        return self._irodori_speaker_ref_durations.get(speaker_id, 0.0)

    def reload_speaker_references(self, speaker_ids: set[str]) -> None:
        for speaker_id in speaker_ids:
            ov_wav = self._cache_dir / f"ovref_{speaker_id}.wav"
            if ov_wav.exists():
                self._omnivoice_refs[speaker_id] = ov_wav
        self._load_omnivoice_prompt_meta()
        self._load_omnivoice_segment_meta()

    def build_irodori_speaker_references(
        self,
        media_path: Path,
        segments: list[Segment],
    ) -> None:
        """Build one reusable latent from ordered short utterances per speaker."""
        self._irodori_speaker_ref_latents.clear()
        self._irodori_speaker_ref_durations.clear()

        speakers: dict[str, list[Segment]] = {}
        for segment in segments:
            if segment.speaker_id:
                speakers.setdefault(segment.speaker_id, []).append(segment)

        codec_settings = _irodori_codec_cache_settings()
        source_identity = _source_audio_identity(media_path)
        meta_path = self._cache_dir / "irodori_ref_latent_meta.json"
        previous_meta = load_json_if_exists(meta_path)
        if not isinstance(previous_meta, dict):
            previous_meta = {}

        reference_meta: dict[str, dict[str, object]] = {}
        pending: list[dict[str, object]] = []
        for speaker_id, speaker_segments in speakers.items():
            ranges = _select_irodori_reference_ranges(
                speaker_segments,
                max_sec=IRODORI_REFERENCE_MAX_SEC,
            )
            if not ranges:
                print_step(f"  警告: Irodori リファレンスなし: {speaker_id}")
                continue

            cache_key = _speaker_cache_key(speaker_id)
            latent_path = (
                self._cache_dir / f"irodori_ref_latent_{cache_key}.pt"
            )
            duration = sum(end - start for start, end in ranges)
            fingerprint_payload = {
                "schema_version": _IRODORI_CACHE_SCHEMA_VERSION,
                "speaker_id": speaker_id,
                "source_audio": source_identity,
                "ranges": [
                    {"start": start, "end": end} for start, end in ranges
                ],
                "codec": codec_settings,
            }
            fingerprint = _json_sha256(fingerprint_payload)
            cached_info = previous_meta.get(speaker_id)

            if _is_valid_irodori_latent_cache(
                latent_path,
                cached_info,
                fingerprint=fingerprint,
            ):
                self._irodori_speaker_ref_latents[speaker_id] = latent_path
                self._irodori_speaker_ref_durations[speaker_id] = duration
                reference_meta[speaker_id] = dict(cached_info)
                print_step(
                    f"  Irodori 話者別参照潜在キャッシュ再利用: "
                    f"{speaker_id} ({duration:.1f}s, {len(ranges)} clips)"
                )
                continue

            pending.append(
                {
                    "speaker_id": speaker_id,
                    "cache_key": cache_key,
                    "ranges": ranges,
                    "latent_path": latent_path,
                    "duration_sec": duration,
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                }
            )

        if pending:
            _build_pending_irodori_latents(
                media_path=media_path,
                cache_dir=self._cache_dir,
                codec_settings=codec_settings,
                pending=pending,
            )

        for item in pending:
            speaker_id = str(item["speaker_id"])
            ranges = item["ranges"]
            latent_path = Path(item["latent_path"])
            duration = float(item["duration_sec"])
            latent_info = _load_irodori_latent_info(latent_path)
            if latent_info is None:
                raise PipelineError(
                    "Irodori 参照潜在キャッシュの生成結果が不正です: "
                    f"{latent_path}"
                )

            info: dict[str, object] = {
                "reference_latent": latent_path.name,
                "duration_sec": duration,
                "ranges": [
                    {"start": start, "end": end} for start, end in ranges
                ],
                "fingerprint": item["fingerprint"],
                "fingerprint_payload": item["fingerprint_payload"],
                "latent_sha256": _file_sha256(latent_path),
                "latent_shape": latent_info["shape"],
                "latent_dtype": latent_info["dtype"],
            }
            self._irodori_speaker_ref_latents[speaker_id] = latent_path
            self._irodori_speaker_ref_durations[speaker_id] = duration
            reference_meta[speaker_id] = info
            print_step(
                f"  Irodori 話者別参照潜在キャッシュ生成: {speaker_id} "
                f"({duration:.1f}s, {len(ranges)} clips)"
            )

        atomic_write_json(meta_path, reference_meta)

    def build_omnivoice_references(
        self,
        video_path: Path,
        diarization: list[DiarizationSegment],
        segments: list[Segment],
    ) -> None:
        from xlanguage_dubbing.asr import transcribe_reference_audio

        speakers: dict[str, list[DiarizationSegment]] = {}
        for dia in diarization:
            speakers.setdefault(dia.speaker, []).append(dia)

        prompt_meta: dict[str, dict[str, str | float]] = {}

        for speaker_id, dia_segments in speakers.items():
            out_wav = self._cache_dir / f"ovref_{speaker_id}.wav"
            if out_wav.exists():
                self._omnivoice_refs[speaker_id] = out_wav
                if speaker_id in self._omnivoice_prompt_texts:
                    continue

            best_seg = _select_best_reference_segment(
                dia_segments,
                min_sec=OMNIVOICE_REFERENCE_MIN_SEC,
                max_sec=OMNIVOICE_REFERENCE_MAX_SEC,
                target_sec=OMNIVOICE_REFERENCE_TARGET_SEC,
            )
            if best_seg is None:
                print_step(f"  警告: リファレンスなし: {speaker_id}")
                continue

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=best_seg.start, end=best_seg.end,
                    sample_rate=44100, channels=1,
                )

            if out_wav.exists() and out_wav.stat().st_size > 100:
                self._omnivoice_refs[speaker_id] = out_wav
                ref_dur = best_seg.end - best_seg.start

                prompt_text = transcribe_reference_audio(out_wav, language="")
                if not prompt_text:
                    prompt_text = _collect_reference_prompt_text(
                        best_seg, segments
                    )

                self._omnivoice_prompt_texts[speaker_id] = prompt_text
                prompt_meta[speaker_id] = {
                    "prompt_text": prompt_text,
                    "ref_start": best_seg.start,
                    "ref_end": best_seg.end,
                }
                preview = prompt_text[:80] if prompt_text else "(empty)"
                print_step(
                    f"  {_reference_engine_label(self._tts_engine)} "
                    f"リファレンス生成: {speaker_id} "
                    f"({ref_dur:.1f}s) text='{preview}'"
                )
            else:
                out_wav.unlink(missing_ok=True)

        self._save_omnivoice_prompt_meta(prompt_meta)

    def build_omnivoice_segment_references(
        self,
        video_path: Path,
        segments: list[Segment],
    ) -> None:
        seg_ref_dir = self.segment_reference_dir
        ensure_dir(seg_ref_dir)

        segment_meta: dict[str, dict[str, str | float]] = {}

        for segno, seg in enumerate(segments, start=1):
            out_wav = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"

            if (
                out_wav.exists()
                and segno in self._omnivoice_segment_prompt_texts
            ):
                self._omnivoice_segment_refs[segno] = out_wav
                continue

            dur = seg.end - seg.start
            if dur < 0.3:
                continue

            effective_end = seg.start + min(dur, OMNIVOICE_REFERENCE_MAX_SEC)

            if not out_wav.exists():
                extract_audio_segment(
                    video_path, out_wav,
                    start=seg.start, end=effective_end,
                    sample_rate=44100, channels=1,
                )

            if not out_wav.exists() or out_wav.stat().st_size <= 100:
                out_wav.unlink(missing_ok=True)
                continue

            self._omnivoice_segment_refs[segno] = out_wav

            prompt_text = normalize_spaces(seg.text_src)
            self._omnivoice_segment_prompt_texts[segno] = prompt_text

            segment_meta[str(segno)] = {
                "prompt_text": prompt_text,
                "ref_start": seg.start,
                "ref_end": effective_end,
            }

        self._save_omnivoice_segment_meta(segment_meta)

        generated = len(self._omnivoice_segment_refs)
        print_step(
            f"  {_reference_engine_label(self._tts_engine)} "
            "セグメント単位リファレンス生成: "
            f"{generated}/{len(segments)} 件"
        )

    def reload_omnivoice_segment_references(self, total_segments: int) -> None:
        seg_ref_dir = self.segment_reference_dir
        if not seg_ref_dir.exists():
            return
        for segno in range(1, total_segments + 1):
            wav_path = seg_ref_dir / f"ovseg_ref_{segno:05d}.wav"
            if wav_path.exists():
                self._omnivoice_segment_refs[segno] = wav_path
        self._load_omnivoice_segment_meta()

    def _save_omnivoice_prompt_meta(self, meta):
        meta_path = self._prompt_meta_path()
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_prompt_meta(self):
        meta_path = self._prompt_meta_path()
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for speaker_id, info in obj.items():
            if not isinstance(info, dict):
                continue
            self._omnivoice_prompt_texts[speaker_id] = str(
                info.get("prompt_text", "") or ""
            )

    def _prompt_meta_path(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_prompt_meta.json"

    def _save_omnivoice_segment_meta(self, meta):
        meta_path = self._segment_meta_path()
        existing = load_json_if_exists(meta_path)
        if isinstance(existing, dict):
            existing.update(meta)
            meta = existing
        atomic_write_json(meta_path, meta)

    def _load_omnivoice_segment_meta(self):
        meta_path = self._segment_meta_path()
        obj = load_json_if_exists(meta_path)
        if not isinstance(obj, dict):
            return
        for segno_str, info in obj.items():
            if not isinstance(info, dict):
                continue
            try:
                segno = int(segno_str)
            except (ValueError, TypeError):
                continue
            self._omnivoice_segment_prompt_texts[segno] = str(
                info.get("prompt_text", "") or ""
            )

    def _segment_meta_path(self) -> Path:
        return self._cache_dir / f"{self._tts_engine}_segment_meta.json"

    def clear(self):
        self._omnivoice_refs.clear()
        self._omnivoice_prompt_texts.clear()
        self._omnivoice_segment_refs.clear()
        self._omnivoice_segment_prompt_texts.clear()
        self._irodori_speaker_ref_latents.clear()
        self._irodori_speaker_ref_durations.clear()
        gc.collect()


def _select_best_reference_segment(dia_segments, min_sec, max_sec, target_sec):
    candidates = []
    for seg in dia_segments:
        dur = seg.end - seg.start
        if dur < min_sec:
            continue
        candidates.append(seg)

    if not candidates:
        if dia_segments:
            longest = max(dia_segments, key=lambda s: s.end - s.start)
            if longest.end - longest.start >= 1.0:
                return longest
        return None

    def score(seg):
        dur = seg.end - seg.start
        if dur > max_sec:
            dur = max_sec
        return abs(dur - target_sec)

    best = min(candidates, key=score)

    dur = best.end - best.start
    if dur > max_sec:
        return DiarizationSegment(
            start=best.start,
            end=best.start + max_sec,
            speaker=best.speaker,
        )
    return best


def _collect_reference_prompt_text(reference_seg, segments):
    overlapped = []
    for seg in segments:
        overlap = min(reference_seg.end, seg.end) - max(
            reference_seg.start, seg.start
        )
        if overlap > 0:
            overlapped.append((overlap, seg))

    if overlapped:
        overlapped.sort(key=lambda item: item[1].start)
        texts = [normalize_spaces(seg.text_src) for _, seg in overlapped]
        return normalize_spaces(" ".join(t for t in texts if t))

    if not segments:
        return ""

    ref_center = (reference_seg.start + reference_seg.end) / 2.0
    nearest = min(
        segments,
        key=lambda seg: abs(((seg.start + seg.end) / 2.0) - ref_center),
    )
    return normalize_spaces(nearest.text_src)


def _source_audio_identity(media_path: Path) -> dict[str, object]:
    resolved = media_path.expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise PipelineError(f"参照元音声が見つかりません: {resolved}") from exc
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "inode": int(stat.st_ino),
        "sha256": _file_sha256(resolved),
    }


def _irodori_codec_cache_settings() -> dict[str, object]:
    server_lock = IRODORI_TTS_DIR.expanduser() / "uv.lock"
    return {
        "repo_id": IRODORI_CODEC_REPO,
        "device": IRODORI_CODEC_DEVICE,
        "precision": _IRODORI_PRECISION,
        "deterministic_encode": _IRODORI_DETERMINISTIC_ENCODE,
        "normalize_db": _IRODORI_NORMALIZE_DB,
        "sample_rate": _IRODORI_SAMPLE_RATE,
        "channels": _IRODORI_CHANNELS,
        "latent_dim": _IRODORI_LATENT_DIM,
        "max_seconds": IRODORI_REFERENCE_MAX_SEC,
        "tts_checkpoint": IRODORI_HF_CHECKPOINT,
        "server_lock_sha256": (
            _file_sha256(server_lock) if server_lock.is_file() else None
        ),
    }


def _build_pending_irodori_latents(
    *,
    media_path: Path,
    cache_dir: Path,
    codec_settings: dict[str, object],
    pending: list[dict[str, object]],
) -> None:
    server_dir = IRODORI_TTS_DIR.expanduser().resolve()
    if not server_dir.is_dir():
        raise PipelineError(
            "Irodori 参照潜在を作成するサーバー環境が見つかりません。\n"
            f"  IRODORI_TTS_DIR={server_dir}\n"
            "  Irodori-TTS-Server を clone して uv sync --extra cpu を実行してください。"
        )

    with tempfile.TemporaryDirectory(
        prefix=".irodori_ref_latent_work_",
        dir=cache_dir,
    ) as temporary_dir_value:
        temporary_dir = Path(temporary_dir_value)
        speakers_manifest: list[dict[str, object]] = []

        for item in pending:
            speaker_id = str(item["speaker_id"])
            ranges = item["ranges"]
            speaker_dir = temporary_dir / str(item["cache_key"])
            ensure_dir(speaker_dir)
            clip_paths: list[str] = []
            for clip_index, (start, end) in enumerate(item["ranges"], start=1):
                clip_path = speaker_dir / f"clip_{clip_index:04d}.wav"
                extract_audio_segment(
                    media_path,
                    clip_path,
                    start=float(start),
                    end=float(end),
                    sample_rate=_IRODORI_SAMPLE_RATE,
                    channels=_IRODORI_CHANNELS,
                )
                if not clip_path.is_file() or clip_path.stat().st_size <= 100:
                    raise PipelineError(
                        f"Irodori 参照音声の抽出に失敗しました: {speaker_id} "
                        f"{start:.3f}-{end:.3f}"
                    )
                clip_paths.append(str(clip_path.resolve()))

            speakers_manifest.append(
                {
                    "speaker_id": speaker_id,
                    "clips": clip_paths,
                    "output_path": str(Path(item["latent_path"]).resolve()),
                    "metadata": {
                        "reference_latent": Path(item["latent_path"]).name,
                        "duration_sec": item["duration_sec"],
                        "ranges": [
                            {"start": start, "end": end}
                            for start, end in ranges
                        ],
                        "fingerprint": item["fingerprint"],
                        "fingerprint_payload": item["fingerprint_payload"],
                    },
                }
            )

        manifest_path = temporary_dir / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": _IRODORI_CACHE_SCHEMA_VERSION,
                "codec": codec_settings,
                "metadata_path": str(
                    (cache_dir / "irodori_ref_latent_meta.json").resolve()
                ),
                "speakers": speakers_manifest,
            },
        )
        _run_irodori_reference_worker(manifest_path, server_dir=server_dir)


def _run_irodori_reference_worker(
    manifest_path: Path,
    *,
    server_dir: Path,
) -> None:
    worker_path = Path(__file__).with_name("irodori_reference_worker.py").resolve()
    command = [
        resolve_executable("uv"),
        "run",
        "--no-sync",
        "python",
        str(worker_path),
        "--manifest",
        str(manifest_path.resolve()),
    ]
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    environment["IRODORI_CODEC_DEVICE"] = IRODORI_CODEC_DEVICE
    environment["IRODORI_CODEC_REPO"] = IRODORI_CODEC_REPO
    environment["IRODORI_CODEC_PRECISION"] = _IRODORI_PRECISION
    environment["IRODORI_CODEC_DETERMINISTIC_ENCODE"] = "true"

    print_step(
        "  Irodori 参照潜在を短命プロセスで事前生成: "
        f"{len(load_json_if_exists(manifest_path).get('speakers', []))} 話者"
    )
    process = subprocess.run(
        command,
        cwd=server_dir,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if process.stdout.strip():
        for line in process.stdout.strip().splitlines():
            print_step(f"  {line}")
    if process.returncode != 0:
        raise PipelineError(
            "Irodori 参照潜在の事前生成に失敗しました。\n"
            f"  command: {' '.join(command)}\n"
            f"  stdout:\n{process.stdout}\n"
            f"  stderr:\n{process.stderr}"
        )


def _load_irodori_latent_info(path: Path) -> dict[str, object] | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    try:
        import torch

        latent = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32:
        return None
    valid_shape = (
        latent.ndim == 2
        and latent.shape[0] > 0
        and latent.shape[1] == _IRODORI_LATENT_DIM
    ) or (
        latent.ndim == 3
        and latent.shape[0] == 1
        and latent.shape[1] > 0
        and latent.shape[2] == _IRODORI_LATENT_DIM
    )
    if not valid_shape or not torch.isfinite(latent).all():
        return None
    return {
        "shape": [int(value) for value in latent.shape],
        "dtype": str(latent.dtype),
    }


def _is_valid_irodori_latent_cache(
    path: Path,
    cached_info: object,
    *,
    fingerprint: str,
) -> bool:
    if not isinstance(cached_info, dict):
        return False
    if cached_info.get("reference_latent") != path.name:
        return False
    if cached_info.get("fingerprint") != fingerprint:
        return False
    expected_sha256 = cached_info.get("latent_sha256")
    if not isinstance(expected_sha256, str):
        return False
    if not path.is_file() or _file_sha256(path) != expected_sha256:
        return False
    latent_info = _load_irodori_latent_info(path)
    if latent_info is None:
        return False
    return (
        cached_info.get("latent_shape") == latent_info["shape"]
        and cached_info.get("latent_dtype") == latent_info["dtype"]
    )


def _json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_irodori_reference_ranges(
    segments: list[Segment],
    *,
    max_sec: float,
) -> list[tuple[float, float]]:
    """Select non-overlapping utterances up to Irodori's 120-second limit."""
    limit = min(120.0, max(0.0, float(max_sec)))
    if limit < MIN_SEGMENT_SEC:
        return []

    candidates = sorted(
        segments,
        key=lambda segment: (-(segment.end - segment.start), segment.start),
    )
    selected: list[tuple[float, float]] = []
    total = 0.0

    for segment in candidates:
        start = max(0.0, float(segment.start))
        end = max(start, float(segment.end))
        if end - start < MIN_SEGMENT_SEC:
            continue
        if any(
            start < selected_end and end > selected_start
            for selected_start, selected_end in selected
        ):
            continue

        remaining = limit - total
        if remaining < MIN_SEGMENT_SEC:
            break
        end = min(end, start + remaining)
        if end - start < MIN_SEGMENT_SEC:
            continue

        selected.append((start, end))
        total += end - start
        if total >= limit - 1e-6:
            break

    return sorted(selected)


def _speaker_cache_key(speaker_id: str) -> str:
    readable = "".join(
        character if character.isascii() and character.isalnum() else "_"
        for character in speaker_id
    ).strip("_")
    readable = readable[:40] or "speaker"
    digest = hashlib.sha256(speaker_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable}_{digest}"


def _normalize_reference_engine(tts_engine: str) -> str:
    normalized = tts_engine.strip().lower()
    if normalized == "voxcpm2":
        return "voxcpm2"
    if normalized in {"irodori", "irodori-tts", "irodori_tts"}:
        return "irodori"
    return "omnivoice"


def _reference_engine_label(tts_engine: str) -> str:
    if tts_engine == "voxcpm2":
        return "VoxCPM2"
    if tts_engine == "irodori":
        return "Irodori"
    return "OmniVoice"
