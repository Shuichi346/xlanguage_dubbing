"""Create and persist native-language speaker references before cloning."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from xlanguage_dubbing import config
from xlanguage_dubbing.audio.ffmpeg import extract_audio_segment
from xlanguage_dubbing.core.models import Segment
from xlanguage_dubbing.tts.age_gender import (
    HF_MODEL_ID,
    HF_MODEL_REVISION,
    AgeGenderEstimator,
    VoiceProfile,
)
from xlanguage_dubbing.tts.reference import (
    SpeakerReferenceCache,
    _file_sha256,
    _irodori_codec_cache_settings,
    _is_valid_irodori_latent_cache,
    _json_sha256,
    _run_irodori_reference_worker,
    _source_audio_identity,
    _speaker_cache_key,
)
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    load_json_if_exists,
    print_step,
)

REFERENCE_TEXT = {
    "ja": (
        "今日は朝から気持ちのよい天気ですね。駅前の通りには、たくさんの人が行き交っています。"
        "少し遠回りになりますが、あの青い屋根の建物まで歩いてみましょう。"
        "新しい発見があるかもしれません。"
    ),
    "en": (
        "The weather has been lovely since this morning. Lots of people are walking along "
        "the street in front of the station. It is a bit of a detour, but let's walk to "
        "that building with the blue roof. We might discover something new."
    ),
}


def validate_voice_design(engine: str, language: str) -> None:
    if language not in REFERENCE_TEXT:
        raise PipelineError("ボイスデザインは OUTPUT_LANG=ja / en の場合のみ使用できます。")
    if engine not in {"omnivoice", "voxcpm2", "irodori", "irodori-tts", "irodori_tts"}:
        raise PipelineError(f"ボイスデザイン未対応の TTS_ENGINE: {engine}")
    if engine.startswith("irodori") and language != "ja":
        raise PipelineError("Irodori-TTS のボイスデザインは日本語のみ対応しています。")


def irodori_synthesis_settings() -> dict:
    """Fingerprint effective settings for both original and designed voices."""
    return {
        "model": config.IRODORI_HF_CHECKPOINT,
        "codec": _irodori_codec_cache_settings(),
        "sampling": config.irodori_sampling_options(),
        "speed": config.IRODORI_TTS_SPEED,
        "language": config.OUTPUT_LANG,
    }


def irodori_render_fingerprint(segments: list[Segment], cache: SpeakerReferenceCache) -> str:
    """Keep previous model audio, translations, and reference voices out of a render."""
    references = {}
    for speaker in sorted({segment.speaker_id for segment in segments}):
        path = cache.get_irodori_speaker_reference_latent_path(speaker)
        references[speaker] = _file_sha256(path) if path is not None else None
    return _json_sha256({
        "settings": irodori_synthesis_settings(),
        "references": references,
        "segments": [(s.speaker_id, s.start, s.end, s.text_tgt) for s in segments],
    })


def design_settings() -> dict:
    """Include synthesis settings in the namespace so old voices cannot leak in."""
    engine = config.TTS_ENGINE
    if engine.startswith("irodori"):
        engine = "irodori"
        options = irodori_synthesis_settings()
    elif engine == "voxcpm2":
        options = {
            "model": config.VOXCPM2_MODEL,
            "steps": config.VOXCPM2_INFERENCE_TIMESTEPS,
            "cfg": config.VOXCPM2_CFG_VALUE,
        }
    else:
        options = {
            "model": config.OMNIVOICE_MODEL,
            "steps": config.OMNIVOICE_NUM_STEP,
            "cfg": config.OMNIVOICE_GUIDANCE_SCALE,
            "dtype": config.OMNIVOICE_DTYPE,
        }
    return {
        "schema": 1,
        "engine": engine,
        "language": config.OUTPUT_LANG,
        "seed": config.VOICE_DESIGN_SEED,
        "model_options": options,
        "age_gender_model": str(config.AGE_GENDER_MODEL_DIR.resolve()),
        "age_gender_checkpoint": {"repo": HF_MODEL_ID, "revision": HF_MODEL_REVISION},
        "reference_text": REFERENCE_TEXT.get(config.OUTPUT_LANG, ""),
    }


def design_namespace() -> str:
    settings = design_settings()
    return f"{settings['engine']}_{settings['language']}_{_json_sha256(settings)[:12]}"


def output_path_for_video(video_path: Path) -> Path:
    suffix = config.OUTPUT_SUFFIX
    if config.ENABLE_VOICE_DESIGN:
        suffix = f"_voice_design_{design_namespace()}{suffix}"
    elif config.TTS_ENGINE.startswith("irodori"):
        suffix = f"_irodori_{_json_sha256(irodori_synthesis_settings())[:12]}{suffix}"
    return video_path.with_name(video_path.stem + suffix)


def _audio_duration(path: Path) -> float:
    """Decode the entire small reference to reject truncated or silent cache files."""
    info = sf.info(path)
    if not 1.0 <= info.duration <= 120.0:
        raise ValueError(f"Invalid voice-design reference length: {path}")
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    if rate <= 0 or len(audio) < rate or len(audio) > rate * 120:
        raise ValueError(f"Invalid voice-design reference length: {path}")
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) < 1e-5:
        raise ValueError(f"Invalid or silent voice-design reference: {path}")
    return len(audio) / rate


def _valid_sample(path: Path, info: dict, fingerprint: str) -> bool:
    try:
        return (
            info.get("fingerprint") == fingerprint
            and info.get("audio_sha256") == _file_sha256(path)
            and _audio_duration(path) > 0
        )
    except (OSError, ValueError, RuntimeError):
        return False


def voice_prompt(profile: VoiceProfile, language: str) -> str:
    if language == "ja":
        gender = "男性" if profile.gender == "male" else "女性"
        age = "若い大人" if profile.age_group == "young_adult" else "落ち着いた大人"
        return f"{age}の{gender}。日本語母語話者の自然で流暢な発音。普通の声の高さで、穏やかに明瞭に話す。"
    age = "young adult" if profile.age_group == "young_adult" else "middle-aged adult"
    return (
        f"A {age} {profile.gender} native English speaker with natural, fluent pronunciation, "
        "a moderate pitch and a calm, clear delivery."
    )


def _generate_sample(
    engine: str, language: str, profile: VoiceProfile, seed: int, path: Path
) -> None:
    text = REFERENCE_TEXT[language]
    prompt = voice_prompt(profile, language)
    if engine == "voxcpm2":
        from xlanguage_dubbing.voxcpm2_tts import voxcpm2_design_voice

        voxcpm2_design_voice(text, path, prompt, seed, language)
    elif engine == "irodori":
        from xlanguage_dubbing.irodori_tts import irodori_design_voice

        irodori_design_voice(text, path, prompt, seed, language)
    else:
        from xlanguage_dubbing.omnivoice_tts import omnivoice_design_voice

        omnivoice_design_voice(text, path, profile.gender, profile.age_group, seed, language)


def _build_latents(references: dict[str, Path], cache_dir: Path) -> tuple[dict, dict]:
    """Encode short utterances using the existing codec-only worker contract."""
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent

    codec = _irodori_codec_cache_settings()
    meta_path = cache_dir / "irodori_ref_latent_meta.json"
    previous = load_json_if_exists(meta_path) or {}
    latents: dict[str, Path] = {}
    durations: dict[str, float] = {}
    with tempfile.TemporaryDirectory(prefix=".design_latents_", dir=cache_dir) as temp:
        pending = []
        for speaker, audio_path in references.items():
            key = _speaker_cache_key(speaker)
            path = cache_dir / f"irodori_ref_latent_{key}.pt"
            fingerprint = _json_sha256({"audio": _file_sha256(audio_path), "codec": codec})
            info = previous.get(speaker) if isinstance(previous, dict) else None
            durations[speaker] = min(_audio_duration(audio_path), config.IRODORI_REFERENCE_MAX_SEC)
            latents[speaker] = path
            if isinstance(info, dict) and _is_valid_irodori_latent_cache(
                path, info, fingerprint=fingerprint
            ):
                continue
            audio = AudioSegment.from_file(audio_path).set_frame_rate(48000).set_channels(1)
            audio = audio[: round(durations[speaker] * 1000)]
            ranges = detect_nonsilent(
                audio, min_silence_len=250, silence_thresh=max(audio.dBFS - 16, -50)
            )
            clips = []
            # Retain short pauses while bounding any utterance without a detected pause.
            for start, end in ranges:
                for offset in range(max(0, start - 80), min(len(audio), end + 80), 10000):
                    clip = Path(temp) / f"{key}_{len(clips):03d}.wav"
                    piece = audio[offset : min(offset + 10000, end + 80)]
                    samples = np.array(piece.get_array_of_samples(), dtype=np.float32)
                    samples /= float(1 << (8 * piece.sample_width - 1))
                    sf.write(clip, samples, 48000, subtype="FLOAT")
                    clips.append(str(clip.resolve()))
            if not clips:
                raise PipelineError(f"ボイスデザイン参照音声に発話がありません: {speaker}")
            pending.append(
                {
                    "speaker_id": speaker,
                    "clips": clips,
                    "output_path": str(path.resolve()),
                    "metadata": {
                        "fingerprint": fingerprint,
                        "reference_latent": path.name,
                        "duration_sec": durations[speaker],
                    },
                }
            )
        if pending:
            from xlanguage_dubbing.irodori_tts import release_irodori_tts_server

            release_irodori_tts_server()
            manifest = Path(temp) / "manifest.json"
            atomic_write_json(
                manifest,
                {
                    "schema_version": 1,
                    "codec": codec,
                    "metadata_path": str(meta_path.resolve()),
                    "speakers": pending,
                },
            )
            _run_irodori_reference_worker(manifest, server_dir=config.IRODORI_TTS_DIR.resolve())
        updated = load_json_if_exists(meta_path) or {}
        for speaker, path in latents.items():
            info = updated.get(speaker) if isinstance(updated, dict) else None
            fingerprint = _json_sha256(
                {"audio": _file_sha256(references[speaker]), "codec": codec}
            )
            if not isinstance(info, dict) or not _is_valid_irodori_latent_cache(
                path, info, fingerprint=fingerprint
            ):
                raise PipelineError(f"生成声の参照潜在が不正です: {speaker}")
    return latents, durations


def build_designed_references(
    media_path: Path, segments: list[Segment], ref_cache: SpeakerReferenceCache
) -> str:
    settings = design_settings()
    engine, language = settings["engine"], settings["language"]
    validate_voice_design(engine, language)
    cache_dir = ref_cache.cache_dir
    source = _source_audio_identity(media_path)
    speakers = sorted({segment.speaker_id for segment in segments})
    print_step(f"  ボイスデザイン: {len(speakers)} 話者 ({engine}, {language})")
    meta_path = cache_dir / f"{engine}_voice_design.json"
    previous = load_json_if_exists(meta_path) or {}
    metadata: dict[str, dict] = {}
    references: dict[str, Path] = {}
    pending: list[tuple[str, Path, str, int, list[Segment]]] = []
    for speaker in speakers:
        key = _speaker_cache_key(speaker)
        path = cache_dir / f"{engine}_designed_{key}.wav"
        utterances = [s for s in segments if s.speaker_id == speaker and s.duration >= 1]
        fingerprint = _json_sha256(
            {
                "source": source,
                "settings": settings,
                "speaker": speaker,
                "ranges": [(s.start, s.end) for s in utterances],
            }
        )
        references[speaker] = path
        info = previous.get(speaker) if isinstance(previous, dict) else None
        if isinstance(info, dict) and _valid_sample(path, info, fingerprint):
            metadata[speaker] = info
            continue
        seed = int(
            hashlib.sha256(
                f"{config.VOICE_DESIGN_SEED}:{speaker}:{source['sha256']}".encode()
            ).hexdigest()[:8],
            16,
        )
        pending.append((speaker, path, fingerprint, seed, utterances))

    profiles: dict[str, VoiceProfile] = {}
    estimator = AgeGenderEstimator(config.AGE_GENDER_MODEL_DIR)
    try:
        with tempfile.TemporaryDirectory(prefix=".design_analysis_", dir=cache_dir) as temp:
            for speaker, _, _, _, utterances in pending:
                profile = VoiceProfile(fallback_reason="No usable speaker utterance")
                if utterances and speaker:
                    best = max(utterances, key=lambda s: s.duration)
                    clip = Path(temp) / f"{_speaker_cache_key(speaker)}.wav"
                    try:
                        extract_audio_segment(
                            media_path,
                            clip,
                            start=best.start,
                            end=min(best.end, best.start + 10),
                            sample_rate=16000,
                            channels=1,
                        )
                        profile = estimator.estimate(clip)
                    except Exception as exc:
                        profile = VoiceProfile(fallback_reason=str(exc))
                profiles[speaker] = profile
                if profile.fallback_reason:
                    print_step(
                        f"  {speaker or 'UNKNOWN'}: 判定不明 → 大人の男性 ({profile.fallback_reason})"
                    )
    finally:
        estimator.close()

    # All acoustic estimation finishes before a TTS model is loaded.
    for speaker, path, fingerprint, seed, _ in pending:
        profile = profiles[speaker]
        print_step(f"  生成声作成: {speaker or 'UNKNOWN'} ({profile.gender}, {profile.age_group})")
        with tempfile.TemporaryDirectory(prefix=".design_sample_", dir=cache_dir) as temp:
            temporary_path = Path(temp) / "sample.wav"
            _generate_sample(engine, language, profile, seed, temporary_path)
            duration = _audio_duration(temporary_path)
            temporary_path.replace(path)
        metadata[speaker] = {
            "fingerprint": fingerprint,
            "seed": seed,
            "profile": asdict(profile),
            "prompt": voice_prompt(profile, language),
            "reference_text": REFERENCE_TEXT[language],
            "duration_sec": duration,
            "audio_sha256": _file_sha256(path),
        }
        # Persist each completed voice to keep it stable across interrupted runs.
        atomic_write_json(
            meta_path, {**previous, **metadata} if isinstance(previous, dict) else metadata
        )
    atomic_write_json(meta_path, metadata)
    latents, durations = _build_latents(references, cache_dir) if engine == "irodori" else ({}, {})
    ref_cache.use_designed_references(references, REFERENCE_TEXT[language], latents, durations)
    return _json_sha256(
        {"voices": metadata, "latents": {s: _file_sha256(p) for s, p in latents.items()}}
    )
