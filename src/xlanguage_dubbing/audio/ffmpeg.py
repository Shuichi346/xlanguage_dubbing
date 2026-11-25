#!/usr/bin/env python3
"""
FFmpeg関連ユーティリティ。
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import soundfile as sf

from xlanguage_dubbing.config import (
    DUBBED_VOLUME,
    ORIGINAL_VOLUME,
    OUTPUT_SIZE,
    TTS_CHANNELS,
    TTS_SAMPLE_RATE,
)
from xlanguage_dubbing.utils import (
    PipelineError,
    atomic_write_json,
    atomic_write_text,
    ensure_dir,
    ffmpeg_concat_quote,
    load_json_if_exists,
    run_cmd,
    which_or_raise,
)

_MIN_TRIM_SEC = 0.05
_FLAC_FRAME_SAMPLES = 4096


def ffprobe_duration_sec(media_path: Path) -> float:
    """メディアファイルの長さを取得する。"""
    which_or_raise("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(media_path),
    ]
    proc = run_cmd(cmd)
    s = (proc.stdout or "").strip()
    try:
        return float(s)
    except ValueError as exc:
        raise PipelineError(
            f"ffprobe duration の解析に失敗: {media_path} / '{s}'"
        ) from exc


def ffprobe_has_audio(media_path: Path) -> bool:
    """メディアファイルに音声があるかを確認する。"""
    which_or_raise("ffprobe")
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(media_path),
    ]
    proc = run_cmd(cmd, check=False)
    return bool((proc.stdout or "").strip())



def ffprobe_video_frame_rate(media_path: Path) -> float:
    """Return a finite positive output cadence from the first video stream."""
    result = run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json", str(media_path),
    ])
    streams = json.loads(result.stdout).get("streams", [])
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            rate = float(Fraction(streams[0][key]))
            if math.isfinite(rate) and rate > 0:
                return rate
        except (IndexError, KeyError, ValueError, ZeroDivisionError):
            continue
    raise PipelineError(f"動画フレームレートを取得できません: {media_path}")


def validate_audio_file(
    path: Path, *, expected_samples: int | None = None,
    sample_rate: int = TTS_SAMPLE_RATE, channels: int = TTS_CHANNELS,
) -> bool:
    """Decode audio with bounded memory before trusting a render/cache file."""
    try:
        with sf.SoundFile(path) as source:
            if source.samplerate != sample_rate or source.channels != channels or source.frames <= 0:
                return False
            if expected_samples is not None and source.frames != expected_samples:
                return False
            decoded = sum(len(block) for block in source.blocks(blocksize=65536))
            return decoded == source.frames
    except (RuntimeError, OSError, ValueError):
        return False


def validate_media_file(
    path: Path, *, require_video: bool = False, require_audio: bool = False,
    expected_duration: float | None = None, duration_tolerance: float = 0.1,
) -> bool:
    """Check required streams, duration and complete decoding before publication."""
    try:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        result = run_cmd([
            "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
            "-of", "json", str(path),
        ])
        info = json.loads(result.stdout)
        kinds = {stream.get("codec_type") for stream in info.get("streams", [])}
        if (require_video and "video" not in kinds) or (require_audio and "audio" not in kinds):
            return False
        duration = float(info["format"]["duration"])
        if not math.isfinite(duration) or duration <= 0:
            return False
        if expected_duration is not None and abs(duration - expected_duration) > duration_tolerance:
            return False
        run_cmd(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"])
        return True
    except (PipelineError, OSError, ValueError, KeyError, TypeError):
        return False


def _run_atomic_media(
    command: list[str], output: Path, *, video: bool = False, audio: bool = False,
    duration: float | None = None, tolerance: float = 0.1,
) -> None:
    """Publish a media result only after FFmpeg and full decoding succeed."""
    ensure_dir(output.parent)
    with TemporaryDirectory(prefix=".media-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        run_cmd([*command[:-1], str(staged)])
        if not validate_media_file(
            staged, require_video=video, require_audio=audio,
            expected_duration=duration, duration_tolerance=tolerance,
        ):
            raise PipelineError(f"生成メディアの検証に失敗しました: {output}")
        staged.replace(output)


def _run_atomic_audio(command: list[str], output: Path, samples: int) -> None:
    ensure_dir(output.parent)
    with TemporaryDirectory(prefix=".audio-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        run_cmd([*command[:-1], str(staged)])
        if not validate_audio_file(staged, expected_samples=samples):
            raise PipelineError(f"生成音声のサンプル数/デコード検証に失敗しました: {output}")
        staged.replace(output)

def extract_audio_segment(
    video_in: Path,
    out_wav: Path,
    *,
    start: float,
    end: float,
    sample_rate: int = 44100,
    channels: int = 1,
) -> None:
    """動画の指定区間から音声のみを抽出する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_wav.parent)
    duration = max(0.0, end - start)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.6f}",
        "-i", str(video_in),
        "-vn",
        "-t", f"{duration:.6f}",
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run_cmd(cmd)


def extract_concatenated_audio_segments(
    media_in: Path,
    out_wav: Path,
    *,
    ranges: list[tuple[float, float]],
    sample_rate: int = 48000,
    channels: int = 1,
) -> None:
    """Extract multiple source ranges and concatenate them into one PCM WAV."""
    valid_ranges = [
        (max(0.0, float(start)), max(0.0, float(end)))
        for start, end in ranges
        if float(end) > float(start)
    ]
    if not valid_ranges:
        raise PipelineError("連結対象の音声区間が空です。")

    if len(valid_ranges) == 1:
        start, end = valid_ranges[0]
        extract_audio_segment(
            media_in,
            out_wav,
            start=start,
            end=end,
            sample_rate=sample_rate,
            channels=channels,
        )
        return

    which_or_raise("ffmpeg")
    ensure_dir(out_wav.parent)

    select_expression = "+".join(
        f"between(t\\,{start:.6f}\\,{end:.6f})"
        for start, end in valid_ranges
    )
    target_duration = sum(end - start for start, end in valid_ranges)
    audio_filter = (
        "asetpts=PTS-STARTPTS,"
        f"aselect={select_expression},"
        "asetpts=N/SR/TB,"
        f"atrim=duration={target_duration:.6f},"
        "asetpts=PTS-STARTPTS"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", str(media_in),
        "-af", audio_filter,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(out_wav),
    ]
    run_cmd(cmd)


def build_atempo_filter(speed_factor: float) -> str:
    """atempoフィルタ文字列を生成する。"""
    if speed_factor <= 0:
        raise PipelineError(f"不正な speed_factor: {speed_factor}")

    factors: list[float] = []
    x = float(speed_factor)

    while x > 2.0:
        factors.append(2.0)
        x /= 2.0
    while x < 0.5:
        factors.append(0.5)
        x /= 0.5

    factors.append(x)
    return ",".join([f"atempo={f:.6f}" for f in factors])


def _safe_trim_range(start: float, end: float) -> tuple[float, float]:
    """trim区間の最低幅を確保する。"""
    s = max(0.0, float(start))
    e = max(s, float(end))
    if e - s < _MIN_TRIM_SEC:
        e = s + _MIN_TRIM_SEC
    return s, e


def encode_video_chunk_ts(
    video_in: Path, out_ts: Path, *, start: float, end: float, speed: float,
    target_frames: int | None = None, frame_rate: float | None = None,
) -> None:
    """Render an exact frame budget; callers share cumulative frame boundaries."""
    which_or_raise("ffmpeg")
    rate = frame_rate if frame_rate is not None else ffprobe_video_frame_rate(video_in)
    if not math.isfinite(rate) or rate <= 0:
        raise PipelineError("動画フレームレートは正の有限値が必要です。")
    duration = max(0.0, float(end) - float(start))
    spd = max(1e-6, float(speed))
    frames = target_frames if target_frames is not None else max(1, round(duration / spd * rate))
    if frames < 1:
        raise PipelineError("映像チャンクは1フレーム以上必要です。")
    cadence = str(Fraction(rate).limit_denominator(1000000))
    seek = max(0.0, float(start))
    if duration < 1.0 / rate:
        # End-of-media simultaneous turns hold the last available boundary frame.
        seek = min(seek, max(0.0, ffprobe_duration_sec(video_in) - 1.0 / rate))
        source_filter = "trim=end_frame=1,setpts=PTS-STARTPTS"
    else:
        source_filter = f"trim=end={duration:.9f},setpts=(PTS-STARTPTS)/{spd:.12f}"
    vf = (
        f"{source_filter},fps=fps={cadence}:start_time=0:round=near,"
        f"tpad=stop_mode=clone:stop=-1,trim=end_frame={frames},"
        f"setpts=N/({cadence}*TB),scale=-2:{OUTPUT_SIZE}"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", f"{seek:.9f}", "-i", str(video_in), "-an", "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        "-r", cadence, "-fps_mode", "cfr", "-frames:v", str(frames), "-bf", "0",
        "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-mpegts_flags", "+resend_headers", "-muxdelay", "0", "-muxpreload", "0",
        "-f", "mpegts", str(out_ts),
    ]
    _run_atomic_media(cmd, out_ts, video=True, duration=frames / rate, tolerance=1.0 / rate + 1e-5)


def encode_original_audio_chunk_flac(
    video_in: Path, out_flac: Path, *, start: float, end: float, speed: float,
    target_samples: int | None = None,
) -> None:
    """Retime background and normalize it to the shared exact sample budget."""
    which_or_raise("ffmpeg")
    duration = max(0.0, float(end) - float(start))
    spd = max(1e-6, float(speed))
    samples = target_samples if target_samples is not None else round(duration / spd * TTS_SAMPLE_RATE)
    if samples < 1:
        raise PipelineError("音声チャンクは1サンプル以上必要です。")
    if duration <= 1e-9:
        create_silence_flac(out_flac, samples / TTS_SAMPLE_RATE, target_samples=samples)
        return
    atempo = build_atempo_filter(spd)
    af = (
        f"atrim=start=0:end={duration:.9f},asetpts=PTS-STARTPTS,{atempo},"
        f"aresample={TTS_SAMPLE_RATE}:async=1:first_pts=0,"
        f"apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB,"
        # A precise seek can produce a tiny first frame rejected by FFmpeg 9 FLAC.
        f"asetnsamples=n={_FLAC_FRAME_SAMPLES}:p=0"
    )
    cmd = [
        "ffmpeg", "-y", "-ss", f"{max(0.0, start):.9f}", "-i", str(video_in), "-vn",
        "-af", af, "-ac", str(TTS_CHANNELS), "-ar", str(TTS_SAMPLE_RATE),
        "-c:a", "flac", str(out_flac),
    ]
    _run_atomic_audio(cmd, out_flac, samples)


def create_silence_flac(
    out_flac: Path, duration_sec: float, *, target_samples: int | None = None,
) -> None:
    """Generate silence with the same exact sample-boundary policy as speech."""
    which_or_raise("ffmpeg")
    samples = target_samples if target_samples is not None else round(max(0.0, duration_sec) * TTS_SAMPLE_RATE)
    if samples < 1:
        raise PipelineError("無音チャンクは1サンプル以上必要です。")
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi", "-i", f"anullsrc=r={TTS_SAMPLE_RATE}:cl=stereo",
        "-af", f"atrim=end_sample={samples},asetnsamples=n={_FLAC_FRAME_SAMPLES}:p=0",
        "-ac", str(TTS_CHANNELS), "-ar", str(TTS_SAMPLE_RATE), "-c:a", "flac", str(out_flac),
    ]
    _run_atomic_audio(cmd, out_flac, samples)


def concat_ts_files(in_files: list[Path], out_ts: Path, list_file: Path) -> None:
    """TSファイルを結合する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_ts.parent)
    ensure_dir(list_file.parent)

    lines = []
    for p in in_files:
        lines.append(f"file '{ffmpeg_concat_quote(str(p.resolve()))}'")
    atomic_write_text(list_file, "\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(out_ts),
    ]
    _run_atomic_media(cmd, out_ts, video=True)


def concat_audio_to_flac(
    in_files: list[Path], out_flac: Path, list_file: Path
) -> None:
    """Decode normalized audio independently and join PCM with bounded memory."""
    if not in_files:
        raise PipelineError("連結対象の音声ファイルが空です。")
    ensure_dir(out_flac.parent)
    ensure_dir(list_file.parent)
    lines = [f"file '{ffmpeg_concat_quote(str(p.resolve()))}'" for p in in_files]
    atomic_write_text(list_file, "\n".join(lines) + "\n", encoding="utf-8")

    # Every pipeline chunk is normalized before reaching this stage. Separate
    # decoders are required because silence and TTS can have different FLAC
    # STREAMINFO block sizes even when their rate and channels are identical.
    from xlanguage_dubbing.core.progress import video_signature

    expected_frames = 0
    for path in in_files:
        info = sf.info(str(path))
        if info.samplerate != TTS_SAMPLE_RATE or info.channels != TTS_CHANNELS:
            raise PipelineError(f"連結音声のサンプルレート/チャンネル数が不一致: {path}")
        if info.frames <= 0:
            raise PipelineError(f"連結音声が空です: {path}")
        expected_frames += info.frames

    # Older versions left a nonempty, truncated FLAC after a failed concat.
    # Its mere existence must never make resume skip the complete track.
    identities = [video_signature(path) for path in in_files]
    manifest_path = out_flac.with_suffix(".concat.json")
    manifest = load_json_if_exists(manifest_path)
    if out_flac.is_file() and isinstance(manifest, dict):
        try:
            info = sf.info(str(out_flac))
            if (
                info.frames == expected_frames
                and info.samplerate == TTS_SAMPLE_RATE
                and info.channels == TTS_CHANNELS
                and manifest.get("inputs") == identities
                and manifest.get("output") == video_signature(out_flac)
            ):
                return
        except (RuntimeError, OSError):
            pass

    with TemporaryDirectory(prefix=".concat-", dir=out_flac.parent) as temp_dir:
        staged = Path(temp_dir) / "audio.flac"
        with sf.SoundFile(
            str(staged), mode="w", samplerate=TTS_SAMPLE_RATE,
            channels=TTS_CHANNELS, format="FLAC", subtype="PCM_24",
        ) as output:
            for path in in_files:
                with sf.SoundFile(str(path)) as source:
                    frames_read = 0
                    for block in source.blocks(blocksize=65536, dtype="int32", always_2d=True):
                        output.write(block)
                        frames_read += len(block)
                    if frames_read != source.frames:
                        raise PipelineError(f"連結音声が途中で切れています: {path}")
        if sf.info(str(staged)).frames != expected_frames:
            raise PipelineError("結合音声のサンプル数が一致しません。")
        staged.replace(out_flac)
    atomic_write_json(manifest_path, {"inputs": identities, "output": video_signature(out_flac)})


def remux_ts_to_mp4(video_ts: Path, out_mp4: Path) -> None:
    """TSをMP4にリマックスする。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_mp4.parent)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_ts),
        "-c", "copy",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    _run_atomic_media(cmd, out_mp4, video=True)


def mux_retimed_video_with_tracks(
    video_in_mp4: Path,
    dubbed_flac: Path,
    out_mp4: Path,
    *,
    original_flac: Path | None,
    original_volume: float | None = None,
) -> None:
    """リタイム済み映像に音声を合成する。"""
    which_or_raise("ffmpeg")
    ensure_dir(out_mp4.parent)
    bg_volume = ORIGINAL_VOLUME if original_volume is None else float(original_volume)

    if original_flac and original_flac.exists():
        filter_complex = (
            f"[1:a]volume={DUBBED_VOLUME}[dubbed];"
            f"[2:a]volume={bg_volume}[orig];"
            f"[dubbed][orig]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_in_mp4),
            "-i", str(dubbed_flac),
            "-i", str(original_flac),
            "-filter_complex", filter_complex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_in_mp4),
            "-i", str(dubbed_flac),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-movflags", "+faststart",
            str(out_mp4),
        ]
    _run_atomic_media(
        cmd, out_mp4, video=True, audio=True, duration=sf.info(dubbed_flac).duration,
    )
