<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_ja.md">日本語</a></th>
      <th style="text-align:center"><a href="README.md">English</a></th>
    </tr>
  </thead>
</table>

# xlanguage-dubbing

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

`xlanguage-dubbing` converts videos into dubbed videos in another language. It extracts speech, translates the transcript locally, generates dubbed speech with one of several TTS engines, retimes the video around the generated speech, and mixes the dubbed voice back with either the original audio or a Demucs-separated background track.

The project is built for local Apple Silicon workflows and is currently tested on macOS with MPS/CPU inference.

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Engine Notes](#engine-notes)
- [Resume and Outputs](#resume-and-outputs)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Features

- Batch-process videos from a configured folder, or prompt for a single file when the folder is empty.
- Supports multilingual ASR through VibeVoice-ASR, with optional whisper.cpp mode.
- Uses CAT-Translate for English/Japanese pairs and TranslateGemma for other language pairs.
- Offers three TTS modes:
  - `omnivoice`: default voice-cloning engine.
  - `voxcpm2`: 30-language VoxCPM2 voice cloning with reference audio and transcript.
  - `kokoro-fastapi`: fast fixed-voice English to Japanese mode using Kokoro-FastAPI.
- Optional Demucs vocal/background separation before ASR and TTS reference extraction.
- Per-video checkpointing so interrupted jobs can resume.
- Segment-level retiming and final audio/video muxing with FFmpeg.

## Requirements

- macOS on Apple Silicon.
- Python 3.13 or newer.
- Homebrew packages:

```bash
brew install ffmpeg cmake uv
```

Linux and CUDA workflows are not supported by this repository configuration.

## Installation

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
uv sync
cp .env.example .env
```

Edit `.env` before running. At minimum, set `VIDEO_FOLDER`, `INPUT_LANG`, `OUTPUT_LANG`, `ASR_ENGINE`, and `TTS_ENGINE` for your job.

For whisper.cpp ASR, run the optional setup script:

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

For Kokoro-FastAPI TTS, clone the server checkout at the configured path:

```bash
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI
uv run python -m unidic download
```

The default `KOKORO_FASTAPI_DIR=./Kokoro-FastAPI` expects that checkout inside this repository root.

## Usage

Put supported videos in `VIDEO_FOLDER` and run:

```bash
uv run xlanguage-dubbing
```

Supported input extensions are `.mp4`, `.mkv`, `.mov`, `.webm`, and `.m4v`.

If `VIDEO_FOLDER` has no videos, the CLI asks for a direct video file path.

To generate a helper server script:

```bash
uv run xlanguage-dubbing --generate-script
```

## Configuration

All runtime settings are read from `.env`. See [.env.example](.env.example) for the full list.

| Setting | Purpose |
|---|---|
| `VIDEO_FOLDER` | Folder scanned for source videos. |
| `TEMP_ROOT` | Checkpoint and intermediate output directory. |
| `OUTPUT_SUFFIX` | Suffix added to generated videos. |
| `INPUT_LANG` | Source audio language, or `auto`. |
| `OUTPUT_LANG` | Dubbed output language. |
| `ASR_ENGINE` | `vibevoice` or `whisper`. |
| `ENABLE_AUDIO_SEPARATION` | Use Demucs `vocals` / `no_vocals` stems when `true`. |
| `TTS_ENGINE` | `omnivoice`, `voxcpm2`, or `kokoro-fastapi`. |
| `HF_AUTH_TOKEN` | Hugging Face token used by gated or authenticated model downloads. |
| `ORIGINAL_VOLUME` | Original-audio volume only when audio separation is disabled. |
| `DUBBED_VOLUME` | Dubbed voice volume in the final mix. |

When `ENABLE_AUDIO_SEPARATION=true`, Demucs writes separated stems and the final mix uses the separated background at full volume. When it is `false`, the pipeline uses the original media audio for ASR, reference extraction, and final background mixing.

## Engine Notes

### ASR

| Engine | Use when | Notes |
|---|---|---|
| `vibevoice` | You want multilingual or code-switched ASR. | Default mode. Uses MLX models on Apple Silicon. |
| `whisper` | You want faster single-language ASR. | Requires `scripts/setup_whisper.sh` and whisper.cpp assets. |

### TTS

| Engine | Use when | Notes |
|---|---|---|
| `omnivoice` | You want the default voice-cloning path. | Uses speaker reference audio. |
| `voxcpm2` | You want VoxCPM2 Ultimate Cloning behavior. | Uses reference audio plus transcript context. |
| `kokoro-fastapi` | You want fast English to Japanese dubbing. | Fixed Japanese voice, no cloning, skips speaker identification. |

Kokoro-FastAPI mode is intentionally limited to `INPUT_LANG=auto` or English and `OUTPUT_LANG=ja`. The default voice is `jf_alpha`.

### Translation

| Language pair | Engine |
|---|---|
| English to Japanese | CAT-Translate |
| Japanese to English | CAT-Translate |
| Other pairs | TranslateGemma |

## Resume and Outputs

The pipeline saves checkpoints under `TEMP_ROOT`, so rerunning the command resumes completed work where possible.

Temporary directories are separated by audio mode:

- `temp/<video>` when Demucs separation is enabled.
- `temp/<video>_rawaudio` when separation is disabled.

To force a full rerun for one video, remove that video's temporary directory.

## Troubleshooting

### `demucs` is missing

Run:

```bash
uv sync
```

If the environment was created before Demucs was added to the project dependencies, resyncing refreshes it.

### Kokoro-FastAPI exits during startup

Use the local checkout expected by `KOKORO_FASTAPI_DIR`, then make sure UniDic has been prepared:

```bash
cd Kokoro-FastAPI
uv run python -m unidic download
```

This project starts Kokoro-FastAPI without inheriting the parent `.venv`, sets Japanese warmup defaults for `jf_alpha`, and sends `lang_code=j` for Japanese speech requests. The local Kokoro-FastAPI checkout also avoids routing Japanese chunk sizing through the English eSpeak phonemizer.

### Whisper mode cannot find whisper.cpp

Run:

```bash
./scripts/setup_whisper.sh
```

Then confirm `WHISPER_CPP_DIR=./whisper.cpp` in `.env`.

## Development

Run a syntax check for the package:

```bash
uv run python -m compileall src
```

There is no repository unit test suite configured yet. For functional verification, run `uv run xlanguage-dubbing` on a short video sample after changing pipeline, ASR, translation, TTS, or FFmpeg behavior.

To verify the supported ASR, audio-separation, and TTS configuration matrix against the fixed sample video at `input_videos/test.mp4`, run:

```bash
./scripts/run_config_matrix.py --clean
```

The matrix runner fixes `INPUT_LANG=en` and `OUTPUT_LANG=ja`, runs every combination of `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, and `TTS_ENGINE`, and writes per-case logs plus `summary.json` under `temp/config_matrix/`. To inspect the planned combinations without launching the full media pipeline, run:

```bash
./scripts/run_config_matrix.py --dry-run
```

## License

MIT License. See [LICENSE](LICENSE).

External models, model weights, and third-party tools keep their own licenses and usage terms.
