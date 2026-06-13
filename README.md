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

`xlanguage-dubbing` is a local video dubbing pipeline for Apple Silicon Macs. It transcribes source speech, detects or uses the source language, translates the transcript, generates cloned speech in the target language, retimes the video around the generated speech, and muxes the dubbed voice with either the original audio or a Demucs-separated background stem.

The repository is aimed at local media workflows where model inference, translation, TTS, checkpointing, and final rendering all run on the user's machine.

## Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
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

- Batch-processes videos found recursively under `VIDEO_FOLDER`.
- Prompts for a single video path when `VIDEO_FOLDER` has no supported videos.
- Supports `.mp4`, `.mkv`, `.mov`, `.webm`, and `.m4v` inputs.
- Uses VibeVoice-ASR by default for multilingual and code-switched transcription with built-in speaker information.
- Offers whisper.cpp ASR with Silero VAD and pyannote speaker diarization as an optional mode.
- Optionally separates vocals and background audio with Demucs `--two-stems vocals`.
- Preserves the raw-audio fallback path when Demucs separation is disabled.
- Normalizes ASR segments with merge rules, spaCy sentence splitting, and sentence-unit merging.
- Selects CAT-Translate for English/Japanese pairs and TranslateGemma for other language pairs.
- Provides three TTS engines: OmniVoice, VoxCPM2, and Irodori-TTS-Server.
- Builds speaker and per-segment reference audio for cloned speech synthesis.
- Retimes the source video to match generated TTS durations.
- Saves resumable checkpoints and intermediate artifacts per video.
- Includes a configuration-matrix runner for supported ASR, audio-source, and TTS combinations.

## How It Works

1. Probe the input video with `ffprobe`.
2. Prepare the ASR/reference audio:
   - `ENABLE_AUDIO_SEPARATION=true`: Demucs creates `vocals.wav` and `no_vocals.wav`.
   - `ENABLE_AUDIO_SEPARATION=false`: the original media audio is used directly.
3. Transcribe speech with `ASR_ENGINE`.
4. Assign or reuse speaker IDs, then merge and split segments.
5. Detect the source language when `INPUT_LANG=auto`.
6. Translate each segment into `OUTPUT_LANG`.
7. Generate cloned TTS audio per translated segment.
8. Retiming splits the source video and background audio into chunks that match TTS timing.
9. FFmpeg muxes the retimed video, dubbed track, and background/original track into the final MP4.

## Tech Stack

- Python 3.13 package managed by `uv` and built with Hatchling.
- FFmpeg and ffprobe for media probing, transcoding, retiming, and muxing.
- Demucs for optional vocal/background separation.
- spaCy for sentence-aware segment splitting.
- VibeVoice-ASR via `mlx-audio` / MLX for the default ASR path.
- whisper.cpp, Silero VAD, and pyannote.audio for the optional whisper ASR path.
- `llama-cpp-python` and Hugging Face Hub for local GGUF translation models.
- OmniVoice, VoxCPM2, and Irodori-TTS-Server for voice-cloned TTS.
- PyTorch, torchaudio, TorchCodec, soundfile, and pydub for audio/model support.

## Requirements

- macOS on Apple Silicon.
- Python 3.13 or newer.
- Homebrew packages:

```bash
brew install ffmpeg cmake uv
```

- Internet access for first-time dependency and model downloads.
- A Hugging Face token in `HF_AUTH_TOKEN` when using whisper mode with pyannote diarization, and for any gated or authenticated model downloads.

Linux, Windows, and CUDA workflows are not supported by this repository configuration.

## Installation

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
uv sync
uv run python -m spacy download en_core_web_sm
cp .env.example .env
```

Edit `.env` before running. At minimum, set `VIDEO_FOLDER`, `INPUT_LANG`, `OUTPUT_LANG`, `ASR_ENGINE`, `TTS_ENGINE`, and any model/token settings required by your selected engines.

For whisper.cpp ASR, run the optional setup script:

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

The script clones or updates `whisper.cpp`, builds `whisper-cli`, and downloads the configured Whisper and VAD models.

For Irodori TTS, clone the server checkout at the configured path and sync the CPU extra:

```bash
git clone https://github.com/Aratako/Irodori-TTS-Server.git
cd Irodori-TTS-Server
uv sync --extra cpu
```

The default `IRODORI_TTS_DIR=./Irodori-TTS-Server` expects that checkout inside this repository root. With `IRODORI_TTS_AUTO_START=true`, the pipeline starts the server automatically when `TTS_ENGINE=irodori`.

## Usage

Put supported videos in `VIDEO_FOLDER` and run:

```bash
uv run xlanguage-dubbing
```

The CLI scans `VIDEO_FOLDER` recursively. If no supported video is found, it asks for a direct video file path.

To generate a helper server script:

```bash
uv run xlanguage-dubbing --generate-script
```

When `TTS_ENGINE=irodori`, the generated `start_servers.sh` starts Irodori-TTS-Server with the configured host and port. For OmniVoice and VoxCPM2, it reports that no external server is needed.

## Configuration

All runtime settings are read from `.env`. See [.env.example](.env.example) for the full list.

### Core Settings

| Setting | Purpose |
|---|---|
| `VIDEO_FOLDER` | Folder scanned recursively for source videos. |
| `TEMP_ROOT` | Checkpoint and intermediate output root. |
| `OUTPUT_SUFFIX` | Suffix appended to generated videos. |
| `KEEP_TEMP` | Keep or delete each video's temporary working directory after muxing. |
| `INPUT_LANG` | Source language ISO code, or `auto`. |
| `OUTPUT_LANG` | Target dubbed language ISO code. |
| `TTS_SAMPLE_RATE` | Project-standard TTS/output audio sample rate. |
| `TTS_CHANNELS` | Project-standard TTS/output audio channel count. |
| `ORIGINAL_VOLUME` | Raw original audio volume when audio separation is disabled. |
| `DUBBED_VOLUME` | Dubbed voice volume in the final mix. |
| `OUTPUT_SIZE` | Output video height used during retiming encode. |

When `ENABLE_AUDIO_SEPARATION=true`, the separated background stem is mixed at full volume. `ORIGINAL_VOLUME` only applies to raw original audio when `ENABLE_AUDIO_SEPARATION=false`.

### ASR and Audio Source

| Setting | Purpose |
|---|---|
| `ASR_ENGINE` | `vibevoice` or `whisper`. |
| `ENABLE_AUDIO_SEPARATION` | Use Demucs `vocals` / `no_vocals` stems when `true`; use original media audio when `false`. |
| `DEMUCS_MODEL` | Demucs model name, default `htdemucs`. Use `htdemucs_ft` for the fine-tuned model when quality is preferred over speed. |
| `DEMUCS_DEVICE` | Demucs inference device, default `mps` for Apple Silicon. Set `cpu` when MPS is unavailable or unstable. |
| `WHISPER_MODEL` | whisper.cpp model name used by `scripts/setup_whisper.sh` and whisper mode. |
| `WHISPER_LANG` | whisper.cpp language argument; normally follows `INPUT_LANG` or `auto`. |
| `VAD_MODEL` | whisper.cpp VAD model name. |
| `WHISPER_CPP_DIR` | Local whisper.cpp checkout path. |
| `VIBEVOICE_MODEL` | MLX VibeVoice-ASR model. |
| `VIBEVOICE_MAX_TOKENS` | Maximum VibeVoice generation tokens. |
| `VIBEVOICE_CONTEXT` | Optional context prompt for VibeVoice-ASR. |
| `HF_AUTH_TOKEN` | Hugging Face token required by pyannote diarization in whisper mode. |
| `PYANNOTE_MODEL` | pyannote diarization pipeline model. |

### Segmentation and Translation

| Setting | Purpose |
|---|---|
| `MIN_SEGMENT_SEC` | Segments shorter than this are skipped for TTS. |
| `MERGE_MAX_SEC`, `MERGE_MAX_CHARS`, `MERGE_GAP_SEC` | Initial segment merge limits. |
| `SPACY_MODEL` | spaCy model used for sentence splitting. |
| `SPACY_CHUNK_MAX_SEC`, `SPACY_CHUNK_MAX_CHARS`, `SPACY_CHUNK_GAP_SEC` | Limits for building spaCy input chunks. |
| `SPACY_UNIT_MAX_SENTENCES`, `SPACY_UNIT_MERGE_MAX_CHARS`, `SPACY_UNIT_MERGE_MAX_GAP_SEC` | Sentence-unit merge limits after spaCy splitting. |
| `CAT_TRANSLATE_*` | CAT-Translate GGUF model, context, retry, and repeat-penalty settings. |
| `TRANSLATEGEMMA_*` | TranslateGemma GGUF model, context, retry, and repeat-penalty settings. |
| `OUTPUT_REPEAT_THRESHOLD`, `INPUT_REPEAT_THRESHOLD` | Repetition thresholds used to detect translation glitches. |

### TTS Engines

| Setting | Purpose |
|---|---|
| `TTS_ENGINE` | `omnivoice`, `voxcpm2`, or `irodori`. Aliases `irodori-tts` and `irodori_tts` are accepted. |
| `OMNIVOICE_*` | OmniVoice model, dtype, sampling, duration, reference, and retry settings. |
| `VOXCPM2_*` | VoxCPM2 model, controllable cloning, duration, reference, and retry settings. |
| `IRODORI_TTS_BASE_URL` | Irodori-TTS-Server base URL. |
| `IRODORI_TTS_DIR` | Local Irodori-TTS-Server checkout. |
| `IRODORI_TTS_AUTO_START` | Start Irodori-TTS-Server automatically when it is not already healthy. |
| `IRODORI_TTS_START_COMMAND` | Optional custom server start command. |
| `IRODORI_TTS_API_KEY` | Optional API key sent to Irodori-TTS-Server. |
| `IRODORI_MODEL_DEVICE` | Device used by the Irodori model, default `cpu`. Due to a PyTorch bug, `mps` may cause increased memory usage — be cautious of processing load. |
| `IRODORI_CODEC_DEVICE` | Device used by the Irodori codec, default `cpu`. Due to a PyTorch bug, `mps` may cause increased memory usage — be cautious of processing load. |
| `IRODORI_TTS_RESPONSE_FORMAT` | Audio response format, default `wav`. |
| `IRODORI_TTS_SPEED` | Speed value sent to the Irodori speech API. |
| `IRODORI_TTS_NUM_STEPS` | Irodori diffusion steps, default `8` for Sway Sampling. |
| `IRODORI_TTS_T_SCHEDULE_MODE` | Irodori sampling schedule, default `sway`. |
| `IRODORI_TTS_SWAY_COEFF` | Irodori Sway Sampling coefficient, default `-1.0`. |

## Engine Notes

### ASR

| Engine | Use When | Notes |
|---|---|---|
| `vibevoice` | You want multilingual or code-switched ASR with integrated speaker information. | Default mode. Uses MLX models on Apple Silicon and skips the pyannote speaker diarization pass. |
| `whisper` | You want whisper.cpp ASR with VAD. | Requires `scripts/setup_whisper.sh`; speaker IDs are assigned with pyannote.audio and require `HF_AUTH_TOKEN`. |

### TTS

| Engine | Use When | Notes |
|---|---|---|
| `omnivoice` | You want the default process-internal cloned TTS path. | Uses speaker and per-segment reference audio plus reference text where available. |
| `voxcpm2` | You want VoxCPM2 Controllable Cloning behavior. | Passes per-segment `reference_wav_path` only; prompt audio/text is not sent to VoxCPM2. |
| `irodori` | You want Japanese cloned TTS through Irodori-TTS-Server. | Recommended here for English-to-Japanese jobs, only allowed with `OUTPUT_LANG=ja`, and sends per-segment reference audio as `irodori.ref_wav`. |

Irodori mode sends Sway Sampling options by default (`num_steps=8`, `t_schedule_mode=sway`, `sway_coeff=-1.0`). It intentionally does not send Caption, Style Prompt, or fixed `seconds`; the server duration predictor is used.

### Translation

| Language Pair | Engine |
|---|---|
| English to Japanese | CAT-Translate |
| Japanese to English | CAT-Translate |
| Other pairs | TranslateGemma |

Both translation engines run through local GGUF inference. The pipeline releases translation models before TTS to free memory.

## Resume and Outputs

The final video is written next to the source video as:

```text
<source-stem><OUTPUT_SUFFIX>
```

By default, that is `<source-stem>_xlDub.mp4`.

The pipeline saves checkpoints and artifacts under `TEMP_ROOT`, including:

- `progress.json`
- `segments_src.json`
- `segments_translated.json`
- `subtitles_src.srt`
- `speaker_refs/`
- `seg_audio/`
- `tts_meta.json`
- `retime/`

Temporary directories are isolated by audio source mode:

- `temp/<video>` when Demucs separation is enabled.
- `temp/<video>_rawaudio` when Demucs separation is disabled.

Rerunning the command resumes completed work when `progress.json` matches the source video's size and modification time. To force a full rerun for one video, remove that video's temporary directory and any existing final output file.

## Troubleshooting

### `demucs` is missing

Run:

```bash
uv sync
```

If the environment was created before Demucs was added to the project dependencies, resyncing refreshes it.

### spaCy cannot find `en_core_web_sm`

The pipeline initializes spaCy before preflight checks. Install the configured model:

```bash
uv run python -m spacy download en_core_web_sm
```

If you change `SPACY_MODEL` in `.env`, install that model instead.

### whisper.cpp files are missing

Run:

```bash
./scripts/setup_whisper.sh
```

Then confirm `WHISPER_CPP_DIR=./whisper.cpp` in `.env`.

### pyannote reports that `HF_AUTH_TOKEN` is unset

Set `HF_AUTH_TOKEN` in `.env` before using `ASR_ENGINE=whisper`. VibeVoice mode does not use the pyannote diarization pass.

### Irodori-TTS-Server exits during startup

Use the local checkout expected by `IRODORI_TTS_DIR`, then make sure the CPU extra has been synced:

```bash
cd Irodori-TTS-Server
uv sync --extra cpu
```

The default server start command is:

```bash
uv run python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

If startup still fails, inspect:

```text
/private/tmp/xlanguage_dubbing_irodori_tts.log
```

### Output already exists

If the target file already exists, the pipeline skips that source video. Delete the existing `<source-stem><OUTPUT_SUFFIX>` file to regenerate it.

## Development

Run a syntax check for the package:

```bash
uv run python -m compileall src
```

There is no repository unit test suite configured yet. For functional verification, run `uv run xlanguage-dubbing` on a short video sample after changing pipeline, ASR, translation, TTS, or FFmpeg behavior.

To verify the supported ASR, audio-source, and TTS configuration matrix against the fixed sample video at `input_videos/test.mp4`, run:

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
