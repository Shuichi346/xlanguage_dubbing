# ja-dubbing

A tool that converts English videos into Japanese dubbing videos while preserving the original speaker's voice characteristics.

## Features

- **Two ASR engines**: Switch between pywhispercpp (fast, English-focused) and VibeVoice-ASR (multilingual, built-in speaker separation)
- **Speaker separation**: Identify who is speaking with pyannote.audio (Whisper mode), VibeVoice-ASR has built-in speaker separation
- **Voice cloning**: Generate Japanese speech that reproduces the original speaker's voice characteristics using MioTTS-Inference
- **High-quality translation**: English-to-Japanese translation using plamo-translate-cli (PLaMo-2-Translate, MLX)
- **Video speed adjustment**: Achieve natural dubbing by stretching/compressing the video without changing audio speed

## System Requirements

- macOS (Apple Silicon) — Tested on Mac mini M4 (24GB)
- Python 3.13+
- ffmpeg / ffprobe
- Ollama (for MioTTS LLM backend)

> **Note**: Linux compatibility has not been verified at this time. The translation engine (plamo-translate-cli) uses MLX, so Apple Silicon Mac is required.

## Prerequisites Installation

Before setup, ensure the following tools are installed.

### Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ffmpeg

```bash
brew install ffmpeg
```

### CMake (required for building sentencepiece, a dependency of plamo-translate-cli)

```bash
brew install cmake
```

### uv (Python package manager)

```bash
brew install uv
```

> uv is a fast Python package manager that replaces pip. This tool uses it for all Python operations.

### Ollama

Download and install the macOS version from https://ollama.com/download.

> Ollama is used as the LLM backend for MioTTS. It is not used for translation.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/ja-dubbing.git
cd ja-dubbing
```

### 2. HuggingFace Preparation

The pyannote.audio speaker separation model is a gated model that requires agreement to terms of use. **If using Whisper mode (`ASR_ENGINE=whisper`)**, complete the following steps in advance. This step is not necessary if only using VibeVoice mode.

1. Create an access token at https://huggingface.co/settings/tokens (Read permission is sufficient)
2. Open the following two model pages and **agree to the terms of use** to request access:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

> Requests are usually **approved immediately**. Model downloads will fail without approval.

### 3. Install Dependencies

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> Note: VibeVoice-ASR is a large Apple Silicon-specific package (9B parameter model). The model will be automatically downloaded on first run (approximately 5GB).

### 4. Setup MioTTS-Inference

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

### 5. Create Configuration File

```bash
cp .env.example .env
```

Open `.env` and edit the following required items:

| Item | Description | Example |
|------|-------------|---------|
| `VIDEO_FOLDER` | Folder containing input videos | `./input_videos` |
| `HF_AUTH_TOKEN` | HuggingFace token (required for Whisper mode) | `hf_xxxxxxxxxxxx` |
| `ASR_ENGINE` | ASR engine selection | `whisper` or `vibevoice` |

Other configuration items will work with default values. See "Configuration Items" below for details.

Place English video files (.mp4, .mkv, .mov, .webm, .m4v) that you want to dub in the `VIDEO_FOLDER`.

## ASR Engine Selection

Switch between speech recognition engines using `ASR_ENGINE` in `.env`.

| Item | `whisper` | `vibevoice` |
|------|-----------|-------------|
| Engine | pywhispercpp (whisper.cpp) | VibeVoice-ASR (Microsoft, mlx-audio) |
| Speaker separation | pyannote.audio (separate step) | Built-in (single pass) |
| Speed | Fast | Slow (several times slower than Whisper) |
| Language support | English-focused | Strong with multilingual mixed content |
| Audio limit | No limit | Approximately 60 minutes |
| Additional dependencies | None (included in standard installation) | `mlx-audio[stt]>=0.3.0` |
| HuggingFace token | Required (for pyannote) | Not required |

### Whisper Mode (Default)

```env
ASR_ENGINE=whisper
```

Performs fast transcription with pywhispercpp and speaker separation with pyannote.audio. Optimal for English audio processing.

### VibeVoice Mode

```env
ASR_ENGINE=vibevoice
```

Microsoft's VibeVoice-ASR outputs transcription, speaker separation, and timestamps in a single pass. Strong with multilingual mixed audio (English + local language, etc.) but slower than Whisper.

In VibeVoice mode, pyannote.audio and HuggingFace tokens are not required. Steps 3 (speaker separation) and 4 (speaker ID assignment) are automatically skipped.

#### VibeVoice-ASR Specific Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | Model to use |
| `VIBEVOICE_MAX_TOKENS` | `65536` | Maximum generation tokens |
| `VIBEVOICE_CONTEXT` | (empty) | Hot words (proper nouns recognition aid, comma-separated) |

Hot words example:

```env
VIBEVOICE_CONTEXT=MLX, Apple Silicon, PyTorch, Transformer
```

## Server Startup

This tool uses three internal servers. **All must be started before running the pipeline.**

### Method A: Using startup script (recommended)

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

PLaMo-2-Translate MLX models and MioTTS Ollama models will be automatically downloaded on first run.

### Method B: Manual startup

Open three terminals and run the following in each:

**Terminal 1: plamo-translate-cli translation server (MLX, 8bit)**

```bash
uv run plamo-translate server --precision 8bit
```

> On first startup, the `mlx-community/plamo-2-translate-8bit` model will be automatically downloaded. Wait for the download to complete before running `uv run ja-dubbing` below.

**Terminal 2: MioTTS LLM backend (port 8000)**

```bash
OLLAMA_HOST=localhost:8000 ollama serve
# First time only: OLLAMA_HOST=localhost:8000 ollama pull hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf
```

**Terminal 3: MioTTS API server (port 8001)**

```bash
cd MioTTS-Inference
uv run python run_server.py \
    --llm-base-url http://localhost:8000/v1 \
    --device mps \
    --max-text-length 500 \
    --port 8001
```

> **Port configuration**: Translation uses plamo-translate-cli with automatic port management via MCP protocol. MioTTS LLM Ollama (8000) port is specified by the `OLLAMA_HOST` environment variable.

## Execution

With all servers running, execute from another terminal:

```bash
uv run ja-dubbing
```

Videos in `VIDEO_FOLDER` are automatically detected and processed sequentially. Output is saved as `*_jaDub.mp4` in the same folder as the input video. Specify any `VIDEO_FOLDER` path in `.env`.

## Processing Flow

### Whisper Mode (`ASR_ENGINE=whisper`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. English transcription with pywhispercpp (large-v3-turbo)
3. Speaker separation with pyannote.audio
4. Assign speaker IDs to Whisper segments
5. Segment merging → spaCy sentence splitting → translation unit merging (maintaining speaker boundaries)
6. Extract reference audio for each speaker from original video
7. English-to-Japanese translation with plamo-translate-cli (PLaMo-2-Translate, MLX 8bit)
8. Generate speaker-cloned Japanese speech with MioTTS-Inference
9. Speed-adjust each section of original video to match TTS audio length
10. Compose video + Japanese audio (+ lightly mixed English audio) for output

### VibeVoice Mode (`ASR_ENGINE=vibevoice`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. Transcription + speaker separation + timestamp acquisition with VibeVoice-ASR (single pass)
3. (Skipped: built into VibeVoice-ASR)
4. (Skipped: built into VibeVoice-ASR)
5. Segment merging → spaCy sentence splitting → translation unit merging (maintaining speaker boundaries)
6. Extract reference audio for each speaker from original video
7. English-to-Japanese translation with plamo-translate-cli (PLaMo-2-Translate, MLX 8bit)
8. Generate speaker-cloned Japanese speech with MioTTS-Inference
9. Speed-adjust each section of original video to match TTS audio length
10. Compose video + Japanese audio (+ lightly mixed English audio) for output

## Resume Function

Processing saves checkpoints at each step. If interrupted, re-execution will resume from where it left off. Checkpoints are saved in `temp/<video_name>/progress.json`.

To start over from the beginning, delete the corresponding `temp/<video_name>/` folder before re-executing.

> **Note when switching ASR engines**: To redo a video that was partially processed in Whisper mode using VibeVoice mode, delete the `temp/<video_name>/` folder before re-executing.

## Configuration Items

All settings are managed in the `.env` file. All configuration items and default values are listed in `.env.example`. Main configuration items are as follows:

| Setting | Default | Description |
|---------|---------|-------------|
| `VIDEO_FOLDER` | `./input_videos` | Input video folder |
| `TEMP_ROOT` | `./temp` | Temporary files folder |
| `ASR_ENGINE` | `whisper` | ASR engine (`whisper` or `vibevoice`) |
| `HF_AUTH_TOKEN` | (Required for Whisper mode) | HuggingFace token |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | VibeVoice-ASR model name |
| `VIBEVOICE_MAX_TOKENS` | `65536` | VibeVoice-ASR maximum tokens |
| `VIBEVOICE_CONTEXT` | (empty) | VibeVoice-ASR hot words |
| `PLAMO_TRANSLATE_PRECISION` | `8bit` | Translation model precision (4bit / 8bit / bf16) |
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_DEVICE` | `mps` | MioTTS codec device |
| `ENGLISH_VOLUME` | `0.10` | English audio volume (0.0-1.0) |
| `JAPANESE_VOLUME` | `1.00` | Japanese audio volume (0.0-1.0) |
| `OUTPUT_SIZE` | `720` | Output video height (pixels) |
| `KEEP_TEMP` | `true` | Whether to keep temporary files |

## Known Limitations

- **English-Japanese mixed content**: When English words are mixed in like "utilizing simulation tools such as Omniverse and ISACsim", TTS may not pronounce them correctly.
- **Processing time**: Even 3-minute videos can take tens of minutes. Long videos may cause errors, so it's recommended to **split them into approximately 8-minute segments** before processing.
- **Translation quality**: Being local LLM-based translation, accuracy varies compared to cloud APIs.
- **VibeVoice-ASR processing speed**: Takes several times longer than Whisper. Cannot process audio longer than 60 minutes.
- **VibeVoice-ASR memory usage**: Being a 9B parameter model, it uses approximately 5GB of memory even with 8bit quantization. Tested on Mac with 24GB unified memory.

## Troubleshooting

### Out of Memory

Tested on Mac mini M4 with 24GB unified memory. To save memory, ASR models (Whisper / VibeVoice-ASR) and pyannote pipelines are released after use. For long videos, set `KEEP_TEMP=true` and utilize interruption/resume functionality.

### MioTTS "Text too long" Error

MioTTS default maximum text length is 300 characters. This tool specifies `--max-text-length 500` at server startup and also performs truncation processing at punctuation marks on the pipeline side.

### VibeVoice-ASR "mlx-audio not installed" Error

Run the following:

```bash
uv pip install 'mlx-audio[stt]>=0.3.0'
```

### VibeVoice-ASR Empty Transcription Results

The audio file may not contain speech or the audio may be too short. Try increasing `VIBEVOICE_MAX_TOKENS` or switching to Whisper mode.

> plamo-translate-cli automatically manages ports via MCP protocol. Configuration files are saved in `$TMPDIR/plamo-translate-config.json`.

## License

MIT License

Note: External models and libraries used by this tool have their own individual licenses.

- MioTTS default preset: Uses audio generated by T5Gemma-TTS / Gemini TTS, not for commercial use
- PLaMo-2-Translate: PLaMo Community License (commercial use requires application)
- plamo-translate-cli: Apache-2.0 License
- pyannote.audio: MIT License (models are gated, requires agreement to terms of use on HuggingFace)
- VibeVoice-ASR: MIT License
- mlx-audio: MIT License
