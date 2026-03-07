<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_en.md">English</a></th>
      <th style="text-align:center">日本語</th>
    </tr>
  </thead>
</table>

# ja-dubbing

A tool for converting English videos into Japanese dubbed videos while maintaining the original speaker's voice characteristics.

## Features

- **Two ASR engines**: Switchable between whisper.cpp + Silero VAD (fast, English-focused, hallucination suppression) and VibeVoice-ASR (multilingual, built-in speaker diarization)
- **Speaker diarization**: Identifies who speaks what using pyannote.audio (in Whisper mode), VibeVoice-ASR has built-in speaker diarization
- **Voice cloning**: Generates Japanese speech that reproduces the original speaker's voice characteristics using MioTTS-Inference
- **High-quality translation**: English-to-Japanese translation using plamo-translate-cli (PLaMo-2-Translate, MLX)
- **Video speed adjustment**: Achieves natural dubbing by stretching/compressing the video side without changing audio speed

## System Requirements

- macOS (Apple Silicon) — Tested on Mac mini M4 (24GB)
- Python 3.13+
- ffmpeg / ffprobe
- CMake (required for whisper.cpp build)
- Ollama (for MioTTS LLM backend)

> **Note**: Linux compatibility has not been verified at this time. The translation engine (plamo-translate-cli) uses MLX, so Apple Silicon Mac is required.

## Installing Prerequisites

Before setup, ensure the following tools are installed.

### Homebrew (if not already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ffmpeg

```bash
brew install ffmpeg
```

### CMake (required for whisper.cpp build and sentencepiece dependency build for plamo-translate-cli)

```bash
brew install cmake
```

### uv (Python package manager)

```bash
brew install uv
```

> uv is a fast Python package manager that replaces pip. It's used for all Python operations in this tool.

### Ollama

Download and install the macOS version from https://ollama.com/download.

> Ollama is used as the LLM backend for MioTTS. It's not used for translation.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 2. HuggingFace preparation

The speaker diarization model for pyannote.audio is a gated model that requires agreement to terms of use. **If you plan to use Whisper mode (`ASR_ENGINE=whisper`)**, complete the following steps beforehand. This step is not required if you only use VibeVoice mode.

1. Create an access token at https://huggingface.co/settings/tokens (Read permission is sufficient)
2. Open the following two model pages and **agree to the terms of use** to request access:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

> Requests are usually **approved immediately**. Model downloads will fail if not approved.

### 3. Install dependencies

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> Note: VibeVoice-ASR is a large Apple Silicon-specific package (9B parameter model). The model will be automatically downloaded on first run (~5GB).

### 4. Build whisper.cpp (for Whisper mode)

If you plan to use Whisper mode (`ASR_ENGINE=whisper`), you need to build whisper.cpp from source and download the Whisper and VAD models. This can be done automatically with the following script:

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

This script performs the following:

1. Clone the `whisper.cpp` repository (build with Apple Silicon Metal GPU support)
2. Download Whisper model (`ggml-large-v3-turbo`)
3. Download Silero VAD model (`ggml-silero-v6.2.0`)

> This step is not required if you only use VibeVoice mode.

### 5. Set up MioTTS-Inference

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

### 6. Create configuration file

```bash
cp .env.example .env
```

Open `.env` and edit the following items:

| Item | Description | Example |
|------|-------------|---------|
| `VIDEO_FOLDER` | Folder for input videos | `./input_videos` |
| `HF_AUTH_TOKEN` | HuggingFace token (required for Whisper mode) | `hf_xxxxxxxxxxxx` |
| `ASR_ENGINE` | ASR engine selection | `whisper` or `vibevoice` |

Other configuration items will work with their defaults. See "Configuration Items" below for details.

Place English video files (.mp4, .mkv, .mov, .webm, .m4v) you want to dub in the `VIDEO_FOLDER`.

## ASR Engine Selection

You can switch the speech recognition engine with `ASR_ENGINE` in `.env`.

| Item | `whisper` | `vibevoice` |
|------|-----------|-------------|
| Engine | whisper.cpp CLI + Silero VAD | VibeVoice-ASR (Microsoft, mlx-audio) |
| Speaker diarization | pyannote.audio (separate step) | Built-in (single pass) |
| Speed | Fast | Slow (several times slower than Whisper) |
| VAD | Built-in Silero VAD (hallucination suppression) | None |
| Language support | English-focused | Strong with multilingual mixed content |
| Audio length limit | No limit | Supports long audio through memory optimization |
| Additional setup | Requires running `scripts/setup_whisper.sh` | `mlx-audio[stt]>=0.3.0` |
| HuggingFace token | Required (for pyannote) | Not required |

### Whisper mode (default)

```env
ASR_ENGINE=whisper
```

Performs fast, high-quality transcription using whisper.cpp combined with Silero VAD, followed by speaker diarization with pyannote.audio. VAD suppresses hallucination (phantom text) in silent sections. Optimal for English audio processing.

### VibeVoice mode

```env
ASR_ENGINE=vibevoice
```

Microsoft's VibeVoice-ASR outputs transcription, speaker diarization, and timestamps in a single pass. Strong with multilingual mixed audio (English + local language, etc.) but slower than Whisper.

Features built-in memory optimization through encoder chunk processing, allowing long audio processing even on 24GB unified memory Mac. Chunk size is automatically determined based on available memory, requiring no special user configuration.

VibeVoice mode does not require pyannote.audio or HuggingFace tokens. Steps 3 (speaker diarization) and 4 (speaker ID assignment) are automatically skipped.

#### VibeVoice-ASR specific settings

| Setting | Default | Description |
|---------|---------|-------------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | Model to use |
| `VIBEVOICE_MAX_TOKENS` | `32768` | Maximum generation tokens |
| `VIBEVOICE_CONTEXT` | (empty) | Hotwords (proper noun recognition assistance, comma-separated) |

Hotword example:

```env
VIBEVOICE_CONTEXT=MLX, Apple Silicon, PyTorch, Transformer
```

## Server Startup

This tool uses three internal servers. **All must be started before running the pipeline.**

### Method A: Use startup script (recommended)

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

PLaMo-2-Translate MLX model and MioTTS Ollama model will be automatically downloaded on first run.

### Method B: Start manually

Open three terminals and run the following in each:

**Terminal 1: plamo-translate-cli translation server (MLX, 8bit)**

```bash
uv run plamo-translate server --precision 8bit
```

> The `mlx-community/plamo-2-translate-8bit` model will be automatically downloaded on first startup. Wait for the download to complete before running `uv run ja-dubbing` below.

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

> **Port configuration**: Translation uses plamo-translate-cli with automatic port management via MCP protocol. MioTTS LLM Ollama (8000) specifies the port using the `OLLAMA_HOST` environment variable.

## Execution

With all servers running, execute from another terminal:

```bash
uv run ja-dubbing
```

Videos in `VIDEO_FOLDER` are automatically detected and processed sequentially. Output is saved as `*_jaDub.mp4` in the same folder as the input video. Specify any `VIDEO_FOLDER` path in `.env`.

## Processing Flow

### Whisper mode (`ASR_ENGINE=whisper`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. English transcription using whisper.cpp + Silero VAD (suppress hallucination in silent sections)
3. Speaker diarization using pyannote.audio
4. Assign speaker IDs to Whisper segments
5. Segment combination → spaCy sentence splitting → translation unit combination (maintain speaker boundaries)
6. Extract reference audio for each speaker from original video
7. English-to-Japanese translation using plamo-translate-cli (PLaMo-2-Translate, MLX 8bit)
8. Generate speaker-cloned Japanese speech using MioTTS-Inference
9. Speed-stretch each section of original video to match TTS audio length
10. Combine video + Japanese audio (+ lightly mixed English audio) for output

### VibeVoice mode (`ASR_ENGINE=vibevoice`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. Transcription + speaker diarization + timestamp acquisition using VibeVoice-ASR (single pass, memory optimization through chunk encoding)
3. (Skipped: built into VibeVoice-ASR)
4. (Skipped: built into VibeVoice-ASR)
5. Segment combination → spaCy sentence splitting → translation unit combination (maintain speaker boundaries)
6. Extract reference audio for each speaker from original video
7. English-to-Japanese translation using plamo-translate-cli (PLaMo-2-Translate, MLX 8bit)
8. Generate speaker-cloned Japanese speech using MioTTS-Inference
9. Speed-stretch each section of original video to match TTS audio length
10. Combine video + Japanese audio (+ lightly mixed English audio) for output

## Resume Function

Processing saves checkpoints at each step. If interrupted, re-execution will resume from where it left off. Checkpoints are saved in `temp/<video_name>/progress.json`.

To start over from the beginning, delete the corresponding `temp/<video_name>/` folder before re-execution.

> **Note when switching ASR engines**: To restart a video that was partially processed in Whisper mode with VibeVoice mode, delete the `temp/<video_name>/` folder before re-execution.

## Configuration Items

All settings are managed in the `.env` file. All configuration items and default values are listed in `.env.example`. Main configuration items are as follows:

| Setting | Default | Description |
|---------|---------|-------------|
| `VIDEO_FOLDER` | `./input_videos` | Input video folder |
| `TEMP_ROOT` | `./temp` | Temporary files folder |
| `ASR_ENGINE` | `whisper` | ASR engine (`whisper` or `vibevoice`) |
| `HF_AUTH_TOKEN` | (Required for Whisper mode) | HuggingFace token |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `VAD_MODEL` | `silero-v6.2.0` | VAD model name |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp installation directory |
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | VibeVoice-ASR model name |
| `VIBEVOICE_MAX_TOKENS` | `32768` | VibeVoice-ASR maximum tokens |
| `VIBEVOICE_CONTEXT` | (empty) | VibeVoice-ASR hotwords |
| `PLAMO_TRANSLATE_PRECISION` | `8bit` | Translation model precision (4bit / 8bit / bf16) |
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_DEVICE` | `mps` | MioTTS codec device |
| `ENGLISH_VOLUME` | `0.10` | English audio volume (0.0-1.0) |
| `JAPANESE_VOLUME` | `1.00` | Japanese audio volume (0.0-1.0) |
| `OUTPUT_SIZE` | `720` | Output video height (pixels) |
| `KEEP_TEMP` | `true` | Whether to keep temporary files |

## Known Limitations

- **English-Japanese mixed content**: When English words are mixed like "utilizing simulation tools such as Omniverse and ISACsim," TTS may not pronounce them correctly.
- **Processing time**: Even 3-minute videos can take tens of minutes. Long videos may cause errors, so **pre-splitting to about 8 minutes** is recommended.
- **Translation quality**: Local LLM translation has varying accuracy compared to cloud APIs.
- **VibeVoice-ASR processing speed**: Takes several times longer than Whisper.
- **VibeVoice-ASR memory usage**: As a 9B parameter model, it uses about 5GB of memory even with 8bit quantization. Built-in memory optimization through chunk encoding has been tested on 24GB unified memory Mac.

## Troubleshooting

### whisper-cli not found

Run `scripts/setup_whisper.sh` to build whisper.cpp.

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

If CMake is not installed, run `brew install cmake` first.

### Out of memory

Tested on Mac mini M4 with 24GB unified memory. For memory conservation, ASR models (Whisper / VibeVoice-ASR) and pyannote pipeline are released after use. VibeVoice mode suppresses memory spikes in long audio through encoder chunk processing. For long videos, set `KEEP_TEMP=true` and utilize interruption/resumption.

### MioTTS text too long error

MioTTS default maximum text length is 300 characters. This tool specifies `--max-text-length 500` at server startup and also performs truncation processing at punctuation marks on the pipeline side.

### VibeVoice-ASR "mlx-audio not installed" error

Run the following:

```bash
uv pip install 'mlx-audio[stt]>=0.3.0'
```

### VibeVoice-ASR returns empty transcription results

The audio file may not contain speech or the audio may be too short. Try increasing `VIBEVOICE_MAX_TOKENS` or switching to Whisper mode.

> plamo-translate-cli automatically manages ports via MCP protocol. Configuration files are saved in `$TMPDIR/plamo-translate-config.json`.

## License

MIT License

Note: External models and libraries used by this tool have their own respective licenses.

- MioTTS default preset: Uses audio generated by T5Gemma-TTS / Gemini TTS, not for commercial use
- PLaMo-2-Translate: PLaMo Community License (commercial use requires application)
- plamo-translate-cli: Apache-2.0 License
- pyannote.audio: MIT License (models are gated, requires agreement to terms of use on HuggingFace)
- whisper.cpp: MIT License
- Silero VAD: MIT License
- VibeVoice-ASR: MIT License
- mlx-audio: MIT License