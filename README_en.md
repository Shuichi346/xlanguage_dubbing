<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_en.md">English</a></th>
      <th style="text-align:center"><a href="README.md">日本語</a></th>
    </tr>
  </thead>
</table>

<p align="center">
  <h1 align="center">ja-dubbing</h1>
  <p align="center">A tool to convert English videos into Japanese dubbed videos while maintaining the original speaker's voice characteristics</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Prerequisites Installation](#prerequisites-installation)
- [Setup](#setup)
- [ASR Engine Selection](#asr-engine-selection)
- [TTS Engine Selection](#tts-engine-selection)
- [Server Startup](#server-startup)
- [Execution](#execution)
- [Processing Flow](#processing-flow)
- [Resume Feature](#resume-feature)
- [Configuration Settings](#configuration-settings)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **Two ASR engines**: Switch between whisper.cpp + Silero VAD (fast, English-focused, hallucination suppression) and VibeVoice-ASR (multilingual, built-in speaker diarization)
- **Speaker diarization**: Identify who is speaking when using pyannote.audio (Whisper mode), VibeVoice-ASR has built-in speaker diarization
- **Three TTS engines**: Switch between MioTTS-Inference (speaker cloning support, high quality, quality validation), GPT-SoVITS V2ProPlus (zero-shot voice cloning), and Kokoro TTS (fast, lightweight, no server required)
- **High-quality translation**: In-process English-to-Japanese translation using CAT-Translate-7b (GGUF, llama-cpp-python) with no server required
- **Video speed adjustment**: Achieve natural dubbing by stretching/compressing video without changing audio speed
- **Resume functionality**: Save checkpoints at each step and resume from where you left off if interrupted

## System Requirements

- macOS (Apple Silicon) — Tested on Mac mini M4 (24GB)
- Python 3.13+
- ffmpeg / ffprobe
- CMake (required for building whisper.cpp)
- Ollama (for MioTTS LLM backend, MioTTS mode only)
- conda (GPT-SoVITS mode only)

> **Note**: Linux compatibility is currently untested. Apple Silicon Mac is recommended as ASR (whisper.cpp, VibeVoice-ASR) and translation (CAT-Translate-7b) utilize MLX / Apple Silicon GPU.

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

### CMake (required for building whisper.cpp and dependency libraries)

```bash
brew install cmake
```

### uv (Python package manager)

```bash
brew install uv
```

> uv is a fast Python package manager that replaces pip. We use it for all Python operations in this tool.

### Ollama (required for MioTTS mode only)

Download and install the macOS version from https://ollama.com/download.

> Ollama is used as the LLM backend for MioTTS. It's not used for translation. Not required for Kokoro TTS mode and GPT-SoVITS mode.

### conda (required for GPT-SoVITS mode only)

GPT-SoVITS runs in an isolated conda environment. We recommend installing miniforge.

```bash
brew install --cask miniforge
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 2. HuggingFace preparation

The pyannote.audio speaker diarization model is a gated model that requires agreement to terms of use. **If you're using Whisper mode (`ASR_ENGINE=whisper`)**, complete the following steps beforehand. This step is not required if you're only using VibeVoice mode or only using Kokoro TTS (which doesn't require speaker diarization).

1. Create an access token at https://huggingface.co/settings/tokens (`Read` permission is sufficient)
2. Open the following two model pages and **agree to the terms of use** to request access:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

> Requests are typically **approved immediately**. Model downloads will fail if not approved.

### 3. Install dependencies

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> Note: VibeVoice-ASR is a large Apple Silicon-specific package (9B parameter model). The model will be automatically downloaded on first run (~5GB). The CAT-Translate-7b GGUF model will also be automatically downloaded via huggingface_hub on first run.

### 4. Build whisper.cpp (for Whisper mode)

If you're using Whisper mode (`ASR_ENGINE=whisper`), you need to build whisper.cpp from source and download the Whisper and VAD models. This can be done automatically with the following script:

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

This script performs the following:

1. Clone the `whisper.cpp` repository (built with Apple Silicon Metal GPU support)
2. Download Whisper model (`ggml-large-v3-turbo`)
3. Download Silero VAD model (`ggml-silero-v6.2.0`)

> This step is not required if you're only using VibeVoice mode.

### 5. TTS engine setup

Set up the TTS engine you plan to use according to the `TTS_ENGINE` setting in `.env`.

#### MioTTS mode (`TTS_ENGINE=miotts`)

High-quality TTS with speaker cloning support. Requires Ollama + MioTTS-Inference server.

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

#### Kokoro TTS mode (`TTS_ENGINE=kokoro`)

Lightweight (82M parameters) and fast TTS. No server required, runs in-process. Doesn't support voice cloning but is fast and easy to use.

Dependencies are automatically installed with `uv sync`. **Additionally download the unidic dictionary required for Japanese normalization.**

```bash
uv run python -m unidic download
```

> **Important**: Skipping this step will result in incorrect Japanese pronunciation.

#### GPT-SoVITS mode (`TTS_ENGINE=gptsovits`)

Zero-shot voice cloning TTS using V2ProPlus model. Runs in an isolated conda environment and accessed via API server.

```bash
chmod +x scripts/setup_gptsovits.sh
./scripts/setup_gptsovits.sh
```

This script performs the following:

1. Clone the GPT-SoVITS repository
2. Create conda environment `gptsovits` (Python 3.11)
3. Install PyTorch + dependencies
4. Download NLTK data and pyopenjtalk dictionary
5. Download pre-trained models (V2ProPlus)
6. Generate `tts_infer.yaml` configuration file

> **Prerequisites**: conda (miniforge/miniconda) must be installed. GPT-SoVITS runs in CPU mode.

### 6. Create configuration file

```bash
cp .env.example .env
```

Open `.env` and edit the following items:

| Item | Description | Example |
|------|-------------|---------|
| `VIDEO_FOLDER` | Folder containing input videos | `./input_videos` |
| `HF_AUTH_TOKEN` | HuggingFace token (required for Whisper mode + MioTTS/GPT-SoVITS) | `hf_xxxxxxxxxxxx` |
| `ASR_ENGINE` | ASR engine selection | `whisper` or `vibevoice` |
| `TTS_ENGINE` | TTS engine selection | `miotts`, `kokoro`, or `gptsovits` |

Other configuration items work with default values. See "Configuration Settings" below for details.

Place the English video files (.mp4, .mkv, .mov, .webm, .m4v) you want to dub in the `VIDEO_FOLDER`.

## ASR Engine Selection

Switch between speech recognition engines using `ASR_ENGINE` in `.env`.

| Item | `whisper` | `vibevoice` |
|------|-----------|-------------|
| Engine | whisper.cpp CLI + Silero VAD | VibeVoice-ASR (Microsoft, mlx-audio) |
| Speaker diarization | pyannote.audio (separate step) | Built-in (single pass) |
| Speed | Fast | Slow (several times slower than Whisper) |
| VAD | Built-in Silero VAD (hallucination suppression) | None |
| Supported languages | English-focused | Strong with multilingual mixed content |
| Audio length limit | No limit | Memory-optimized for long audio |
| Additional setup | Requires running `scripts/setup_whisper.sh` | `mlx-audio[stt]>=0.3.0` |
| HuggingFace token | Required (for pyannote) | Not required |

### Whisper mode (default)

```env
ASR_ENGINE=whisper
```

Performs fast and high-quality transcription using whisper.cpp combined with Silero VAD, then performs speaker diarization with pyannote.audio. VAD suppresses hallucination (phantom text) in silent sections. Optimal for processing English audio.

### VibeVoice mode

```env
ASR_ENGINE=vibevoice
```

Microsoft's VibeVoice-ASR outputs transcription, speaker diarization, and timestamps in a single pass. Strong with multilingual mixed audio (English + local languages) but slower than Whisper.

Built-in encoder chunk processing for memory optimization allows processing of long audio even on 24GB unified memory Macs. Chunk size is automatically determined based on available memory, so no special user configuration is required.

VibeVoice mode doesn't require pyannote.audio and HuggingFace tokens. Steps 3 (speaker diarization) and 4 (speaker ID assignment) are automatically skipped.

#### VibeVoice-ASR specific settings

| Setting | Default | Description |
|---------|---------|-------------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | Model to use |
| `VIBEVOICE_MAX_TOKENS` | `32768` | Maximum generation tokens |
| `VIBEVOICE_CONTEXT` | (empty) | Hot words (proper noun recognition assistance, comma-separated) |

Hot word example:

```env
VIBEVOICE_CONTEXT=MLX, Apple Silicon, PyTorch, Transformer
```

## TTS Engine Selection

Switch between text-to-speech engines using `TTS_ENGINE` in `.env`.

| Item | `miotts` | `gptsovits` | `kokoro` |
|------|----------|-------------|----------|
| Engine | MioTTS-Inference | GPT-SoVITS V2ProPlus | Kokoro TTS (82M parameters) |
| Voice cloning | Supported (segment-based reference) | Supported (zero-shot, speaker representative reference) | Not supported (fixed voice) |
| Processing speed | Slow | Medium | Fast |
| Server | Required (Ollama + MioTTS API) | Required (conda environment + API server) | Not required (in-process inference) |
| Additional setup | MioTTS-Inference clone + Ollama | `scripts/setup_gptsovits.sh` + conda | `uv run python -m unidic download` |
| Audio quality | High quality with different voice characteristics per speaker | Zero-shot voice characteristic reproduction | Fixed voice but natural speech |
| Speaker diarization | Required | Required | Not required (all speakers same voice) |
| Quality validation | Yes (automatic detection and retry of abnormal audio) | No | No |

### MioTTS mode (default)

```env
TTS_ENGINE=miotts
```

Speaker cloning TTS using MioTTS-Inference. Generates Japanese audio that reproduces the original speaker's voice characteristics. Prioritizes segment-based reference audio and reflects emotion and tempo. Requires starting Ollama (LLM backend) and MioTTS API server.

MioTTS uses LLM for speech token generation and allows specifying LLM sampling parameters (temperature, repetition_penalty, etc.) from ja-dubbing per request. Default values are tuned for stability. It also includes a mechanism to automatically detect abnormal audio (extremely slow, fast, or corrupted audio) through generated audio length validation and retry with adjusted parameters.

### GPT-SoVITS mode

```env
TTS_ENGINE=gptsovits
```

Zero-shot voice cloning TTS using GPT-SoVITS V2ProPlus. Extracts voice quality from 3-10 second short reference audio and reuses speaker representative references. Reference audio transcription text (prompt_text) is automatically generated by the ASR engine. Runs in an isolated conda environment so it doesn't affect the ja-dubbing main Python environment.

### Kokoro TTS mode

```env
TTS_ENGINE=kokoro
```

Kokoro is a lightweight open-weight TTS model with 82M parameters. Doesn't support voice cloning but can generate Japanese audio quickly. No server startup required and translation is also completed in-process. Speaker diarization is also skipped, making it the most convenient to use.

#### Kokoro TTS specific settings

| Setting | Default | Description |
|---------|---------|-------------|
| `KOKORO_MODEL` | `kokoro` | Model name |
| `KOKORO_VOICE` | `jf_alpha` | Japanese voice name |
| `KOKORO_SPEED` | `1.0` | Reading speed (0.8-1.2 recommended) |

#### Available Japanese voices

| Voice name | Gender | Grade | Description |
|------------|--------|-------|-------------|
| `jf_alpha` | Female | C+ | Standard Japanese female voice (recommended) |
| `jf_gongitsune` | Female | C | "Gon the Fox" voice database |
| `jf_nezumi` | Female | C- | "The Mouse's Wedding" voice database |
| `jf_tebukuro` | Female | C | "Buying Gloves" voice database |
| `jm_kumo` | Male | C- | "The Spider's Thread" voice database |

#### GPT-SoVITS specific settings

| Setting | Default | Description |
|---------|---------|-------------|
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API URL |
| `GPTSOVITS_CONDA_ENV` | `gptsovits` | conda environment name |
| `GPTSOVITS_DIR` | `./GPT-SoVITS` | Installation directory |
| `GPTSOVITS_TEXT_LANG` | `ja` | Synthesis text language |
| `GPTSOVITS_PROMPT_LANG` | `en` | Reference audio language |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | Reading speed |
| `GPTSOVITS_REPETITION_PENALTY` | `1.35` | Repetition suppression penalty |
| `GPTSOVITS_REFERENCE_MIN_SEC` | `3.0` | Minimum reference audio duration |
| `GPTSOVITS_REFERENCE_MAX_SEC` | `10.0` | Maximum reference audio duration |
| `GPTSOVITS_REFERENCE_TARGET_SEC` | `5.0` | Target reference audio duration |

## Server Startup

Required servers differ by TTS engine. Translation uses CAT-Translate-7b with in-process inference so no server is required.

### For Kokoro TTS mode

**No external server startup required.** Both translation (CAT-Translate-7b) and TTS (Kokoro) run in-process.

```bash
uv run ja-dubbing
```

### For MioTTS mode

#### Method A: Use startup script (recommended)

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

MioTTS Ollama + API server will be started. Ollama model will be automatically downloaded on first run.

#### Method B: Manual startup (2 terminals)

**Terminal 1: MioTTS LLM backend (port 8000)**

```bash
OLLAMA_HOST=localhost:8000 ollama serve
# First time only: OLLAMA_HOST=localhost:8000 ollama pull hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf
```

**Terminal 2: MioTTS API server (port 8001)**

```bash
cd MioTTS-Inference
uv run python run_server.py \
    --llm-base-url http://localhost:8000/v1 \
    --device mps \
    --max-text-length 500 \
    --port 8001
```

> **Note**: MioTTS server-side sampling parameters (`MIOTTS_LLM_TEMPERATURE` etc.) are set as server defaults, but ja-dubbing sends parameters per request, so no need to modify server-side settings.

### For GPT-SoVITS mode

#### Method A: Use startup script (recommended)

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

GPT-SoVITS API server will be started in conda environment.

#### Method B: Manual startup (1 terminal)

```bash
conda activate gptsovits
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

## Execution

For TTS engines that require servers, run from a separate terminal with servers already started.

```bash
uv run ja-dubbing
```

Automatically detects videos in `VIDEO_FOLDER` and processes them sequentially. Output is saved as `*_jaDub.mp4` in the same folder as the input video. Specify any `VIDEO_FOLDER` path in `.env`.

### Generate startup script

```bash
uv run ja-dubbing --generate-script
```

Automatically generates appropriate server startup script `start_servers.sh` based on TTS engine settings.

## Processing Flow

### Whisper mode + MioTTS (`ASR_ENGINE=whisper`, `TTS_ENGINE=miotts`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. English transcription using whisper.cpp + Silero VAD (suppress hallucinations in silent sections)
3. Speaker diarization using pyannote.audio
4. Assign speaker IDs to Whisper segments
5. Segment merging → spaCy sentence splitting → translation unit merging (maintain speaker boundaries)
6. Extract speaker representative reference audio + segment-based reference audio from original video
7. English-to-Japanese translation using CAT-Translate-7b (GGUF, llama-cpp-python) with in-process inference
8. Generate speaker cloning Japanese audio using MioTTS-Inference (prioritize segment-based reference, with quality validation and auto-retry)
9. Speed stretch/compress original video sections to match TTS audio length
10. Composite video + Japanese audio (+ lightly mixed English audio) for output

### Whisper mode + GPT-SoVITS (`ASR_ENGINE=whisper`, `TTS_ENGINE=gptsovits`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. English transcription using whisper.cpp + Silero VAD
3. Speaker diarization using pyannote.audio
4. Assign speaker IDs to Whisper segments
5. Segment merging → spaCy sentence splitting → translation unit merging
6. Extract speaker representative reference audio (3-10 seconds) + transcribe with ASR
7. English-to-Japanese translation using CAT-Translate-7b (GGUF, llama-cpp-python) with in-process inference
8. Generate zero-shot voice cloning Japanese audio using GPT-SoVITS V2ProPlus
9. Speed stretch/compress original video sections to match TTS audio length
10. Composite video + Japanese audio (+ lightly mixed English audio) for output

### Whisper mode + Kokoro (`ASR_ENGINE=whisper`, `TTS_ENGINE=kokoro`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. English transcription using whisper.cpp + Silero VAD (suppress hallucinations in silent sections)
3. Speaker diarization: Skip (Kokoro doesn't support cloning, so pyannote not used)
4. Assign unified speaker ID to all segments
5. Segment merging → spaCy sentence splitting → translation unit merging
6. Reference audio extraction skipped (Kokoro doesn't support cloning)
7. English-to-Japanese translation using CAT-Translate-7b (GGUF, llama-cpp-python) with in-process inference
8. Generate Japanese audio quickly using Kokoro TTS
9. Speed stretch/compress original video sections to match TTS audio length
10. Composite video + Japanese audio (+ lightly mixed English audio) for output

### VibeVoice mode (`ASR_ENGINE=vibevoice`)

1. Extract 16kHz mono WAV from video using ffmpeg
2. Transcription + speaker diarization + timestamp acquisition using VibeVoice-ASR (single pass, memory optimization through chunk encoding)
3. (Skip: Built into VibeVoice-ASR)
4. (Skip: Built into VibeVoice-ASR)
5. Segment merging → spaCy sentence splitting → translation unit merging (maintain speaker boundaries)
6. MioTTS: Extract speaker reference audio from original video / GPT-SoVITS: Extract representative reference (3-10 seconds) / Kokoro: Skip
7. English-to-Japanese translation using CAT-Translate-7b (GGUF, llama-cpp-python) with in-process inference
8. MioTTS: Generate speaker cloning Japanese audio (with quality validation and auto-retry) / GPT-SoVITS: Generate zero-shot cloning audio / Kokoro: Generate Japanese audio quickly
9. Speed stretch/compress original video sections to match TTS audio length
10. Composite video + Japanese audio (+ lightly mixed English audio) for output

## Resume Feature

Processing saves checkpoints at each step. If interrupted, re-running will resume from where you left off. Checkpoints are saved in `temp/<video_name>/progress.json`.

To start from scratch, delete the corresponding `temp/<video_name>/` folder before re-running.

> **Note when switching ASR or TTS engines**: To redo a partially processed video with different engines, delete the `temp/<video_name>/` folder before re-running.

> **When changing MioTTS quality parameters**: To redo just the TTS step after changing `MIOTTS_LLM_TEMPERATURE` etc., delete `temp/<video_name>/tts_meta.json` and `temp/<video_name>/seg_audio/` before re-running.

## Configuration Settings

All settings are managed in the `.env` file. All configuration items and default values are documented in `.env.example`. Main configuration items are as follows:

### Basic Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `VIDEO_FOLDER` | `./input_videos` | Input video folder |
| `TEMP_ROOT` | `./temp` | Temporary files folder |
| `ASR_ENGINE` | `whisper` | ASR engine (`whisper` or `vibevoice`) |
| `TTS_ENGINE` | `miotts` | TTS engine (`miotts`, `kokoro`, or `gptsovits`) |
| `HF_AUTH_TOKEN` | (Required for Whisper mode + cloning-capable TTS) | HuggingFace token |

### ASR Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `WHISPER_LANG` | `en` | Whisper recognition language |
| `VAD_MODEL` | `silero-v6.2.0` | VAD model name |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp installation directory |
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | VibeVoice-ASR model name |
| `VIBEVOICE_MAX_TOKENS` | `32768` | VibeVoice-ASR maximum tokens |
| `VIBEVOICE_CONTEXT` | (empty) | VibeVoice-ASR hot words |

### Translation Settings (CAT-Translate-7b)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAT_TRANSLATE_REPO` | `mradermacher/CAT-Translate-7b-GGUF` | GGUF model HuggingFace repository |
| `CAT_TRANSLATE_FILE` | `CAT-Translate-7b.Q8_0.gguf` | GGUF filename |
| `CAT_TRANSLATE_N_GPU_LAYERS` | `-1` | GPU offload layers (-1 for all layers) |
| `CAT_TRANSLATE_N_CTX` | `4096` | Context window size |
| `CAT_TRANSLATE_RETRIES` | `3` | Translation retry count |
| `CAT_TRANSLATE_REPEAT_PENALTY` | `1.2` | Repetition suppression penalty |

### TTS Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_DEVICE` | `mps` | MioTTS codec device |
| `MIOTTS_REFERENCE_MAX_SEC` | `20.0` | MioTTS reference audio limit (seconds) |
| `MIOTTS_TTS_RETRIES` | `2` | MioTTS HTTP error retry count |
| `MIOTTS_QUALITY_RETRIES` | `2` | MioTTS quality validation failure retry count |
| `KOKORO_MODEL` | `kokoro` | Kokoro model name |
| `KOKORO_VOICE` | `jf_alpha` | Kokoro Japanese voice |
| `KOKORO_SPEED` | `1.0` | Kokoro reading speed |
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API URL |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | GPT-SoVITS reading speed |

### MioTTS LLM Sampling Parameters

MioTTS uses LLM for speech token generation and allows controlling generation stability and diversity with the following parameters. Default values are tuned for stability.

| Setting | Default | Description |
|---------|---------|-------------|
| `MIOTTS_LLM_TEMPERATURE` | `0.5` | Temperature parameter. Lower values produce more stable audio (0.1-0.8) |
| `MIOTTS_LLM_TOP_P` | `1.0` | Token selection diversity (1.0 for full sampling) |
| `MIOTTS_LLM_MAX_TOKENS` | `700` | Maximum generation tokens |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.1` | Token repetition suppression (1.0 to disable, 1.1-1.3 recommended) |
| `MIOTTS_LLM_PRESENCE_PENALTY` | `0.0` | Suppress reappearance of seen tokens (0.0-1.0) |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.3` | Suppress high-frequency tokens (0.0-1.0, 0.2-0.4 recommended) |

### MioTTS Quality Validation

Inspects the "seconds per character" ratio by dividing generated audio length by text character count to automatically detect abnormal audio. Automatically retries with adjusted parameters when detected.

| Setting | Default | Description |
|---------|---------|-------------|
| `MIOTTS_DURATION_PER_CHAR_MIN` | `0.05` | Lower limit of seconds per character (below this is "abnormally short") |
| `MIOTTS_DURATION_PER_CHAR_MAX` | `0.5` | Upper limit of seconds per character (above this is "abnormally long") |
| `MIOTTS_VALIDATION_MIN_CHARS` | `4` | Minimum characters for validation (skip texts that are too short) |

### Translation Anomaly Detection Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `OUTPUT_REPEAT_THRESHOLD` | `3` | Translation output repetition detection threshold |
| `INPUT_REPEAT_THRESHOLD` | `4` | Translation input repetition detection threshold |
| `INPUT_UNIQUE_RATIO_THRESHOLD` | `0.3` | Translation input unique ratio threshold |

### Output Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `ENGLISH_VOLUME` | `0.10` | English audio volume (0.0-1.0) |
| `JAPANESE_VOLUME` | `1.00` | Japanese audio volume (0.0-1.0) |
| `OUTPUT_SIZE` | `720` | Output video height (pixels) |
| `KEEP_TEMP` | `true` | Whether to keep temporary files |

## Known Limitations

- **Mixed English and Japanese**: When English words are mixed in like "utilizing simulation tools such as Omniverse and ISACsim," TTS may not read them correctly.
- **Processing time**: Even ~3 minute videos can take tens of minutes. Long videos may error, so we recommend **splitting to ~8 minutes beforehand**.
- **Translation quality**: Since it uses local LLM (CAT-Translate-7b), accuracy varies compared to cloud APIs.
- **MioTTS audio quality**: While MioTTS has high voice cloning quality, LLM-based token generation can probabilistically produce unstable audio (extremely slow, mysterious audio, etc.). Quality validation and auto-retry improve most cases but can't prevent all issues. See troubleshooting below if quality is consistently unstable.
- **Kokoro TTS**: Doesn't support voice cloning, so all speakers will have the same voice. Suitable when prioritizing speed or when voice characteristic reproduction is not needed.
- **GPT-SoVITS**: Runs in CPU mode, so inference is slower compared to MPS/CUDA environments. Reference audio only extracts voice quality (timbre), so intonation and tempo are not reflected.
- **VibeVoice-ASR processing speed**: Takes several times longer compared to Whisper.
- **VibeVoice-ASR memory usage**: Uses about 5GB memory even with 8bit quantization due to 9B parameter model. Built-in memory optimization through chunk encoding, tested on 24GB unified memory Mac.

## Troubleshooting

### whisper-cli not found

Run `scripts/setup_whisper.sh` to build whisper.cpp.

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

If CMake is not installed, run `brew install cmake` first.

### Out of memory

Tested on Mac mini M4 with 24GB unified memory. To save memory, ASR models (Whisper/VibeVoice-ASR) and pyannote pipeline are released after use. VibeVoice mode suppresses memory spikes for long audio through encoder chunk processing. MLX and PyTorch MPS caches are also cleared after each step. For long videos, set `KEEP_TEMP=true` and utilize interrupt/resume functionality.

### MioTTS text too long error

MioTTS default max text length is 300 characters. This tool specifies `--max-text-length 500` at server startup and also performs truncation processing at punctuation positions on the pipeline side.

### MioTTS unstable audio (slow/corrupted)

MioTTS performs LLM-based token generation, so probabilistically unstable audio may be generated. Quality validation and auto-retry are built-in, but if issues persist, adjust the following parameters in `.env`:

```env
# Lower temperature (improved stability, reduced expressiveness)
MIOTTS_LLM_TEMPERATURE=0.3

# Increase repetition_penalty (further suppress token loops)
MIOTTS_LLM_REPETITION_PENALTY=1.2

# Increase quality retry count
MIOTTS_QUALITY_RETRIES=3

# Stricter quality judgment upper limit (more aggressively eliminate abnormally long audio)
MIOTTS_DURATION_PER_CHAR_MAX=0.4
```

To redo just the TTS step after changes:

```bash
rm -f temp/<video_name>/tts_meta.json
rm -rf temp/<video_name>/seg_audio/
uv run ja-dubbing
```

If quality retry messages like the following appear in logs, validation is working correctly:

```
    Quality retry 1/3: Audio abnormally long: 15.2sec / 8chars = 1.900sec/char (limit: 0.500sec/char) → Regenerating after 1sec
```

### Kokoro TTS incorrect Japanese pronunciation

The unidic dictionary may not be downloaded. Run the following:

```bash
uv run python -m unidic download
```

### VibeVoice-ASR "mlx-audio not installed" error

Run the following:

```bash
uv pip install 'mlx-audio[stt]>=0.3.0'
```

### VibeVoice-ASR empty transcription results

The audio file may not contain speech or the audio may be too short. Try increasing `VIBEVOICE_MAX_TOKENS` or switch to Whisper mode.

### CAT-Translate-7b model download failure

Automatically downloaded via huggingface_hub. Check network connection and re-run. Model is downloaded from `mradermacher/CAT-Translate-7b-GGUF`.

### Cannot connect to GPT-SoVITS API server

Verify that the conda environment is set up correctly.

```bash
conda activate gptsovits
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

If not set up, run `scripts/setup_gptsovits.sh` first.

## License

MIT License

Note: External models and libraries used by this tool have their own licenses.

- MioTTS default preset
- CAT-Translate-7b
- pyannote.audio
- whisper.cpp
- Silero VAD
- VibeVoice-ASR
- mlx-audio
- Kokoro TTS (Kokoro-82M)
- misaki (G2P)
- GPT-SoVITS
- llama-cpp-python