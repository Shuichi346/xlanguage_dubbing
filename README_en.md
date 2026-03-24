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
  <p align="center">A tool that converts English videos into Japanese dubbed videos.<br>It can also create dubbing that reproduces the original speaker's voice.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## What This Tool Can Do

- Input an English video and output a Japanese dubbed video
- Create dubbing that mimics the original speaker's voice (voice cloning)
- Automatically adjust video speed for natural dubbing
- Resume from where it left off even if processing stops midway

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Engine Combinations](#engine-combinations)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## System Requirements

- **Mac (Apple Silicon)** — Tested on Mac mini M4 (24GB)
- **Python 3.13 or higher**
- Linux is not tested

---

## Setup

### 1. Install Required Tools

First, install several tools on your Mac. Open Terminal and execute the following commands **in order**.

**Homebrew** (Mac package manager. Skip if already installed)

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**ffmpeg, CMake, uv**

```bash
brew install ffmpeg cmake uv
```

### 2. Download Repository

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 3. Install Dependencies

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

### 4. Create Configuration File

```bash
cp .env.example .env
```

Open `.env` in a text editor and **make sure to configure the following 4 items**.

| Item | Description | Example |
|------|-------------|---------|
| `VIDEO_FOLDER` | Folder containing videos to dub | `./input_videos` |
| `ASR_ENGINE` | Speech recognition engine | `whisper` or `vibevoice` |
| `TTS_ENGINE` | Text-to-speech engine | `kokoro`, `miotts`, `gptsovits`, or `t5gemma` |
| `HF_AUTH_TOKEN` | HuggingFace token (※conditional) | `hf_xxxxxxxxxxxx` |

> **When `HF_AUTH_TOKEN` is needed**: Required when `ASR_ENGINE=whisper` and `TTS_ENGINE=miotts` or `gptsovits`. Not needed for `kokoro` and `t5gemma`. Also not needed when `ASR_ENGINE=vibevoice`.

### 5. ASR Engine Setup

#### For Whisper Mode (`ASR_ENGINE=whisper`)

Build whisper.cpp and download the model.

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

**HuggingFace Setup** (Only needed for Whisper + MioTTS/GPT-SoVITS combination)

1. Create a token at https://huggingface.co/settings/tokens (`Read` permission)
2. **Accept the terms of use** on the following 2 pages (instantly approved)
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

#### For VibeVoice Mode (`ASR_ENGINE=vibevoice`)

No additional setup required. The model (~5GB) will be automatically downloaded on first run.

### 6. TTS Engine Setup

#### For Kokoro TTS (`TTS_ENGINE=kokoro`) — Easiest Option

No server startup required. Just run the following command:

```bash
uv run python -m unidic download
```

> **Important**: Skipping this step will result in incorrect Japanese pronunciation.

#### For MioTTS (`TTS_ENGINE=miotts`)

Requires Ollama installation and MioTTS-Inference cloning.

**Install Ollama**: Download the macOS version from https://ollama.com/download

**Clone MioTTS-Inference**:

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

#### For T5Gemma-TTS (`TTS_ENGINE=t5gemma`) — Voice Cloning + Duration Control

No server startup required. The model and audio codec will be automatically downloaded on first run.

> **Important**: While it can run on a 24GB memory Mac, the initial loading is heavy. Close heavy applications like browsers or video editing software.

> **Note**: By default, `T5GEMMA_CPU_CODEC=true` offloads XCodec2 to CPU to reduce memory usage.

#### For GPT-SoVITS (`TTS_ENGINE=gptsovits`)

Requires conda and GPT-SoVITS setup.

```bash
brew install --cask miniforge
chmod +x scripts/setup_gptsovits.sh
./scripts/setup_gptsovits.sh
```

---

## Usage

### Step 1: Place Videos

Put English videos (.mp4, .mkv, .mov, .webm, .m4v) to be dubbed in `VIDEO_FOLDER`.

> **Recommendation**: Long videos may cause errors.

### Step 2: Start Servers (MioTTS / GPT-SoVITS only)

**For Kokoro TTS / T5Gemma-TTS**: No server startup needed. Proceed to Step 3.

**For MioTTS / GPT-SoVITS**: Start servers in a separate terminal.

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

### Step 3: Execute

```bash
uv run ja-dubbing
```

Videos in `VIDEO_FOLDER` will be processed sequentially and output as `*_jaDub.mp4` in the same folder.

---

## Engine Combinations

### ASR Engines (Speech Recognition)

| | Whisper | VibeVoice |
|---|---------|-----------|
| Speed | Fast | Slow |
| Best for | English only | Mixed multilingual audio |
| Additional Setup | Requires running `setup_whisper.sh` | Not needed (auto-download on first run) |
| HuggingFace Token | Needed when combined with MioTTS/GPT-SoVITS | Not needed |

### TTS Engines (Text-to-Speech)

| | Kokoro | MioTTS | GPT-SoVITS | T5Gemma-TTS |
|---|--------|--------|------------|--------------|
| Ease of Use | ★★★ Easiest | ★★ Server startup needed | ★ conda environment needed | ★★★ No server needed |
| Voice Cloning | Not supported (fixed voice) | Supported (high quality) | Supported (zero-shot) | Supported (zero-shot) |
| Duration Control | Not supported | Not supported | Not supported | Supported |
| Speed | Fast | Slow | Medium | Slow |
| Server | Not needed | Ollama + MioTTS API | conda + API server | Not needed |

**If unsure**: Try the `ASR_ENGINE=whisper` + `TTS_ENGINE=kokoro` combination first. It has the simplest setup and doesn't require a HuggingFace token.

---

## Configuration Options

All settings are managed in the `.env` file. All items and default values are listed in `.env.example`.

### Basic Settings (Must Configure)

| Setting | Default | Description |
|---------|---------|-------------|
| `VIDEO_FOLDER` | `./input_videos` | Input video folder |
| `ASR_ENGINE` | `whisper` | Speech recognition engine (`whisper` / `vibevoice`) |
| `TTS_ENGINE` | `miotts` | Text-to-speech engine (`kokoro` / `miotts` / `gptsovits` / `t5gemma`) |
| `HF_AUTH_TOKEN` | — | HuggingFace token (conditionally required) |

### Output Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `ENGLISH_VOLUME` | `0.10` | Original English audio volume (0.0-1.0) |
| `JAPANESE_VOLUME` | `1.00` | Japanese dub audio volume (0.0-1.0) |
| `OUTPUT_SIZE` | `720` | Output video height (pixels) |
| `KEEP_TEMP` | `true` | Whether to keep temporary files (needed for resuming) |

### Whisper Settings (`ASR_ENGINE=whisper`)

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model name |
| `WHISPER_LANG` | `en` | Recognition language |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp installation directory |

### VibeVoice Settings (`ASR_ENGINE=vibevoice`)

| Setting | Default | Description |
|---------|---------|-------------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | Model name |
| `VIBEVOICE_MAX_TOKENS` | `32768` | Maximum generation tokens |
| `VIBEVOICE_CONTEXT` | (empty) | Hotwords (proper noun recognition assistance, comma-separated) |

### Translation Settings (CAT-Translate-7b)

| Setting | Default | Description |
|---------|---------|-------------|
| `CAT_TRANSLATE_REPO` | `mradermacher/CAT-Translate-7b-GGUF` | Model repository |
| `CAT_TRANSLATE_FILE` | `CAT-Translate-7b.Q8_0.gguf` | Model filename |
| `CAT_TRANSLATE_N_GPU_LAYERS` | `-1` | GPU offload (-1 for all layers) |
| `CAT_TRANSLATE_RETRIES` | `3` | Retry count |

### Kokoro TTS Settings (`TTS_ENGINE=kokoro`)

| Setting | Default | Description |
|---------|---------|-------------|
| `KOKORO_VOICE` | `jf_alpha` | Japanese voice name |
| `KOKORO_SPEED` | `1.0` | Speech speed (0.8-1.2 recommended) |

Available voices: `jf_alpha` (female, recommended), `jf_gongitsune` (female), `jf_nezumi` (female), `jf_tebukuro` (female), `jm_kumo` (male)

### MioTTS Settings (`TTS_ENGINE=miotts`)

| Setting | Default | Description |
|---------|---------|-------------|
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_LLM_TEMPERATURE` | `0.5` | Temperature (lower is more stable, 0.1-0.8) |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.1` | Repetition penalty (1.0-1.3 recommended) |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.3` | High-frequency token penalty (0.0-1.0) |
| `MIOTTS_QUALITY_RETRIES` | `2` | Retry count on quality validation failure |
| `MIOTTS_DURATION_PER_CHAR_MAX` | `0.5` | Maximum seconds per character |

### GPT-SoVITS Settings (`TTS_ENGINE=gptsovits`)

| Setting | Default | Description |
|---------|---------|-------------|
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API URL |
| `GPTSOVITS_CONDA_ENV` | `gptsovits` | conda environment name |
| `GPTSOVITS_DIR` | `./GPT-SoVITS` | Installation directory |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | Speech speed |
| `GPTSOVITS_REFERENCE_TARGET_SEC` | `5.0` | Target seconds for reference audio |

### T5Gemma-TTS Settings (`TTS_ENGINE=t5gemma`)

| Setting | Default | Description |
|---------|---------|-------------|
| `T5GEMMA_MODEL_DIR` | `Aratako/T5Gemma-TTS-2b-2b` | Model HuggingFace repository |
| `T5GEMMA_XCODEC2_MODEL` | `NandemoGHS/Anime-XCodec2-44.1kHz-v2` | Audio codec repository |
| `T5GEMMA_DURATION_SCALE` | `1.15` | Multiplier for original segment length |
| `T5GEMMA_CPU_CODEC` | `true` | Offload XCodec2 to CPU to save memory |
| `T5GEMMA_DURATION_TOLERANCE` | `0.5` | Tolerance rate for generated audio length |
| `T5GEMMA_QUALITY_RETRIES` | `2` | Quality regeneration attempts |

`T5Gemma-TTS` uses the English transcription (`text_en`) obtained before translation directly as Reference Text. It does not re-transcribe reference audio.

---

## About Resume Functionality

Processing saves checkpoints at each step, so if it stops midway, you can resume by re-running `uv run ja-dubbing`.

**To start over**: Delete the `temp/<video_name>/` folder and re-run.

**To retry with different engines**: Similarly, delete the `temp/<video_name>/` folder.

## License

MIT License

External models and libraries used by this tool (MioTTS, CAT-Translate-7b, pyannote.audio, whisper.cpp, Kokoro TTS, GPT-SoVITS, T5Gemma-TTS, etc.) each have their own licenses. Please check them when using.