<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README.md">English</a></th>
      <th style="text-align:center"><a href="README_ja.md">日本語</a></th>
    </tr>
  </thead>
</table>

<p align="center">
  <h1 align="center">xlanguage-dubbing</h1>
  <p align="center">A tool for converting multilingual videos into dubbed videos in other languages.<br>Creates dubbing that reproduces the original speaker's voice using high-precision voice cloning.</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-8.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## What this tool can do

- Input multilingual videos and output dubbed videos in other languages
- Create dubbing that mimics the original speaker's voice using high-precision voice cloning (OmniVoice)
- Automatically selects high-precision CAT-Translate-7b for Japanese-English and English-Japanese translation, and TranslateGemma-12b-it supporting 55 languages for other language pairs
- Automatic video speed adjustment for natural dubbing
- Resume from where it left off even if processing is interrupted

### Demo Video

<a href="https://www.youtube.com/watch?v=amYVIorgOQQ">
  <img src="https://img.youtube.com/vi/amYVIorgOQQ/0.jpg" width="250" alt="Video Title">
</a>

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Setup](#setup)
- [Usage](#usage)
- [ASR Engine Selection](#asr-engines)
- [Language Settings](#language-settings)
- [Configuration Options](#configuration-options)
- [License](#license)

---

## System Requirements

- **Mac (Apple Silicon)** — Tested on Mac mini M4 (24GB)
- **Python 3.13 or higher**
- Linux is untested

---

## Setup

### 1. Install Required Tools

```bash
brew install ffmpeg cmake uv
```

### 2. Download Repository

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
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

Edit `.env` to configure the following:

| Item | Description | Example |
|------|-------------|---------|
| `VIDEO_FOLDER` | Folder containing videos to dub | `./input_videos` |
| `INPUT_LANG` | Audio language of original video (`auto` for auto-detection) | `auto`, `en`, `ja` |
| `OUTPUT_LANG` | Output dubbing language | `ja`, `en`, `fr` |
| `ASR_ENGINE` | Speech recognition engine | `vibevoice` (recommended), `whisper` |
| `HF_AUTH_TOKEN` | HuggingFace token (only when using whisper) | `hf_xxxxxxxxxxxx` |

### 5. ASR Engine Setup

#### VibeVoice Mode (Default/Recommended)

No additional setup required. Models will be automatically downloaded on first run.

#### Whisper Mode

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

### 6. Run

```bash
uv run xlanguage-dubbing
```

---

## ASR Engine Selection

| | VibeVoice (Recommended) | Whisper |
|---|---|---|
| Speed | Slow | Fast |
| Multilingual mixing (Code-switching) | Supported | Not supported (single language only) |
| Additional setup | Not required | Requires running `setup_whisper.sh` |
| HuggingFace token | Not required | Required |

---

## Language Settings

### INPUT_LANG

Specifies the audio language of the original video. When set to `auto`, the ASR engine will automatically detect it. VibeVoice-ASR supports code-switching, so it can handle multiple languages mixed within a single video without any issues.

### OUTPUT_LANG

Explicitly specifies the audio language of the output dubbed video. Use ISO 639-1 codes (`en`, `ja`, `fr`, `de`, `zh`, `ko`, etc.).

### Automatic Translation Engine Selection

The optimal translation engine is automatically selected based on the input-output language combination.

| Language Pair | Engine Used | Notes |
|---|---|---|
| English → Japanese | CAT-Translate-7b | Japanese-English specialized, high precision |
| Japanese → English | CAT-Translate-7b | Japanese-English specialized, high precision |
| All others | TranslateGemma-12b-it | Supports 55 languages |

---

## Configuration Options

All settings are managed in the `.env` file. Refer to `.env.example` for details.

---

## Resume Functionality

Processing saves checkpoints at each step, so if it stops midway, you can resume from where it left off by re-running `uv run xlanguage-dubbing`.

**To start over from the beginning**: Delete the `temp/<video_name>/` folder and re-run.

## License

MIT License

External models and libraries used by this tool have their own respective licenses.