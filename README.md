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
  <p align="center">英語の動画を、日本語の吹き替え動画に変換するツールです。<br>元の話者の声色を再現した吹き替えも可能です。</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## このツールでできること

- 英語の動画を入力すると、日本語に吹き替えた動画が出力されます
- 元の話者の声に似せた吹き替え（ボイスクローン）ができます
- 動画の速度を自動調整して、自然な吹き替えを実現します
- 処理が途中で止まっても、続きから再開できます

---

## 目次

- [動作環境](#動作環境)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [エンジンの組み合わせ](#エンジンの組み合わせ)
- [設定項目一覧](#設定項目一覧)
- [トラブルシューティング](#トラブルシューティング)
- [ライセンス](#ライセンス)

---

## 動作環境

- **Mac（Apple Silicon）** — Mac mini M4 (24GB) で動作確認済み
- **Python 3.13 以上**
- Linux は未検証です

---

## セットアップ

### 1. 必要なツールをインストールする

まず、Mac にいくつかのツールをインストールします。ターミナルを開いて、以下のコマンドを**順番に**実行してください。

**Homebrew**（Mac のパッケージ管理ツール。既に入っている場合はスキップ）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**ffmpeg・CMake・uv**

```bash
brew install ffmpeg cmake uv
```

### 2. リポジトリをダウンロードする

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 3. 依存パッケージをインストールする

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

### 4. 設定ファイルを作成する

```bash
cp .env.example .env
```

テキストエディタで `.env` を開き、**以下の4項目を必ず設定**してください。

| 項目 | 説明 | 設定例 |
|------|------|--------|
| `VIDEO_FOLDER` | 吹き替えたい動画を置くフォルダ | `./input_videos` |
| `ASR_ENGINE` | 音声認識エンジン | `whisper` または `vibevoice` |
| `TTS_ENGINE` | 音声合成エンジン | `kokoro`、`miotts`、`gptsovits`、または `t5gemma` |
| `HF_AUTH_TOKEN` | HuggingFace のトークン（※条件付き） | `hf_xxxxxxxxxxxx` |

> **`HF_AUTH_TOKEN` が必要な条件**: `ASR_ENGINE=whisper` かつ `TTS_ENGINE=miotts` または `gptsovits` の場合に必要です。`kokoro` と `t5gemma` は不要です。`ASR_ENGINE=vibevoice` の場合も不要です。

### 5. ASR エンジンのセットアップ

#### Whisper モードの場合（`ASR_ENGINE=whisper`）

whisper.cpp をビルドし、モデルをダウンロードします。

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

**HuggingFace の準備**（Whisper + MioTTS/GPT-SoVITS の組み合わせのみ必要）

1. https://huggingface.co/settings/tokens でトークンを作成（`Read` 権限）
2. 以下の2つのページで**利用規約に同意**する（即時承認されます）
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

#### VibeVoice モードの場合（`ASR_ENGINE=vibevoice`）

追加のセットアップは不要です。初回実行時にモデル（約5GB）が自動ダウンロードされます。

### 6. TTS エンジンのセットアップ

#### Kokoro TTS の場合（`TTS_ENGINE=kokoro`）— 一番かんたん

サーバー起動不要。以下のコマンドだけ実行してください。

```bash
uv run python -m unidic download
```

> **重要**: この手順を省略すると日本語の発音がおかしくなります。

#### MioTTS の場合（`TTS_ENGINE=miotts`）

Ollama のインストールと MioTTS-Inference のクローンが必要です。

**Ollama をインストール**: https://ollama.com/download から macOS 版をダウンロード

**MioTTS-Inference をクローン**:

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

#### T5Gemma-TTS の場合（`TTS_ENGINE=t5gemma`）— ボイスクローン + 再生時間制御

サーバー起動は不要です。初回実行時にモデル本体と音声コーデックが自動ダウンロードされます。

> **重要**: 24GB メモリの Mac で動かせますが、初回ロードは重いです。ブラウザや動画編集ソフトなど、重いアプリは閉じてください。

> **補足**: デフォルトでは `T5GEMMA_CPU_CODEC=true` なので、XCodec2 は CPU 側へ逃がしてメモリ使用量を抑えます。

#### GPT-SoVITS の場合（`TTS_ENGINE=gptsovits`）

conda と GPT-SoVITS のセットアップが必要です。

```bash
brew install --cask miniforge
chmod +x scripts/setup_gptsovits.sh
./scripts/setup_gptsovits.sh
```

---

## 使い方

### 手順1: 動画を配置する

吹き替えたい英語動画（.mp4, .mkv, .mov, .webm, .m4v）を `VIDEO_FOLDER` に入れます。

> **推奨**: 長い動画はエラーになることがある。

### 手順2: サーバーを起動する（MioTTS / GPT-SoVITS のみ）

**Kokoro TTS / T5Gemma-TTS の場合**: サーバー起動は不要です。手順3に進んでください。

**MioTTS / GPT-SoVITS の場合**: 別のターミナルでサーバーを起動します。

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

### 手順3: 実行する

```bash
uv run ja-dubbing
```

`VIDEO_FOLDER` 内の動画が順に処理され、同じフォルダに `*_jaDub.mp4` として出力されます。

---

## エンジンの組み合わせ

### ASR エンジン（音声認識）

| | Whisper | VibeVoice |
|---|---------|-----------|
| 速度 | 高速 | 低速 |
| 得意な音声 | 英語のみ | 多言語が混ざった音声 |
| 追加セットアップ | `setup_whisper.sh` の実行が必要 | 不要（初回自動ダウンロード） |
| HuggingFace トークン | MioTTS/GPT-SoVITS と組み合わせる場合は必要 | 不要 |

### TTS エンジン（音声合成）

| | Kokoro | MioTTS | GPT-SoVITS | T5Gemma-TTS |
|---|--------|--------|------------|--------------|
| 手軽さ | ★★★ 一番かんたん | ★★ サーバー起動が必要 | ★ conda 環境が必要 | ★★★ サーバー不要 |
| ボイスクローン | 非対応（固定の声） | 対応（高品質） | 対応（ゼロショット） | 対応（ゼロショット） |
| 再生時間制御 | 非対応 | 非対応 | 非対応 | 対応 |
| 速度 | 高速 | 低速 | 中速 | 低速 |
| サーバー | 不要 | Ollama + MioTTS API | conda + API サーバー | 不要 |

**迷ったら**: まずは `ASR_ENGINE=whisper` + `TTS_ENGINE=kokoro` の組み合わせで試してみてください。セットアップが最も簡単で、HuggingFace トークンも不要です。

---

## 設定項目一覧

すべての設定は `.env` ファイルで管理します。`.env.example` に全項目とデフォルト値が記載されています。

### 基本設定（必ず確認）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `VIDEO_FOLDER` | `./input_videos` | 入力動画フォルダ |
| `ASR_ENGINE` | `whisper` | 音声認識エンジン（`whisper` / `vibevoice`） |
| `TTS_ENGINE` | `miotts` | 音声合成エンジン（`kokoro` / `miotts` / `gptsovits` / `t5gemma`） |
| `HF_AUTH_TOKEN` | — | HuggingFace トークン（条件付きで必要） |

### 出力設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `ENGLISH_VOLUME` | `0.10` | 元の英語音声の音量（0.0〜1.0） |
| `JAPANESE_VOLUME` | `1.00` | 日本語吹き替え音声の音量（0.0〜1.0） |
| `OUTPUT_SIZE` | `720` | 出力動画の高さ（ピクセル） |
| `KEEP_TEMP` | `true` | 一時ファイルを残すか（再開に必要） |

### Whisper 設定（`ASR_ENGINE=whisper`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper モデル名 |
| `WHISPER_LANG` | `en` | 認識言語 |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp のインストール先 |

### VibeVoice 設定（`ASR_ENGINE=vibevoice`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | モデル名 |
| `VIBEVOICE_MAX_TOKENS` | `32768` | 最大生成トークン数 |
| `VIBEVOICE_CONTEXT` | （空） | ホットワード（固有名詞の認識補助、カンマ区切り） |

### 翻訳設定（CAT-Translate-7b）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `CAT_TRANSLATE_REPO` | `mradermacher/CAT-Translate-7b-GGUF` | モデルのリポジトリ |
| `CAT_TRANSLATE_FILE` | `CAT-Translate-7b.Q8_0.gguf` | モデルファイル名 |
| `CAT_TRANSLATE_N_GPU_LAYERS` | `-1` | GPU オフロード（-1 で全レイヤー） |
| `CAT_TRANSLATE_RETRIES` | `3` | リトライ回数 |

### Kokoro TTS 設定（`TTS_ENGINE=kokoro`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `KOKORO_VOICE` | `jf_alpha` | 日本語ボイス名 |
| `KOKORO_SPEED` | `1.0` | 読み上げ速度（0.8〜1.2 推奨） |

利用可能なボイス: `jf_alpha`（女性・推奨）、`jf_gongitsune`（女性）、`jf_nezumi`（女性）、`jf_tebukuro`（女性）、`jm_kumo`（男性）

### MioTTS 設定（`TTS_ENGINE=miotts`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API の URL |
| `MIOTTS_LLM_TEMPERATURE` | `0.5` | 温度（低いほど安定、0.1〜0.8） |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.1` | 繰り返し抑制（1.0〜1.3 推奨） |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.3` | 高頻度トークン抑制（0.0〜1.0） |
| `MIOTTS_QUALITY_RETRIES` | `2` | 品質バリデーション失敗時のリトライ回数 |
| `MIOTTS_DURATION_PER_CHAR_MAX` | `0.5` | 1文字あたり秒数の上限 |

### GPT-SoVITS 設定（`TTS_ENGINE=gptsovits`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API の URL |
| `GPTSOVITS_CONDA_ENV` | `gptsovits` | conda 環境名 |
| `GPTSOVITS_DIR` | `./GPT-SoVITS` | インストール先ディレクトリ |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | 読み上げ速度 |
| `GPTSOVITS_REFERENCE_TARGET_SEC` | `5.0` | 参照音声の目標秒数 |

### T5Gemma-TTS 設定（`TTS_ENGINE=t5gemma`）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `T5GEMMA_MODEL_DIR` | `Aratako/T5Gemma-TTS-2b-2b` | モデルの HuggingFace リポジトリ |
| `T5GEMMA_XCODEC2_MODEL` | `NandemoGHS/Anime-XCodec2-44.1kHz-v2` | 音声コーデックのリポジトリ |
| `T5GEMMA_DURATION_SCALE` | `1.15` | 元セグメント長に掛ける倍率 |
| `T5GEMMA_CPU_CODEC` | `true` | XCodec2 を CPU に逃がしてメモリを節約する |
| `T5GEMMA_DURATION_TOLERANCE` | `0.5` | 生成音声長の許容ずれ率 |
| `T5GEMMA_QUALITY_RETRIES` | `2` | 品質再生成の回数 |

`T5Gemma-TTS` は、翻訳前に得た英語文字起こし (`text_en`) をそのまま Reference Text として使います。参照音声の再文字起こしはしません。

---

## 再開機能について

処理は各ステップでチェックポイントを保存するので、途中で止まっても `uv run ja-dubbing` を再実行すれば続きから処理されます。

**最初からやり直したい場合**: `temp/<動画名>/` フォルダを削除してから再実行してください。

**エンジンを切り替えてやり直す場合**: 同じく `temp/<動画名>/` フォルダを削除してください。

## ライセンス

MIT License

本ツールが使用する外部モデル・ライブラリ（MioTTS、CAT-Translate-7b、pyannote.audio、whisper.cpp、Kokoro TTS、GPT-SoVITS、T5Gemma-TTS など）にはそれぞれ固有のライセンスがあります。利用の際はご確認ください。
