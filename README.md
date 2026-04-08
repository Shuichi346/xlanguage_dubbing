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
  <p align="center">英語動画を日本語吹き替え動画に変換するツールです。<br>高精度な音声クローニングを使って、元の話者の声を再現した吹き替えも作成できます。</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## このツールでできること

- 英語動画を入力して日本語吹き替え動画を出力
- 高精度音声クローニング（OmniVoice）を使用した元話者の声を模倣した吹き替えの作成
- 自然な吹き替えのための動画速度の自動調整
- 処理が途中で止まっても続きから再開

### デモ動画

<a href="https://www.youtube.com/watch?v=amYVIorgOQQ">
  <img src="https://img.youtube.com/vi/amYVIorgOQQ/0.jpg" width="250" alt="動画タイトル">
</a>

---

## 目次

- [システム要件](#system-requirements)
- [セットアップ](#setup)
- [使用方法](#usage)
- [エンジンの組み合わせ](#engine-combinations)
- [設定オプション](#configuration-options)
- [トラブルシューティング](#troubleshooting)
- [ライセンス](#license)

---

## システム要件

- **Mac（Apple Silicon）** — Mac mini M4（24GB）でテスト済み
- **Python 3.13以上**
- Linuxは未テスト

---

## セットアップ

### 1. 必要なツールのインストール

まず、Macにいくつかのツールをインストールします。ターミナルを開いて以下のコマンドを**順番に**実行してください。

**Homebrew**（Macのパッケージマネージャー。既にインストール済みの場合はスキップ）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**ffmpeg、CMake、uv**

```bash
brew install ffmpeg cmake uv
```

### 2. リポジトリのダウンロード

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 3. 依存関係のインストール

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

### 4. 設定ファイルの作成

```bash
cp .env.example .env
```

`.env`をテキストエディタで開いて、**以下の4項目を必ず設定**してください。

| 項目 | 説明 | 例 |
|------|-------------|---------|
| `VIDEO_FOLDER` | 吹き替えする動画が入っているフォルダ | `./input_videos` |
| `ASR_ENGINE` | 音声認識エンジン | `whisper` または `vibevoice` |
| `TTS_ENGINE` | 音声合成エンジン | `kokoro` または `omnivoice` |
| `HF_AUTH_TOKEN` | HuggingFaceトークン（※条件付き） | `hf_xxxxxxxxxxxx` |

> **`HF_AUTH_TOKEN`が必要なとき**: `ASR_ENGINE=whisper`の場合に必要。`ASR_ENGINE=vibevoice`の場合は不要。

### 5. ASRエンジンのセットアップ

#### Whisperモード（`ASR_ENGINE=whisper`）の場合

whisper.cppをビルドしてモデルをダウンロードします。

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

**HuggingFaceの設定**（Whisperのみ必要）

1. https://huggingface.co/settings/tokens でトークンを作成（`Read`権限）
2. 以下の2ページで**利用規約に同意**（即座に承認されます）
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/segmentation-3.0

#### VibeVoiceモード（`ASR_ENGINE=vibevoice`）の場合

追加セットアップは不要です。初回実行時にモデル（約5GB）が自動でダウンロードされます。

### 6. TTSエンジンのセットアップ

#### Kokoro TTS（`TTS_ENGINE=kokoro`）— 最も簡単なオプション

サーバーの起動は不要です。以下のコマンドを実行するだけです：

```bash
uv run python -m unidic download
```

> **重要**: このステップをスキップすると日本語の発音が正しくなくなります。

#### OmniVoice（`TTS_ENGINE=omnivoice`）— 高精度音声クローニング

現在の音声クローニング音声合成機能は、Kokoro TTSを維持しながら[OmniVoice](https://github.com/k2-fsa/OmniVoice)（[HuggingFace](https://huggingface.co/k2-fsa/OmniVoice)）のみを使用するように合理化されました。これはOmniVoiceが高精度を提供するためです。

追加セットアップは不要です。初回実行時にモデルが自動でダウンロードされます。

---

## 使用方法

### ステップ1: 動画を配置

`VIDEO_FOLDER`に吹き替えしたい英語動画（.mp4、.mkv、.mov、.webm、.m4v）を置きます。

> **推奨**: 長時間の動画はエラーの原因となる場合があります。

### ステップ2: サーバーの起動（不要）

どのTTSエンジンでもサーバーの起動は不要です。ステップ3に進んでください。

### ステップ3: 実行

```bash
uv run ja-dubbing
```

`VIDEO_FOLDER`内の動画が順次処理され、同じフォルダに`*_jaDub.mp4`として出力されます。

---

## エンジンの組み合わせ

### ASRエンジン（音声認識）

| | Whisper | VibeVoice |
|---|---|---|
| 速度 | 高速 | 低速 |
| 適用場面 | 英語のみ | 多言語混在音声 |
| 追加セットアップ | `setup_whisper.sh`の実行が必要 | 不要（初回実行時に自動ダウンロード） |
| HuggingFaceトークン | 必要 | 不要 |

### TTSエンジン（音声合成）

| | Kokoro | OmniVoice |
|---|---|---|
| 使いやすさ | ★★★ 最も簡単 | ★★★ サーバー不要 |
| 音声クローニング | 非対応（固定音声） | 対応（高精度） |
| 速度 | 高速 | 中程度 |
| サーバー | 不要 | 不要 |

**迷ったら**: まず`ASR_ENGINE=whisper` + `TTS_ENGINE=kokoro`の組み合わせを試してください。最もシンプルなセットアップでHuggingFaceトークンも不要です。

---

## 設定オプション

すべての設定は`.env`ファイルで管理されます。全項目とデフォルト値は`.env.example`に記載されています。

### 基本設定（必須設定）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `VIDEO_FOLDER` | `./input_videos` | 入力動画フォルダ |
| `ASR_ENGINE` | `whisper` | 音声認識エンジン（`whisper` / `vibevoice`） |
| `TTS_ENGINE` | `kokoro` | 音声合成エンジン（`kokoro` / `omnivoice`） |
| `HF_AUTH_TOKEN` | — | HuggingFaceトークン（条件付き必須） |

### 出力設定

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `ENGLISH_VOLUME` | `0.10` | 元の英語音声の音量（0.0-1.0） |
| `JAPANESE_VOLUME` | `1.00` | 日本語吹き替え音声の音量（0.0-1.0） |
| `OUTPUT_SIZE` | `720` | 出力動画の高さ（ピクセル） |
| `KEEP_TEMP` | `true` | 一時ファイルを保持するか（再開に必要） |

### Whisper設定（`ASR_ENGINE=whisper`）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisperモデル名 |
| `WHISPER_LANG` | `en` | 認識言語 |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cppインストールディレクトリ |

### VibeVoice設定（`ASR_ENGINE=vibevoice`）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | モデル名 |
| `VIBEVOICE_MAX_TOKENS` | `32768` | 最大生成トークン数 |
| `VIBEVOICE_CONTEXT` | （空） | ホットワード（固有名詞認識補助、カンマ区切り） |

### 翻訳設定（CAT-Translate-7b）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `CAT_TRANSLATE_REPO` | `mradermacher/CAT-Translate-7b-GGUF` | モデルリポジトリ |
| `CAT_TRANSLATE_FILE` | `CAT-Translate-7b.Q8_0.gguf` | モデルファイル名 |
| `CAT_TRANSLATE_N_GPU_LAYERS` | `-1` | GPU オフロード（-1で全レイヤー） |
| `CAT_TRANSLATE_RETRIES` | `3` | リトライ回数 |

### Kokoro TTS設定（`TTS_ENGINE=kokoro`）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `KOKORO_VOICE` | `jf_alpha` | 日本語音声名 |
| `KOKORO_SPEED` | `1.0` | 話速（0.8-1.2推奨） |

利用可能な音声: `jf_alpha`（女性、推奨）、`jf_gongitsune`（女性）、`jf_nezumi`（女性）、`jf_tebukuro`（女性）、`jm_kumo`（男性）

### OmniVoice設定（`TTS_ENGINE=omnivoice`）

| 設定 | デフォルト | 説明 |
|---------|---------|-------------|
| `OMNIVoice_MODEL` | `k2-fsa/OmniVoice` | OmniVoiceモデルリポジトリ |
| `OMNIVoice_SPEED` | `1.0` | 話速 |

---

## 再開機能について

処理は各ステップでチェックポイントを保存するため、途中で止まっても`uv run ja-dubbing`を再実行すれば続きから処理できます。

**最初からやり直したい場合**: `temp/<動画名>/`フォルダを削除して再実行。

**異なるエンジンでリトライしたい場合**: 同様に`temp/<動画名>/`フォルダを削除。

## ライセンス

MIT License

このツールが使用する外部モデルやライブラリ（OmniVoice、CAT-Translate-7b、pyannote.audio、whisper.cpp、Kokoro TTSなど）はそれぞれ独自のライセンスを持ちます。使用時にはそれらもご確認ください。
