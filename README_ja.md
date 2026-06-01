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
  <p align="center">多言語動画を他の言語の吹き替え動画に変換するツール。<br>高精度音声クローニングを使用して、元話者の声を再現した吹き替えを作成します。</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-9.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## このツールでできること

- 多言語動画を入力し、他の言語の吹き替え動画を出力
- 高精度音声クローニングを使用して、元話者の声を模倣した吹き替えを作成
- Demucsで人声と背景音を分離し、ASR/TTSは人声だけを処理してから背景音と再合成
- 2つのTTSエンジンから選択：OmniVoiceまたはVoxCPM2（30言語、48kHz、Ultimate Cloning）
- 日本語-英語および英語-日本語翻訳には高精度なCAT-Translate-7bを、その他の言語ペアには55言語対応のTranslateGemma-12b-itを自動選択
- 自然な吹き替えのための動画速度自動調整
- 処理が中断されても途中から再開可能

### デモ動画

<a href="https://www.youtube.com/watch?v=amYVIorgOQQ">
  <img src="https://img.youtube.com/vi/amYVIorgOQQ/0.jpg" width="250" alt="Video Title">
</a>

---

## 目次

- [システム要件](#system-requirements)
- [セットアップ](#setup)
- [使用方法](#usage)
- [ASRエンジン選択](#asr-engines)
- [TTSエンジン選択](#tts-engines)
- [言語設定](#language-settings)
- [設定オプション](#configuration-options)
- [ライセンス](#license)

---

## システム要件

- **Mac (Apple Silicon)** — Mac mini M4 (24GB)でテスト済み
- **Python 3.13以上**
- Linuxは未テスト

---

## セットアップ

### 1. 必要なツールのインストール

```bash
brew install ffmpeg cmake uv
```

### 2. リポジトリのダウンロード

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
```

### 3. 依存関係のインストール

```bash
uv sync
uv pip install demucs
uv run python -m spacy download en_core_web_sm
```

### 4. 設定ファイルの作成

```bash
cp .env.example .env
```

`.env`を編集して以下を設定：

| 項目 | 説明 | 例 |
|------|-------------|---------|
| `VIDEO_FOLDER` | 吹き替えする動画が入っているフォルダ | `./input_videos` |
| `INPUT_LANG` | 元動画の音声言語（`auto`で自動検出） | `auto`, `en`, `ja` |
| `OUTPUT_LANG` | 出力する吹き替え言語 | `ja`, `en`, `fr` |
| `ASR_ENGINE` | 音声認識エンジン | `vibevoice`（推奨）, `whisper` |
| `ENABLE_AUDIO_SEPARATION` | Demucsで人声/背景音分離を使用するか | `true` |
| `DEMUCS_MODEL` | 人声/背景音の分離モデル | `htdemucs_ft` |
| `TTS_ENGINE` | 音声合成エンジン | `omnivoice`（デフォルト）, `voxcpm2`, `kokoro-fastapi` |
| `HF_AUTH_TOKEN` | HuggingFaceトークン（whisper使用時のみ） | `hf_xxxxxxxxxxxx` |

### 5. ASRエンジンのセットアップ

#### VibeVoiceモード（デフォルト/推奨）

追加セットアップは不要。初回実行時にモデルが自動ダウンロードされます。

#### Whisperモード

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

### 6. 実行

```bash
uv run xlanguage-dubbing
```

---

## ASRエンジン選択

| | VibeVoice（推奨） | Whisper |
|---|---|---|
| 速度 | 遅い | 速い |
| 多言語混合（コードスイッチング） | 対応 | 非対応（単一言語のみ） |
| 追加セットアップ | 不要 | `setup_whisper.sh`の実行が必要 |
| HuggingFaceトークン | 不要 | 必要 |

---

## TTSエンジン選択

`.env`の`TTS_ENGINE`変数を設定して音声合成エンジンを選択します。

| | OmniVoice（デフォルト） | VoxCPM2 | Kokoro-FastAPI |
|---|---|---|---|
| 言語 | 600+ | 30 | 英語→日本語のみ |
| 出力サンプルレート | 24kHz | 48kHz | API出力をプロジェクト標準FLACへ変換 |
| モデルサイズ | 小 | 2Bパラメータ | Kokoro-82M |
| クローニングモード | 音声クローニング | Ultimate Cloning（参照音声＋転写） | 非クローン・速度優先固定ボイス |
| 話者識別 | 参照音声生成に使用 | 参照音声生成に使用 | スキップ |
| 長さ制御 | 対応（目標時間） | 直接対応なし（自然な長さ） | 自然な長さ |
| VRAM使用量 | 少 | ~8GB | 少 |
| 設定 | `TTS_ENGINE=omnivoice` | `TTS_ENGINE=voxcpm2` | `TTS_ENGINE=kokoro-fastapi` |

### Kokoro-FastAPIモード

Kokoro-FastAPIはローカルのOpenAI互換TTS APIサーバーとして動作します。このプロジェクトは`KOKORO_FASTAPI_BASE_URL`の既存サーバーを再利用し、起動していない場合は`KOKORO_FASTAPI_DIR`のDirect Run環境を`uv`で起動します。

```bash
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI
uv run python -m unidic download
```

`TTS_ENGINE=kokoro-fastapi`、`INPUT_LANG=en`または`auto`、`OUTPUT_LANG=ja`で使用してください。日本語音声のvoiceはデフォルトで`jf_alpha`です。

---

## 言語設定

### INPUT_LANG

元動画の音声言語を指定します。`auto`に設定すると、ASRエンジンが自動検出します。VibeVoice-ASRはコードスイッチングに対応しているため、単一動画内で複数言語が混在していても問題なく処理できます。

### OUTPUT_LANG

出力する吹き替え動画の音声言語を明示的に指定します。ISO 639-1コード（`en`, `ja`, `fr`, `de`, `zh`, `ko`など）を使用してください。

### 自動翻訳エンジン選択

入出力言語の組み合わせに基づいて、最適な翻訳エンジンが自動選択されます。

| 言語ペア | 使用エンジン | 備考 |
|---|---|---|
| 英語 → 日本語 | CAT-Translate-7b | 日英専門、高精度 |
| 日本語 → 英語 | CAT-Translate-7b | 日英専門、高精度 |
| その他すべて | TranslateGemma-12b-it | 55言語対応 |

---

## 設定オプション

すべての設定は`.env`ファイルで管理されています。詳細は`.env.example`を参照してください。

---

## 再開機能

処理は各ステップでチェックポイントを保存するため、途中で停止しても`uv run xlanguage-dubbing`を再実行することで途中から再開できます。

**最初からやり直したい場合**：`temp/<動画名>/`フォルダを削除して再実行してください。

## ライセンス

MIT License

このツールで使用される外部モデルやライブラリは、それぞれのライセンスに従います。
