<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_en.md">English</a></th>
      <th style="text-align:center"><a href="README.md">日本語</a></th>
    </tr>
  </thead>
</table>

<p align="center">
  <h1 align="center">xlanguage-dubbing</h1>
  <p align="center">多言語動画を他言語吹き替え動画に変換するツールです。<br>高精度な音声クローニングを使って、元の話者の声を再現した吹き替えを作成します。</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-8.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## このツールでできること

- 多言語の動画を入力して他言語の吹き替え動画を出力
- 高精度音声クローニング（OmniVoice）を使用した元話者の声を模倣した吹き替えの作成
- 日英・英日翻訳は高精度な CAT-Translate-7b、それ以外は 55 言語対応の TranslateGemma-12b-it を自動選択
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
- [ASRエンジンの選択](#asr-engines)
- [言語設定](#language-settings)
- [設定オプション](#configuration-options)
- [ライセンス](#license)

---

## システム要件

- **Mac（Apple Silicon）** — Mac mini M4（24GB）でテスト済み
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
uv run python -m spacy download en_core_web_sm
```

### 4. 設定ファイルの作成

```bash
cp .env.example .env
```

`.env` を編集して以下を設定してください。

| 項目 | 説明 | 例 |
|------|------|------|
| `VIDEO_FOLDER` | 吹き替えする動画のフォルダ | `./input_videos` |
| `INPUT_LANG` | 元動画の音声言語（`auto` で自動判定） | `auto`, `en`, `ja` |
| `OUTPUT_LANG` | 出力する吹き替え言語 | `ja`, `en`, `fr` |
| `ASR_ENGINE` | 音声認識エンジン | `vibevoice`（推奨）, `whisper` |
| `HF_AUTH_TOKEN` | HuggingFaceトークン（whisper使用時のみ） | `hf_xxxxxxxxxxxx` |

### 5. ASRエンジンのセットアップ

#### VibeVoiceモード（デフォルト・推奨）

追加セットアップ不要です。初回実行時にモデルが自動ダウンロードされます。

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

## ASRエンジンの選択

| | VibeVoice（推奨） | Whisper |
|---|---|---|
| 速度 | 低速 | 高速 |
| 多言語混在（コードスイッチング） | 対応 | 非対応（1言語のみ） |
| 追加セットアップ | 不要 | `setup_whisper.sh` 実行が必要 |
| HuggingFaceトークン | 不要 | 必要 |

---

## 言語設定

### INPUT_LANG

元の動画の音声言語を指定します。`auto` に設定すると ASR エンジンが自動で判定します。VibeVoice-ASR はコードスイッチング対応のため、1つの動画内に複数言語が混在していても問題なく処理できます。

### OUTPUT_LANG

出力される吹き替え動画の音声言語を明示的に指定します。ISO 639-1 コード（`en`, `ja`, `fr`, `de`, `zh`, `ko` など）で指定してください。

### 翻訳エンジンの自動選択

入出力言語の組み合わせに応じて、最適な翻訳エンジンが自動的に選択されます。

| 言語ペア | 使用エンジン | 備考 |
|---|---|---|
| 英語 → 日本語 | CAT-Translate-7b | 日英特化・高精度 |
| 日本語 → 英語 | CAT-Translate-7b | 日英特化・高精度 |
| その他すべて | TranslateGemma-12b-it | 55言語対応 |

---

## 設定オプション

すべての設定は `.env` ファイルで管理されます。詳細は `.env.example` を参照してください。

---

## 再開機能

処理は各ステップでチェックポイントを保存するため、途中で止まっても `uv run xlanguage-dubbing` を再実行すれば続きから処理できます。

**最初からやり直したい場合**: `temp/<動画名>/` フォルダを削除して再実行。

## ライセンス

MIT License

このツールが使用する外部モデルやライブラリはそれぞれ独自のライセンスを持ちます。
