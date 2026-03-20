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
  <p align="center">英語動画を、元話者の声色を維持した日本語吹き替え動画に変換するツール</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-5.0.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.13%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

---

## 目次

- [特徴](#特徴)
- [動作環境](#動作環境)
- [前提ツールのインストール](#前提ツールのインストール)
- [セットアップ](#セットアップ)
- [ASR エンジンの選択](#asr-エンジンの選択)
- [TTS エンジンの選択](#tts-エンジンの選択)
- [サーバー起動](#サーバー起動)
- [実行](#実行)
- [処理フロー](#処理フロー)
- [再開機能](#再開機能)
- [設定項目](#設定項目)
- [既知の制限事項](#既知の制限事項)
- [トラブルシューティング](#トラブルシューティング)
- [ライセンス](#ライセンス)

## 特徴

- **2つのASRエンジン**: whisper.cpp + Silero VAD（高速・英語主体・ハルシネーション抑制）と VibeVoice-ASR（多言語・話者分離内蔵）を切り替え可能
- **話者分離**: pyannote.audio で誰が何を喋っているかを特定（Whisper モード時）、VibeVoice-ASR は話者分離を内蔵
- **3つのTTSエンジン**: MioTTS-Inference（話者クローン対応・高品質・品質バリデーション付き）、GPT-SoVITS V2ProPlus（ゼロショットボイスクローン）、Kokoro TTS（高速・軽量・サーバー不要）を切り替え可能
- **高品質翻訳**: CAT-Translate-7b（GGUF, llama-cpp-python）によるプロセス内英日翻訳（サーバー不要）
- **動画速度調整**: 音声の速度は変えず、動画側を伸縮して自然な吹き替えを実現
- **再開機能**: 各ステップでチェックポイントを保存し、中断しても途中から再開可能

## 動作環境

- macOS (Apple Silicon) — Mac mini M4 (24GB) で動作確認済み
- Python 3.13+
- ffmpeg / ffprobe
- CMake（whisper.cpp のビルドに必要）
- Ollama（MioTTS LLM バックエンド用、MioTTS モードのみ）
- conda（GPT-SoVITS モードのみ）

> **注意**: 現時点では Linux での動作は未検証です。ASR（whisper.cpp, VibeVoice-ASR）および翻訳（CAT-Translate-7b）は MLX / Apple Silicon GPU を活用するため、Apple Silicon Mac が推奨です。

## 前提ツールのインストール

セットアップに入る前に、以下のツールがインストール済みであることを確認してください。

### Homebrew（未導入の場合）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### ffmpeg

```bash
brew install ffmpeg
```

### CMake（whisper.cpp のビルドおよび依存ライブラリのビルドに必要）

```bash
brew install cmake
```

### uv（Python パッケージマネージャ）

```bash
brew install uv
```

> uv は pip に代わる高速な Python パッケージマネージャです。本ツールのすべての Python 操作に使用します。

### Ollama（MioTTS モードのみ必要）

https://ollama.com/download から macOS 版をダウンロードしてインストールしてください。

> Ollama は MioTTS の LLM バックエンドとして使用します。翻訳には使用しません。Kokoro TTS モードおよび GPT-SoVITS モードでは不要です。

### conda（GPT-SoVITS モードのみ必要）

GPT-SoVITS は独立した conda 環境で動作します。miniforge の導入を推奨します。

```bash
brew install --cask miniforge
```

## セットアップ

### 1. リポジトリの取得

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 2. HuggingFace の準備

pyannote.audio の話者分離モデルは利用規約への同意が必要な gated model です。**Whisper モード（`ASR_ENGINE=whisper`）を使う場合**は事前に以下を済ませてください。VibeVoice モードのみ使用する場合、または Kokoro TTS（話者分離不要）のみ使用する場合はこの手順は不要です。

1. https://huggingface.co/settings/tokens でアクセストークンを作成（`Read` 権限で OK）
2. 以下の2つのモデルページを開き、それぞれ **利用規約に同意** してアクセスを申請
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

> 申請は通常 **即時承認** されます。承認後でないとモデルのダウンロードに失敗します。

### 3. 依存パッケージのインストール

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

> 補足: VibeVoice-ASR は Apple Silicon 専用の大きなパッケージ（9B パラメータモデル）です。初回実行時にモデルが自動ダウンロードされます（約5GB）。CAT-Translate-7b の GGUF モデルも初回実行時に huggingface_hub 経由で自動ダウンロードされます。

### 4. whisper.cpp のビルド（Whisper モード使用時）

Whisper モード（`ASR_ENGINE=whisper`）を使う場合は、whisper.cpp をソースからビルドし、Whisper モデルと VAD モデルをダウンロードする必要があります。以下のスクリプトで自動的に行えます。

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

このスクリプトは以下を実行します:

1. `whisper.cpp` リポジトリをクローン（Apple Silicon Metal GPU 対応でビルド）
2. Whisper モデル（`ggml-large-v3-turbo`）をダウンロード
3. Silero VAD モデル（`ggml-silero-v6.2.0`）をダウンロード

> VibeVoice モードのみ使用する場合はこの手順は不要です。

### 5. TTS エンジンのセットアップ

`.env` の `TTS_ENGINE` 設定に応じて、使用する TTS エンジンをセットアップしてください。

#### MioTTS モード（`TTS_ENGINE=miotts`）

話者クローン対応の高品質 TTS です。Ollama + MioTTS-Inference サーバーが必要です。

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

#### Kokoro TTS モード（`TTS_ENGINE=kokoro`）

軽量（82M パラメータ）で高速な TTS です。サーバー不要でプロセス内で直接推論します。ボイスクローンには対応していませんが、処理速度が速いため手軽に使えます。

依存パッケージは `uv sync` で自動インストールされます。**日本語の正規化に必要な unidic 辞書を追加でダウンロードしてください。**

```bash
uv run python -m unidic download
```

> **重要**: この手順を省略すると、日本語の読み上げ発音が正しく行われません。

#### GPT-SoVITS モード（`TTS_ENGINE=gptsovits`）

V2ProPlus モデルによるゼロショットボイスクローン TTS です。独立した conda 環境で動作し、API サーバー経由でアクセスします。

```bash
chmod +x scripts/setup_gptsovits.sh
./scripts/setup_gptsovits.sh
```

このスクリプトは以下を実行します:

1. GPT-SoVITS リポジトリのクローン
2. conda 環境 `gptsovits`（Python 3.11）の作成
3. PyTorch + 依存パッケージのインストール
4. NLTK データ・pyopenjtalk 辞書のダウンロード
5. 学習済みモデル（V2ProPlus）のダウンロード
6. `tts_infer.yaml` 設定ファイルの生成

> **前提**: conda (miniforge/miniconda) がインストール済みであること。GPT-SoVITS は CPU モードで動作します。

### 6. 設定ファイルの作成

```bash
cp .env.example .env
```

`.env` を開き、以下の項目を必ず編集してください。

| 項目 | 説明 | 例 |
|------|------|----|
| `VIDEO_FOLDER` | 入力動画を置くフォルダ | `./input_videos` |
| `HF_AUTH_TOKEN` | HuggingFace トークン（Whisper モード + MioTTS/GPT-SoVITS 使用時に必要） | `hf_xxxxxxxxxxxx` |
| `ASR_ENGINE` | ASR エンジンの選択 | `whisper` または `vibevoice` |
| `TTS_ENGINE` | TTS エンジンの選択 | `miotts`、`kokoro`、または `gptsovits` |

他の設定項目はデフォルトのままで動作します。詳細は後述の「設定項目」を参照してください。

吹き替えたい英語動画ファイル（.mp4, .mkv, .mov, .webm, .m4v）を `VIDEO_FOLDER` に入れてください。

## ASR エンジンの選択

`.env` の `ASR_ENGINE` で音声認識エンジンを切り替えられます。

| 項目 | `whisper` | `vibevoice` |
|------|-----------|-------------|
| エンジン | whisper.cpp CLI + Silero VAD | VibeVoice-ASR (Microsoft, mlx-audio) |
| 話者分離 | pyannote.audio（別ステップ） | 内蔵（1パス） |
| 速度 | 高速 | 低速（Whisper の数倍かかる） |
| VAD | Silero VAD 内蔵（ハルシネーション抑制） | なし |
| 対応言語 | 英語主体 | 多言語混在に強い |
| 音声上限 | 制限なし | メモリ最適化により長時間音声にも対応 |
| 追加セットアップ | `scripts/setup_whisper.sh` の実行が必要 | `mlx-audio[stt]>=0.3.0` |
| HuggingFace トークン | 必要（pyannote 用） | 不要 |

### Whisper モード（デフォルト）

```env
ASR_ENGINE=whisper
```

whisper.cpp に Silero VAD を組み合わせて高速かつ高品質に文字起こしを行い、pyannote.audio で話者分離を行います。VAD により無音区間のハルシネーション（幻聴テキスト）を抑制します。英語音声の処理に最適です。

### VibeVoice モード

```env
ASR_ENGINE=vibevoice
```

Microsoft の VibeVoice-ASR が文字起こし・話者分離・タイムスタンプを1パスで出力します。多言語混在の音声（英語 + 現地語など）に強いですが、処理速度は Whisper より遅くなります。

エンコーダのチャンク処理によるメモリ最適化を内蔵しており、24GB ユニファイドメモリの Mac でも長時間の音声を処理できます。空きメモリに基づいてチャンクサイズが自動決定されるため、ユーザー側で特別な設定は不要です。

VibeVoice モードでは pyannote.audio と HuggingFace トークンは不要です。ステップ3（話者分離）とステップ4（話者ID割り当て）は自動的にスキップされます。

#### VibeVoice-ASR 固有の設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | 使用するモデル |
| `VIBEVOICE_MAX_TOKENS` | `32768` | 最大生成トークン数 |
| `VIBEVOICE_CONTEXT` | (空) | ホットワード（固有名詞の認識補助、カンマ区切り） |

ホットワードの例:

```env
VIBEVOICE_CONTEXT=MLX, Apple Silicon, PyTorch, Transformer
```

## TTS エンジンの選択

`.env` の `TTS_ENGINE` でテキスト読み上げエンジンを切り替えられます。

| 項目 | `miotts` | `gptsovits` | `kokoro` |
|------|----------|-------------|----------|
| エンジン | MioTTS-Inference | GPT-SoVITS V2ProPlus | Kokoro TTS (82M パラメータ) |
| ボイスクローン | 対応（セグメント単位リファレンス） | 対応（ゼロショット、話者代表リファレンス） | 非対応（固定ボイス） |
| 処理速度 | 低速 | 中速 | 高速 |
| サーバー | 必要（Ollama + MioTTS API） | 必要（conda 環境 + API サーバー） | 不要（プロセス内推論） |
| 追加セットアップ | MioTTS-Inference クローン + Ollama | `scripts/setup_gptsovits.sh` + conda | `uv run python -m unidic download` |
| 音質 | 話者ごとに異なる声色で高品質 | ゼロショットで声色を再現 | 固定ボイスだが自然な発声 |
| 話者分離 | 必要 | 必要 | 不要（全話者同一声） |
| 品質バリデーション | あり（異常音声の自動検出・リトライ） | なし | なし |

### MioTTS モード（デフォルト）

```env
TTS_ENGINE=miotts
```

MioTTS-Inference による話者クローン TTS です。元の話者の声色を再現した日本語音声を生成します。セグメント単位のリファレンス音声を優先使用し、感情やテンポを反映します。Ollama（LLM バックエンド）と MioTTS API サーバーの起動が必要です。

MioTTS はスピーチトークンの生成に LLM を使用しており、LLM サンプリングパラメータ（temperature、repetition\_penalty 等）を ja-dubbing 側からリクエストごとに指定できます。デフォルト値は安定性重視にチューニング済みです。また、生成音声の長さバリデーションにより異常な音声（極端に遅い・速い・崩壊した音声）を自動検出し、パラメータを調整しながらリトライする仕組みを備えています。

### GPT-SoVITS モード

```env
TTS_ENGINE=gptsovits
```

GPT-SoVITS V2ProPlus によるゼロショットボイスクローン TTS です。3〜10秒の短い参照音声から声質を抽出し、話者ごとの代表リファレンスを使い回します。参照音声の書き起こしテキスト（prompt_text）は ASR エンジンで自動生成されます。独立した conda 環境で動作するため、ja-dubbing 本体の Python 環境には影響しません。

### Kokoro TTS モード

```env
TTS_ENGINE=kokoro
```

Kokoro は 82M パラメータの軽量オープンウェイト TTS モデルです。ボイスクローンには対応していませんが、高速に日本語音声を生成できます。サーバー起動は不要で、翻訳もプロセス内で完結します。話者分離も省略されるため、最も手軽に使えます。

#### Kokoro TTS 固有の設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `KOKORO_MODEL` | `kokoro` | モデル名 |
| `KOKORO_VOICE` | `jf_alpha` | 日本語ボイス名 |
| `KOKORO_SPEED` | `1.0` | 読み上げ速度（0.8〜1.2 推奨） |

#### 利用可能な日本語ボイス

| ボイス名 | 性別 | グレード | 説明 |
|----------|------|----------|------|
| `jf_alpha` | 女性 | C+ | 標準的な日本語女性音声（推奨） |
| `jf_gongitsune` | 女性 | C | 「ごん狐」音声データベース |
| `jf_nezumi` | 女性 | C- | 「ねずみの嫁入り」音声データベース |
| `jf_tebukuro` | 女性 | C | 「手袋を買いに」音声データベース |
| `jm_kumo` | 男性 | C- | 「蜘蛛の糸」音声データベース |

#### GPT-SoVITS 固有の設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API URL |
| `GPTSOVITS_CONDA_ENV` | `gptsovits` | conda 環境名 |
| `GPTSOVITS_DIR` | `./GPT-SoVITS` | インストール先ディレクトリ |
| `GPTSOVITS_TEXT_LANG` | `ja` | 合成テキストの言語 |
| `GPTSOVITS_PROMPT_LANG` | `en` | 参照音声の言語 |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | 読み上げ速度 |
| `GPTSOVITS_REPETITION_PENALTY` | `1.35` | 繰り返し抑制ペナルティ |
| `GPTSOVITS_REFERENCE_MIN_SEC` | `3.0` | 参照音声の最短秒数 |
| `GPTSOVITS_REFERENCE_MAX_SEC` | `10.0` | 参照音声の最長秒数 |
| `GPTSOVITS_REFERENCE_TARGET_SEC` | `5.0` | 参照音声の目標秒数 |

## サーバー起動

TTS エンジンによって必要なサーバーが異なります。翻訳は CAT-Translate-7b でプロセス内推論するためサーバーは不要です。

### Kokoro TTS モードの場合

**外部サーバーの起動は不要です。** 翻訳（CAT-Translate-7b）も TTS（Kokoro）もプロセス内で動作します。

```bash
uv run ja-dubbing
```

### MioTTS モードの場合

#### 方法A: 起動スクリプトを使う（推奨）

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

MioTTS の Ollama + API サーバーが起動されます。初回実行時に Ollama モデルが自動ダウンロードされます。

#### 方法B: 手動で起動する（ターミナル2つ）

**ターミナル1: MioTTS LLM バックエンド（ポート 8000）**

```bash
OLLAMA_HOST=localhost:8000 ollama serve
# 初回のみ: OLLAMA_HOST=localhost:8000 ollama pull hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf
```

**ターミナル2: MioTTS API サーバー（ポート 8001）**

```bash
cd MioTTS-Inference
uv run python run_server.py \
    --llm-base-url http://localhost:8000/v1 \
    --device mps \
    --max-text-length 500 \
    --port 8001
```

> **補足**: MioTTS サーバー側のサンプリングパラメータ（`MIOTTS_LLM_TEMPERATURE` 等）はサーバーのデフォルト値として設定されますが、ja-dubbing はリクエストごとにパラメータを送信するため、サーバー側の設定を変更する必要はありません。

### GPT-SoVITS モードの場合

#### 方法A: 起動スクリプトを使う（推奨）

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

GPT-SoVITS API サーバーが conda 環境で起動されます。

#### 方法B: 手動で起動する（ターミナル1つ）

```bash
conda activate gptsovits
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

## 実行

サーバーが必要な TTS エンジンの場合は起動済みの状態で、別のターミナルから実行します。

```bash
uv run ja-dubbing
```

`VIDEO_FOLDER` 内の動画を自動検出し、順次処理します。出力は入力動画と同じフォルダに `*_jaDub.mp4` として保存されます。`.env` で任意の `VIDEO_FOLDER` パスを指定してください。

### 起動スクリプトの生成

```bash
uv run ja-dubbing --generate-script
```

TTS エンジンの設定に応じて、適切なサーバー起動スクリプト `start_servers.sh` が自動生成されます。

## 処理フロー

### Whisper モード + MioTTS（`ASR_ENGINE=whisper`, `TTS_ENGINE=miotts`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. whisper.cpp + Silero VAD で英語文字起こし（無音区間のハルシネーションを抑制）
3. pyannote.audio で話者分離
4. Whisper セグメントに話者 ID を割り当て
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合（話者境界を維持）
6. 話者ごとの代表リファレンス音声 + セグメント単位リファレンス音声を元動画から抽出
7. CAT-Translate-7b (GGUF, llama-cpp-python) で英日翻訳（プロセス内推論）
8. MioTTS-Inference で話者クローン日本語音声を生成（セグメント単位リファレンス優先、品質バリデーション・自動リトライ付き）
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

### Whisper モード + GPT-SoVITS（`ASR_ENGINE=whisper`, `TTS_ENGINE=gptsovits`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. whisper.cpp + Silero VAD で英語文字起こし
3. pyannote.audio で話者分離
4. Whisper セグメントに話者 ID を割り当て
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合
6. 話者ごとの代表リファレンス音声（3〜10秒）を抽出 + ASR で書き起こし
7. CAT-Translate-7b (GGUF, llama-cpp-python) で英日翻訳（プロセス内推論）
8. GPT-SoVITS V2ProPlus でゼロショットボイスクローン日本語音声を生成
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

### Whisper モード + Kokoro（`ASR_ENGINE=whisper`, `TTS_ENGINE=kokoro`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. whisper.cpp + Silero VAD で英語文字起こし（無音区間のハルシネーションを抑制）
3. 話者分離: 省略（Kokoro はクローン非対応のため pyannote 不使用）
4. 全セグメントに統一話者IDを付与
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合
6. リファレンス音声抽出は省略（Kokoro はクローン非対応）
7. CAT-Translate-7b (GGUF, llama-cpp-python) で英日翻訳（プロセス内推論）
8. Kokoro TTS で日本語音声を高速生成
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

### VibeVoice モード（`ASR_ENGINE=vibevoice`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. VibeVoice-ASR で文字起こし + 話者分離 + タイムスタンプ取得（1パス、チャンクエンコードによるメモリ最適化）
3. （省略: VibeVoice-ASR 内蔵のため）
4. （省略: VibeVoice-ASR 内蔵のため）
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合（話者境界を維持）
6. MioTTS: 話者ごとのリファレンス音声を元動画から抽出 / GPT-SoVITS: 代表リファレンス（3〜10秒）を抽出 / Kokoro: 省略
7. CAT-Translate-7b (GGUF, llama-cpp-python) で英日翻訳（プロセス内推論）
8. MioTTS: 話者クローン日本語音声を生成（品質バリデーション・自動リトライ付き） / GPT-SoVITS: ゼロショットクローン音声を生成 / Kokoro: 日本語音声を高速生成
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

## 再開機能

処理は各ステップでチェックポイントを保存します。中断しても再実行すれば途中から再開されます。チェックポイントは `temp/<動画名>/progress.json` に保存されます。

最初からやり直したい場合は、該当する `temp/<動画名>/` フォルダを削除してから再実行してください。

> **ASR エンジンや TTS エンジンを切り替える場合の注意**: 途中まで処理した動画を別のエンジンでやり直すには、`temp/<動画名>/` フォルダを削除してから再実行してください。

> **MioTTS の品質パラメータを変更した場合**: `MIOTTS_LLM_TEMPERATURE` 等を変更した後、TTS ステップだけやり直すには `temp/<動画名>/tts_meta.json` と `temp/<動画名>/seg_audio/` を削除してから再実行してください。

## 設定項目

全ての設定は `.env` ファイルで管理します。`.env.example` に全設定項目とデフォルト値が記載されています。主要な設定項目は以下の通りです。

### 基本設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `VIDEO_FOLDER` | `./input_videos` | 入力動画フォルダ |
| `TEMP_ROOT` | `./temp` | 一時ファイルフォルダ |
| `ASR_ENGINE` | `whisper` | ASR エンジン（`whisper` または `vibevoice`） |
| `TTS_ENGINE` | `miotts` | TTS エンジン（`miotts`、`kokoro`、または `gptsovits`） |
| `HF_AUTH_TOKEN` | (Whisper モード + クローン対応 TTS 使用時は必須) | HuggingFace トークン |

### ASR 設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper モデル名 |
| `WHISPER_LANG` | `en` | Whisper 認識言語 |
| `VAD_MODEL` | `silero-v6.2.0` | VAD モデル名 |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp のインストール先 |
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | VibeVoice-ASR モデル名 |
| `VIBEVOICE_MAX_TOKENS` | `32768` | VibeVoice-ASR 最大トークン数 |
| `VIBEVOICE_CONTEXT` | (空) | VibeVoice-ASR ホットワード |

### 翻訳設定（CAT-Translate-7b）

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `CAT_TRANSLATE_REPO` | `mradermacher/CAT-Translate-7b-GGUF` | GGUF モデルの HuggingFace リポジトリ |
| `CAT_TRANSLATE_FILE` | `CAT-Translate-7b.Q8_0.gguf` | GGUF ファイル名 |
| `CAT_TRANSLATE_N_GPU_LAYERS` | `-1` | GPU オフロードレイヤー数（-1 で全レイヤー） |
| `CAT_TRANSLATE_N_CTX` | `4096` | コンテキストウィンドウサイズ |
| `CAT_TRANSLATE_RETRIES` | `3` | 翻訳リトライ回数 |
| `CAT_TRANSLATE_REPEAT_PENALTY` | `1.2` | 繰り返し抑制ペナルティ |

### TTS 設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_DEVICE` | `mps` | MioTTS コーデックデバイス |
| `MIOTTS_REFERENCE_MAX_SEC` | `20.0` | MioTTS リファレンス音声上限（秒） |
| `MIOTTS_TTS_RETRIES` | `2` | MioTTS HTTP エラー時のリトライ回数 |
| `MIOTTS_QUALITY_RETRIES` | `2` | MioTTS 品質バリデーション失敗時のリトライ回数 |
| `KOKORO_MODEL` | `kokoro` | Kokoro モデル名 |
| `KOKORO_VOICE` | `jf_alpha` | Kokoro 日本語ボイス |
| `KOKORO_SPEED` | `1.0` | Kokoro 読み上げ速度 |
| `GPTSOVITS_API_URL` | `http://127.0.0.1:9880` | GPT-SoVITS API URL |
| `GPTSOVITS_SPEED_FACTOR` | `1.0` | GPT-SoVITS 読み上げ速度 |

### MioTTS LLM サンプリングパラメータ

MioTTS はスピーチトークンの生成に LLM を使用しており、以下のパラメータで生成の安定性・多様性を制御できます。デフォルト値は安定性重視にチューニング済みです。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `MIOTTS_LLM_TEMPERATURE` | `0.5` | 温度パラメータ。低いほど安定した音声を生成（0.1〜0.8） |
| `MIOTTS_LLM_TOP_P` | `1.0` | トークン選択の多様性（1.0 でフルサンプリング） |
| `MIOTTS_LLM_MAX_TOKENS` | `700` | 最大生成トークン数 |
| `MIOTTS_LLM_REPETITION_PENALTY` | `1.1` | トークン繰り返し抑制（1.0 で無効、1.1〜1.3 推奨） |
| `MIOTTS_LLM_PRESENCE_PENALTY` | `0.0` | 既出トークンの再出現を抑制（0.0〜1.0） |
| `MIOTTS_LLM_FREQUENCY_PENALTY` | `0.3` | 高頻度トークンを抑制（0.0〜1.0、0.2〜0.4 推奨） |

### MioTTS 品質バリデーション

生成音声の長さをテキスト文字数で割った「1文字あたり秒数」を検査し、異常な音声を自動検出します。検出時はパラメータを調整して自動リトライします。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `MIOTTS_DURATION_PER_CHAR_MIN` | `0.05` | 1文字あたり秒数の下限（これ未満は「異常に短い」） |
| `MIOTTS_DURATION_PER_CHAR_MAX` | `0.5` | 1文字あたり秒数の上限（これ超過は「異常に長い」） |
| `MIOTTS_VALIDATION_MIN_CHARS` | `4` | バリデーション対象の最小文字数（短すぎるテキストはスキップ） |

### 翻訳異常検出設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `OUTPUT_REPEAT_THRESHOLD` | `3` | 翻訳出力の繰り返し検出閾値 |
| `INPUT_REPEAT_THRESHOLD` | `4` | 翻訳入力の繰り返し検出閾値 |
| `INPUT_UNIQUE_RATIO_THRESHOLD` | `0.3` | 翻訳入力のユニーク率閾値 |

### 出力設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `ENGLISH_VOLUME` | `0.10` | 英語音声の音量（0.0〜1.0） |
| `JAPANESE_VOLUME` | `1.00` | 日本語音声の音量（0.0〜1.0） |
| `OUTPUT_SIZE` | `720` | 出力動画の高さ（ピクセル） |
| `KEEP_TEMP` | `true` | 一時ファイルを保持するか |

## 既知の制限事項

- **英語と日本語の混在**: 「Omniverse、ISACsim などのシミュレーションツールを活用」のように英単語が混在すると、TTS が正しく読み上げられないことがあります。
- **処理時間**: 3分程度の動画でも数十分かかります。長尺動画はエラーになることがあるため、**あらかじめ8分程度に分割**してから処理することを推奨します。
- **翻訳品質**: ローカル LLM（CAT-Translate-7b）による翻訳のため、クラウド API と比べると精度にばらつきがあります。
- **MioTTS の音声品質**: MioTTS はボイスクローンの品質が高い一方、LLM ベースのトークン生成のため確率的に不安定な音声（極端に遅い、謎の音声になる等）が生成されることがあります。品質バリデーションと自動リトライで大半は改善されますが、完全には防げません。品質が安定しない場合は後述のトラブルシューティングを参照してください。
- **Kokoro TTS**: ボイスクローンには対応していないため、全話者が同一の声になります。速度を重視する場合や、声色の再現が不要な場合に適しています。
- **GPT-SoVITS**: CPU モードで動作するため、MPS / CUDA 環境と比べて推論が遅くなります。参照音声は声質（音色）のみ抽出されるため、抑揚やテンポは反映されません。
- **VibeVoice-ASR の処理速度**: Whisper と比較して数倍の時間がかかります。
- **VibeVoice-ASR のメモリ使用量**: 9B パラメータモデルのため、8bit 量子化でも約5GB のメモリを使用します。チャンクエンコードによるメモリ最適化を内蔵しており、24GB ユニファイドメモリの Mac で動作確認済みです。

## トラブルシューティング

### whisper-cli が見つからない

`scripts/setup_whisper.sh` を実行して whisper.cpp をビルドしてください。

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

CMake がインストールされていない場合は、先に `brew install cmake` を実行してください。

### メモリ不足

24GB ユニファイドメモリの Mac mini M4 で動作確認済みです。メモリ節約のため、ASR モデル（Whisper / VibeVoice-ASR）と pyannote パイプラインは使用後に解放されます。VibeVoice モードではエンコーダのチャンク処理により、長時間音声でもメモリスパイクを抑制します。MLX と PyTorch MPS のキャッシュも各ステップ後にクリアされます。長時間の動画を処理する場合は `KEEP_TEMP=true` にして、中断・再開を活用してください。

### MioTTS でテキストが長すぎるエラー

MioTTS のデフォルト最大テキスト長は 300 文字です。本ツールではサーバー起動時に `--max-text-length 500` を指定し、さらにパイプライン側でも句読点位置での切り詰め処理を行っています。

### MioTTS で音声が不安定（スロー・崩壊する）

MioTTS は LLM ベースのトークン生成を行うため、確率的に不安定な音声が生成されることがあります。品質バリデーションと自動リトライが組み込まれていますが、それでも改善しない場合は `.env` で以下のパラメータを調整してください。

```env
# temperature を下げる（安定性向上、表現力は低下）
MIOTTS_LLM_TEMPERATURE=0.3

# repetition_penalty を上げる（トークンループをさらに抑制）
MIOTTS_LLM_REPETITION_PENALTY=1.2

# 品質リトライ回数を増やす
MIOTTS_QUALITY_RETRIES=3

# 品質判定の上限を厳しくする（異常に長い音声をより積極的に排除）
MIOTTS_DURATION_PER_CHAR_MAX=0.4
```

変更後、TTS ステップだけやり直す場合は以下を実行してください。

```bash
rm -f temp/<動画名>/tts_meta.json
rm -rf temp/<動画名>/seg_audio/
uv run ja-dubbing
```

ログに以下のような品質リトライメッセージが出力されていれば、バリデーションが正しく動作しています。

```
    品質リトライ 1/3: 音声が異常に長い: 15.2秒 / 8文字 = 1.900秒/文字 (上限: 0.500秒/文字) → 1秒後に再生成
```

### Kokoro TTS で日本語の発音がおかしい

unidic 辞書がダウンロードされていない可能性があります。以下を実行してください。

```bash
uv run python -m unidic download
```

### VibeVoice-ASR で「mlx-audio がインストールされていない」エラー

以下を実行してください。

```bash
uv pip install 'mlx-audio[stt]>=0.3.0'
```

### VibeVoice-ASR で文字起こし結果が空になる

音声ファイルに発話が含まれていない、または音声が短すぎる可能性があります。`VIBEVOICE_MAX_TOKENS` を増やすか、Whisper モードに切り替えて試してください。

### CAT-Translate-7b のモデルダウンロードに失敗する

huggingface_hub 経由で自動ダウンロードされます。ネットワーク接続を確認し、再実行してください。モデルは `mradermacher/CAT-Translate-7b-GGUF` からダウンロードされます。

### GPT-SoVITS API サーバーに接続できない

conda 環境が正しくセットアップされているか確認してください。

```bash
conda activate gptsovits
cd GPT-SoVITS
python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml
```

セットアップ前の場合は `scripts/setup_gptsovits.sh` を先に実行してください。

## ライセンス

MIT License

注意: 本ツールが使用する外部モデル・ライブラリにはそれぞれ固有のライセンスがあります。

- MioTTS デフォルトプリセット
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