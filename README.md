<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_en.md">English</a></th>
      <th style="text-align:center">日本語</th>
    </tr>
  </thead>
</table>

# ja-dubbing

これはCodex用
英語動画を、元話者の声色を維持した日本語吹き替え動画に変換するツール。

## 特徴

- **2つのASRエンジン**: whisper.cpp + Silero VAD（高速・英語主体・ハルシネーション抑制）と VibeVoice-ASR（多言語・話者分離内蔵）を切り替え可能
- **話者分離**: pyannote.audio で誰が何を喋っているかを特定（Whisper モード時）、VibeVoice-ASR は話者分離を内蔵
- **音声クローン**: MioTTS-Inference で元話者の声色を再現した日本語音声を生成
- **高品質翻訳**: plamo-translate-cli (PLaMo-2-Translate, MLX) による英日翻訳
- **動画速度調整**: 音声の速度は変えず、動画側を伸縮して自然な吹き替えを実現

## 動作環境

- macOS (Apple Silicon) — Mac mini M4 (24GB) で動作確認済み
- Python 3.13+
- ffmpeg / ffprobe
- CMake（whisper.cpp のビルドに必要）
- Ollama（MioTTS LLM バックエンド用）

> **注意**: 現時点では Linux での動作は未検証です。翻訳エンジン (plamo-translate-cli) は MLX を使用するため、Apple Silicon Mac が必須です。

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

### CMake（whisper.cpp のビルドおよび plamo-translate-cli の依存ライブラリ sentencepiece のビルドに必要）

```bash
brew install cmake
```

### uv（Python パッケージマネージャ）

```bash
brew install uv
```

> uv は pip に代わる高速な Python パッケージマネージャです。本ツールのすべての Python 操作に使用します。

### Ollama

https://ollama.com/download から macOS 版をダウンロードしてインストールしてください。

> Ollama は MioTTS の LLM バックエンドとして使用します。翻訳には使用しません。

## セットアップ

### 1. リポジトリの取得

```bash
git clone https://github.com/Shuichi346/ja-dubbing.git
cd ja-dubbing
```

### 2. HuggingFace の準備

pyannote.audio の話者分離モデルは利用規約への同意が必要な gated model です。**Whisper モード（`ASR_ENGINE=whisper`）を使う場合**は事前に以下を済ませてください。VibeVoice モードのみ使用する場合はこの手順は不要です。

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

> 補足: VibeVoice-ASR は Apple Silicon 専用の大きなパッケージ（9B パラメータモデル）です。初回実行時にモデルが自動ダウンロードされます（約5GB）。

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

### 5. MioTTS-Inference のセットアップ

```bash
git clone https://github.com/Aratako/MioTTS-Inference.git
cd MioTTS-Inference
uv sync
cd ..
```

### 6. 設定ファイルの作成

```bash
cp .env.example .env
```

`.env` を開き、以下の項目を必ず編集してください。

| 項目 | 説明 | 例 |
|------|------|----|
| `VIDEO_FOLDER` | 入力動画を置くフォルダ | `./input_videos` |
| `HF_AUTH_TOKEN` | HuggingFace トークン（Whisper モード時に必要） | `hf_xxxxxxxxxxxx` |
| `ASR_ENGINE` | ASR エンジンの選択 | `whisper` または `vibevoice` |

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

## サーバー起動

本ツールは内部で3つのサーバーを使用します。**パイプライン実行前に全て起動しておく必要があります。**

### 方法A: 起動スクリプトを使う（推奨）

```bash
uv run ja-dubbing --generate-script
./start_servers.sh
```

初回実行時に PLaMo-2-Translate の MLX モデルと MioTTS の Ollama モデルが自動でダウンロードされます。

### 方法B: 手動で起動する

ターミナルを3つ開き、それぞれで以下を実行します。

**ターミナル1: plamo-translate-cli 翻訳サーバー（MLX, 8bit）**

```bash
uv run plamo-translate server --precision 8bit
```

> 初回起動時に `mlx-community/plamo-2-translate-8bit` モデルが自動ダウンロードされます。ダウンロードが完了してから、下記の `uv run ja-dubbing` を実行してください。

**ターミナル2: MioTTS LLM バックエンド（ポート 8000）**

```bash
OLLAMA_HOST=localhost:8000 ollama serve
# 初回のみ: OLLAMA_HOST=localhost:8000 ollama pull hf.co/Aratako/MioTTS-GGUF:MioTTS-1.7B-Q8_0.gguf
```

**ターミナル3: MioTTS API サーバー（ポート 8001）**

```bash
cd MioTTS-Inference
uv run python run_server.py \
    --llm-base-url http://localhost:8000/v1 \
    --device mps \
    --max-text-length 500 \
    --port 8001
```

> **ポート構成**: 翻訳は plamo-translate-cli が MCP プロトコルで自動ポート管理します。MioTTS LLM 用 Ollama (8000) は `OLLAMA_HOST` 環境変数でポートを指定しています。

## 実行

サーバーが全て起動した状態で、別のターミナルから実行します。

```bash
uv run ja-dubbing
```

`VIDEO_FOLDER` 内の動画を自動検出し、順次処理します。出力は入力動画と同じフォルダに `*_jaDub.mp4` として保存されます。`.env` で任意の `VIDEO_FOLDER` パスを指定してください。

## 処理フロー

### Whisper モード（`ASR_ENGINE=whisper`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. whisper.cpp + Silero VAD で英語文字起こし（無音区間のハルシネーションを抑制）
3. pyannote.audio で話者分離
4. Whisper セグメントに話者 ID を割り当て
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合（話者境界を維持）
6. 話者ごとのリファレンス音声を元動画から抽出
7. plamo-translate-cli (PLaMo-2-Translate, MLX 8bit) で英日翻訳
8. MioTTS-Inference で話者クローン日本語音声を生成
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

### VibeVoice モード（`ASR_ENGINE=vibevoice`）

1. ffmpeg で動画から 16kHz mono WAV を抽出
2. VibeVoice-ASR で文字起こし + 話者分離 + タイムスタンプ取得（1パス、チャンクエンコードによるメモリ最適化）
3. （省略: VibeVoice-ASR 内蔵のため）
4. （省略: VibeVoice-ASR 内蔵のため）
5. セグメント結合 → spaCy 文分割 → 翻訳ユニット結合（話者境界を維持）
6. 話者ごとのリファレンス音声を元動画から抽出
7. plamo-translate-cli (PLaMo-2-Translate, MLX 8bit) で英日翻訳
8. MioTTS-Inference で話者クローン日本語音声を生成
9. 元動画の各区間を TTS 音声長に合わせて速度伸縮
10. 映像 + 日本語音声（+ 英語音声を薄くミックス）を合成して出力

## 再開機能

処理は各ステップでチェックポイントを保存します。中断しても再実行すれば途中から再開されます。チェックポイントは `temp/<動画名>/progress.json` に保存されます。

最初からやり直したい場合は、該当する `temp/<動画名>/` フォルダを削除してから再実行してください。

> **ASR エンジンを切り替える場合の注意**: 途中まで Whisper モードで処理した動画を VibeVoice モードでやり直すには、`temp/<動画名>/` フォルダを削除してから再実行してください。

## 設定項目

全ての設定は `.env` ファイルで管理します。`.env.example` に全設定項目とデフォルト値が記載されています。主要な設定項目は以下の通りです。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `VIDEO_FOLDER` | `./input_videos` | 入力動画フォルダ |
| `TEMP_ROOT` | `./temp` | 一時ファイルフォルダ |
| `ASR_ENGINE` | `whisper` | ASR エンジン（`whisper` または `vibevoice`） |
| `HF_AUTH_TOKEN` | (Whisper モード時は必須) | HuggingFace トークン |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper モデル名 |
| `VAD_MODEL` | `silero-v6.2.0` | VAD モデル名 |
| `WHISPER_CPP_DIR` | `./whisper.cpp` | whisper.cpp のインストール先 |
| `VIBEVOICE_MODEL` | `mlx-community/VibeVoice-ASR-8bit` | VibeVoice-ASR モデル名 |
| `VIBEVOICE_MAX_TOKENS` | `32768` | VibeVoice-ASR 最大トークン数 |
| `VIBEVOICE_CONTEXT` | (空) | VibeVoice-ASR ホットワード |
| `PLAMO_TRANSLATE_PRECISION` | `8bit` | 翻訳モデルの精度（4bit / 8bit / bf16） |
| `MIOTTS_API_URL` | `http://localhost:8001` | MioTTS API URL |
| `MIOTTS_DEVICE` | `mps` | MioTTS コーデックデバイス |
| `ENGLISH_VOLUME` | `0.10` | 英語音声の音量（0.0〜1.0） |
| `JAPANESE_VOLUME` | `1.00` | 日本語音声の音量（0.0〜1.0） |
| `OUTPUT_SIZE` | `720` | 出力動画の高さ（ピクセル） |
| `KEEP_TEMP` | `true` | 一時ファイルを保持するか |

## 既知の制限事項

- **英語と日本語の混在**: 「Omniverse、ISACsim などのシミュレーションツールを活用」のように英単語が混在すると、TTS が正しく読み上げられないことがあります。
- **処理時間**: 3分程度の動画でも数十分かかります。長尺動画はエラーになることがあるため、**あらかじめ8分程度に分割**してから処理することを推奨します。
- **翻訳品質**: ローカル LLM による翻訳のため、クラウド API と比べると精度にばらつきがあります。
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

24GB ユニファイドメモリの Mac mini M4 で動作確認済みです。メモリ節約のため、ASR モデル（Whisper / VibeVoice-ASR）と pyannote パイプラインは使用後に解放されます。VibeVoice モードではエンコーダのチャンク処理により、長時間音声でもメモリスパイクを抑制します。長時間の動画を処理する場合は `KEEP_TEMP=true` にして、中断・再開を活用してください。

### MioTTS でテキストが長すぎるエラー

MioTTS のデフォルト最大テキスト長は 300 文字です。本ツールではサーバー起動時に `--max-text-length 500` を指定し、さらにパイプライン側でも句読点位置での切り詰め処理を行っています。

### VibeVoice-ASR で「mlx-audio がインストールされていない」エラー

以下を実行してください。

```bash
uv pip install 'mlx-audio[stt]>=0.3.0'
```

### VibeVoice-ASR で文字起こし結果が空になる

音声ファイルに発話が含まれていない、または音声が短すぎる可能性があります。`VIBEVOICE_MAX_TOKENS` を増やすか、Whisper モードに切り替えて試してください。

> plamo-translate-cli は MCP プロトコルでポートを自動管理します。設定ファイルは `$TMPDIR/plamo-translate-config.json` に保存されます。

## ライセンス

MIT License

注意: 本ツールが使用する外部モデル・ライブラリにはそれぞれ固有のライセンスがあります。

- MioTTS デフォルトプリセット: T5Gemma-TTS / Gemini TTS で生成された音声を使用しており、商用利用不可
- PLaMo-2-Translate: PLaMo Community License（商用利用は要申請）
- plamo-translate-cli: Apache-2.0 License
- pyannote.audio: MIT License（モデルは gated、HuggingFace での利用規約同意が必要）
- whisper.cpp: MIT License
- Silero VAD: MIT License
- VibeVoice-ASR: MIT License
- mlx-audio: MIT License
