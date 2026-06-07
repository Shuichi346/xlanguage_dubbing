<table>
  <thead>
    <tr>
      <th style="text-align:center"><a href="README_ja.md">日本語</a></th>
      <th style="text-align:center"><a href="README.md">English</a></th>
    </tr>
  </thead>
</table>

# xlanguage-dubbing

![Python](https://img.shields.io/badge/python-3.13%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

`xlanguage-dubbing` は、動画を別の言語に吹き替えた動画へ変換するツールです。音声を抽出し、文字起こしをローカルで翻訳し、複数のTTSエンジンのいずれかを使用して吹き替え音声を生成し、生成した音声に合わせて動画のタイミングを調整し、吹き替えた音声をオリジナル音声またはDemucsで分離した背景トラックと合成します。

このプロジェクトはApple Siliconのローカル環境向けに構築されており、現在はMPS/CPU推論を使用したmacOSでテストされています。

## 目次

- [機能](#機能)
- [動作環境](#動作環境)
- [インストール](#インストール)
- [使い方](#使い方)
- [設定](#設定)
- [エンジンについて](#エンジンについて)
- [再開と出力ファイル](#再開と出力ファイル)
- [トラブルシューティング](#トラブルシューティング)
- [開発](#開発)
- [ライセンス](#ライセンス)

## 機能

- 設定フォルダ内の動画をバッチ処理、またはフォルダが空の場合は単一ファイルをプロンプトで指定可能。
- VibeVoice-ASRによる多言語ASRをサポートし、whisper.cppモードもオプションで利用可能。
- 英語/日本語の組み合わせにはCAT-Translateを、その他の言語ペアにはTranslateGemmaを使用。
- 3つのTTSモードを提供:
  - `omnivoice`: デフォルトの音声クローニングエンジン。
  - `voxcpm2`: リファレンス音声と文字起こしを使用した30言語対応のVoxCPM2音声クローニング。
  - `kokoro-fastapi`: Kokoro-FastAPIを使用した、英語から日本語への高速固定音声モード。
- ASRおよびTTSリファレンス抽出前のオプションのDemucs音声/背景分離。
- 中断したジョブを再開できる動画ごとのチェックポイント機能。
- セグメントレベルのタイミング調整とFFmpegによる最終的な音声/映像のマルチプレクシング。

## 動作環境

- Apple Silicon搭載のmacOS。
- Python 3.13以上。
- Homebrewパッケージ:

```bash
brew install ffmpeg cmake uv
```

LinuxおよびCUDAを使用したワークフローは、このリポジトリの構成ではサポートされていません。

## インストール

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
uv sync
cp .env.example .env
```

実行前に `.env` を編集してください。最低限、`VIDEO_FOLDER`、`INPUT_LANG`、`OUTPUT_LANG`、`ASR_ENGINE`、`TTS_ENGINE` をジョブに合わせて設定してください。

whisper.cpp ASRを使用する場合は、オプションのセットアップスクリプトを実行してください:

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

Kokoro-FastAPI TTSを使用する場合は、設定されたパスにサーバーのチェックアウトをクローンしてください:

```bash
git clone https://github.com/remsky/Kokoro-FastAPI.git
cd Kokoro-FastAPI
uv run python -m unidic download
```

デフォルトの `KOKORO_FASTAPI_DIR=./Kokoro-FastAPI` は、このリポジトリのルート内にそのチェックアウトがあることを想定しています。

## 使い方

対応動画を `VIDEO_FOLDER` に配置して実行してください:

```bash
uv run xlanguage-dubbing
```

対応する入力拡張子は `.mp4`、`.mkv`、`.mov`、`.webm`、`.m4v` です。

`VIDEO_FOLDER` に動画がない場合、CLIは直接動画ファイルのパスを尋ねます。

ヘルパーサーバースクリプトを生成する場合:

```bash
uv run xlanguage-dubbing --generate-script
```

## 設定

すべての実行時設定は `.env` から読み込まれます。完全なリストは [.env.example](.env.example) を参照してください。

| 設定項目 | 用途 |
|---|---|
| `VIDEO_FOLDER` | ソース動画をスキャンするフォルダ。 |
| `TEMP_ROOT` | チェックポイントおよび中間出力ディレクトリ。 |
| `OUTPUT_SUFFIX` | 生成された動画に追加されるサフィックス。 |
| `INPUT_LANG` | ソース音声の言語、または `auto`。 |
| `OUTPUT_LANG` | 吹き替え後の出力言語。 |
| `ASR_ENGINE` | `vibevoice` または `whisper`。 |
| `ENABLE_AUDIO_SEPARATION` | `true` の場合、Demucsの `vocals` / `no_vocals` ステムを使用。 |
| `TTS_ENGINE` | `omnivoice`、`voxcpm2`、または `kokoro-fastapi`。 |
| `HF_AUTH_TOKEN` | ゲートまたは認証が必要なモデルのダウンロードに使用するHugging Faceトークン。 |
| `ORIGINAL_VOLUME` | 音声分離が無効の場合のオリジナル音声の音量。 |
| `DUBBED_VOLUME` | 最終ミックスにおける吹き替え音声の音量。 |

`ENABLE_AUDIO_SEPARATION=true` の場合、Demucsは分離したステムを書き出し、最終ミックスでは分離した背景音をフルボリュームで使用します。`false` の場合、パイプラインはASR・リファレンス抽出・最終背景ミックスにオリジナルのメディア音声を使用します。

## エンジンについて

### ASR

| エンジン | 推奨ケース | 備考 |
|---|---|---|
| `vibevoice` | 多言語またはコードスイッチングASRを使用したい場合。 | デフォルトモード。Apple SiliconのMLXモデルを使用。 |
| `whisper` | より高速な単一言語ASRを使用したい場合。 | `scripts/setup_whisper.sh` およびwhisper.cppアセットが必要。 |

### TTS

| エンジン | 推奨ケース | 備考 |
|---|---|---|
| `omnivoice` | デフォルトの音声クローニングパスを使用したい場合。 | スピーカーリファレンス音声を使用。 |
| `voxcpm2` | VoxCPM2 Controllable Cloningの動作を使用したい場合。 | セグメント単位のリファレンス音声を使用。 |
| `kokoro-fastapi` | 高速な英語から日本語への吹き替えを使用したい場合。 | 固定日本語音声、クローニングなし、話者識別をスキップ。 |

Kokoro-FastAPIモードは意図的に `INPUT_LANG=auto` または英語、`OUTPUT_LANG=ja` に限定されています。デフォルト音声は `jf_alpha` です。

### 翻訳

| 言語ペア | エンジン |
|---|---|
| 英語から日本語 | CAT-Translate |
| 日本語から英語 | CAT-Translate |
| その他のペア | TranslateGemma |

## 再開と出力ファイル

パイプラインは `TEMP_ROOT` 配下にチェックポイントを保存するため、コマンドを再実行すると可能な限り完了済みの作業から再開されます。

一時ディレクトリは音声モードごとに分けられます:

- Demucs分離が有効の場合: `temp/<video>`
- 分離が無効の場合: `temp/<video>_rawaudio`

特定の動画を強制的に最初から再実行するには、その動画の一時ディレクトリを削除してください。

## トラブルシューティング

### `demucs` が見つからない

以下を実行してください:

```bash
uv sync
```

DemucsがプロジェクトのDependenciesに追加される前に環境が作成されていた場合、再同期することで更新されます。

### Kokoro-FastAPIが起動中に終了する

`KOKORO_FASTAPI_DIR` で指定されたローカルチェックアウトを使用し、UniDicが準備されていることを確認してください:

```bash
cd Kokoro-FastAPI
uv run python -m unidic download
```

このプロジェクトは親の `.venv` を継承せずにKokoro-FastAPIを起動し、`jf_alpha` の日本語ウォームアップデフォルトを設定し、日本語音声リクエストには `lang_code=j` を送信します。また、ローカルのKokoro-FastAPIチェックアウトを使用することで、日本語のチャンクサイジングが英語のeSpeak音素変換器を経由しないようにします。

### WhisperモードでWhisper.cppが見つからない

以下を実行してください:

```bash
./scripts/setup_whisper.sh
```

その後、`.env` の `WHISPER_CPP_DIR=./whisper.cpp` を確認してください。

## 開発

パッケージの構文チェックを実行するには:

```bash
uv run python -m compileall src
```

リポジトリのユニットテストスイートはまだ設定されていません。機能検証を行うには、パイプライン、ASR、翻訳、TTS、またはFFmpegの動作を変更した後、短い動画サンプルに対して `uv run xlanguage-dubbing` を実行してください。

`input_videos/test.mp4` にある固定サンプル動画を使用して、サポートされているASR・音声分離・TTSの設定マトリックスを検証するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --clean
```

マトリックスランナーは `INPUT_LANG=en`、`OUTPUT_LANG=ja` を固定し、`ASR_ENGINE`、`ENABLE_AUDIO_SEPARATION`、`TTS_ENGINE` のすべての組み合わせを実行して、ケースごとのログと `summary.json` を `temp/config_matrix/` 以下に出力します。メディアパイプライン全体を起動せずに予定されている組み合わせを確認するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --dry-run
```

## ライセンス

MITライセンス。[LICENSE](LICENSE) を参照してください。

外部モデル、モデルの重み、およびサードパーティツールは、それぞれ独自のライセンスおよび使用条件に従います。
