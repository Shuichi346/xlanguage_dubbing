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

`xlanguage-dubbing` は Apple Silicon Mac 向けのローカル動画吹き替えパイプラインです。元の音声を文字起こしし、ソース言語を検出または指定し、書き起こしを翻訳して、ターゲット言語でクローン音声を生成し、生成された音声に合わせて動画のタイミングを再調整し、吹き替え音声と元の音声または Demucs で分離された背景音を合成します。

このリポジトリは、モデル推論・翻訳・TTS・チェックポイント・最終レンダリングをすべてユーザーのマシン上で実行するローカルメディアワークフローを対象としています。

## 目次

- [機能](#機能)
- [動作の仕組み](#動作の仕組み)
- [技術スタック](#技術スタック)
- [要件](#要件)
- [インストール](#インストール)
- [使い方](#使い方)
- [設定](#設定)
- [エンジンについて](#エンジンについて)
- [再開と出力](#再開と出力)
- [トラブルシューティング](#トラブルシューティング)
- [開発](#開発)
- [ライセンス](#ライセンス)

## 機能

- `VIDEO_FOLDER` 以下を再帰的に検索し、動画をバッチ処理します。
- `VIDEO_FOLDER` に対応動画が存在しない場合、単一の動画ファイルパスを入力するよう促します。
- `.mp4`、`.mkv`、`.mov`、`.webm`、`.m4v` の入力形式に対応しています。
- デフォルトでは VibeVoice-ASR を使用し、話者情報を内包した多言語およびコードスイッチング転写を行います。
- オプションモードとして whisper.cpp ASR（Silero VAD と pyannote 話者ダイアライゼーション付き）も利用できます。
- Demucs の `--two-stems vocals` を使って、任意でボーカルと背景音を分離できます。
- Demucs 分離が無効の場合は、元の音声をそのまま使うフォールバックパスを保持します。
- マージルール・spaCy 文分割・文単位マージによって ASR セグメントを正規化します。
- 英日ペアには CAT-Translate を、その他の言語ペアには TranslateGemma を選択します。
- OmniVoice・VoxCPM2・Irodori-TTS-Server の 3 つの TTS エンジンを提供します。
- クローン音声合成のための話者ごと・セグメントごとの参照音声を構築します。
- 生成された TTS の長さに合わせてソース動画のタイミングを再調整します。
- 動画ごとに再開可能なチェックポイントと中間成果物を保存します。
- 対応する ASR・音声ソース・TTS の組み合わせを網羅した設定マトリクス実行ツールを含みます。

## 動作の仕組み

1. `ffprobe` で入力動画を解析します。
2. ASR／参照音声を準備します：
   - `ENABLE_AUDIO_SEPARATION=true`：Demucs が `vocals.wav` と `no_vocals.wav` を生成します。
   - `ENABLE_AUDIO_SEPARATION=false`：元メディアの音声をそのまま使用します。
3. `ASR_ENGINE` で音声を文字起こしします。
4. 話者 ID を割り当てまたは再利用し、セグメントをマージ・分割します。
5. `INPUT_LANG=auto` の場合、ソース言語を検出します。
6. 各セグメントを `OUTPUT_LANG` に翻訳します。
7. 翻訳された各セグメントに対してクローン TTS 音声を生成します。
8. タイミング再調整により、TTS のタイミングに合わせてソース動画と背景音をチャンクに分割します。
9. FFmpeg がタイミング再調整済みの動画・吹き替えトラック・背景／元音声トラックを最終 MP4 に合成します。

## 技術スタック

- Python 3.13 パッケージを `uv` で管理し、Hatchling でビルドします。
- FFmpeg と ffprobe でメディア解析・トランスコード・タイミング再調整・合成を行います。
- Demucs でボーカル／背景の分離をオプションで行います。
- spaCy で文を意識したセグメント分割を行います。
- デフォルトの ASR パスには `mlx-audio` / MLX 経由の VibeVoice-ASR を使用します。
- オプションの whisper ASR パスには whisper.cpp・Silero VAD・pyannote.audio を使用します。
- ローカル GGUF 翻訳モデルには `llama-cpp-python` と Hugging Face Hub を使用します。
- 音声クローン TTS には OmniVoice・VoxCPM2・Irodori-TTS-Server を使用します。
- 音声／モデルサポートには PyTorch・torchaudio・TorchCodec・soundfile・pydub を使用します。

## 要件

- Apple Silicon 搭載の macOS。
- Python 3.13 以降。
- Homebrew パッケージ：

```bash
brew install ffmpeg cmake uv
```

- 初回の依存関係およびモデルダウンロードのためのインターネット接続。
- pyannote ダイアライゼーションを使う whisper モードおよびゲート付きまたは認証が必要なモデルのダウンロードには、`HF_AUTH_TOKEN` に Hugging Face トークンが必要です。

Linux・Windows・CUDA ワークフローはこのリポジトリ構成ではサポートされていません。

## インストール

```bash
git clone https://github.com/Shuichi346/xlanguage-dubbing.git
cd xlanguage-dubbing
uv sync
uv run python -m spacy download en_core_web_sm
cp .env.example .env
```

実行前に `.env` を編集してください。最低限、`VIDEO_FOLDER`・`INPUT_LANG`・`OUTPUT_LANG`・`ASR_ENGINE`・`TTS_ENGINE`、および選択したエンジンに必要なモデル／トークン設定を行ってください。

whisper.cpp ASR を使う場合は、オプションのセットアップスクリプトを実行してください：

```bash
chmod +x scripts/setup_whisper.sh
./scripts/setup_whisper.sh
```

このスクリプトは `whisper.cpp` をクローンまたは更新し、`whisper-cli` をビルドし、設定された Whisper および VAD モデルをダウンロードします。

Irodori TTS を使う場合は、設定されたパスにサーバーのチェックアウトをクローンし、CPU エクストラを同期してください：

```bash
git clone https://github.com/Aratako/Irodori-TTS-Server.git
cd Irodori-TTS-Server
uv sync --extra cpu
```

デフォルトの `IRODORI_TTS_DIR=./Irodori-TTS-Server` は、このリポジトリルート内にそのチェックアウトがあることを想定しています。`IRODORI_TTS_AUTO_START=true` の場合、`TTS_ENGINE=irodori` のときにパイプラインが自動でサーバーを起動します。

## 使い方

対応動画を `VIDEO_FOLDER` に置き、以下を実行します：

```bash
uv run xlanguage-dubbing
```

CLI は `VIDEO_FOLDER` を再帰的にスキャンします。対応動画が見つからない場合は、動画ファイルのパスを直接入力するよう求められます。

ヘルパーサーバースクリプトを生成するには：

```bash
uv run xlanguage-dubbing --generate-script
```

`TTS_ENGINE=irodori` の場合、生成された `start_servers.sh` は設定されたホストとポートで Irodori-TTS-Server を起動します。OmniVoice と VoxCPM2 の場合は、外部サーバーが不要である旨が表示されます。

## 設定

すべての実行時設定は `.env` から読み込まれます。完全なリストは [.env.example](.env.example) を参照してください。

### コア設定

| 設定 | 目的 |
|---|---|
| `VIDEO_FOLDER` | ソース動画を再帰的にスキャンするフォルダ。 |
| `TEMP_ROOT` | チェックポイントと中間出力のルート。 |
| `OUTPUT_SUFFIX` | 生成された動画に付加されるサフィックス。 |
| `KEEP_TEMP` | 合成後に各動画の一時作業ディレクトリを保持するか削除するか。 |
| `INPUT_LANG` | ソース言語の ISO コード、または `auto`。 |
| `OUTPUT_LANG` | ターゲット吹き替え言語の ISO コード。 |
| `TTS_SAMPLE_RATE` | プロジェクト標準の TTS／出力音声サンプルレート。 |
| `TTS_CHANNELS` | プロジェクト標準の TTS／出力音声チャンネル数。 |
| `ORIGINAL_VOLUME` | 音声分離が無効のときの元の生音声ボリューム。 |
| `DUBBED_VOLUME` | 最終ミックスにおける吹き替え音声のボリューム。 |
| `OUTPUT_SIZE` | タイミング再調整エンコード時に使用する出力動画の高さ。 |

`ENABLE_AUDIO_SEPARATION=true` の場合、分離された背景音はフルボリュームでミックスされます。`ORIGINAL_VOLUME` は `ENABLE_AUDIO_SEPARATION=false` のとき、元の生音声にのみ適用されます。

### ASR と音声ソース

| 設定 | 目的 |
|---|---|
| `ASR_ENGINE` | `vibevoice` または `whisper`。 |
| `ENABLE_AUDIO_SEPARATION` | `true` のとき Demucs の `vocals` / `no_vocals` ステムを使用し、`false` のとき元のメディア音声を使用します。 |
| `DEMUCS_MODEL` | Demucs モデル名、デフォルトは `htdemucs_ft`。 |
| `WHISPER_MODEL` | `scripts/setup_whisper.sh` と whisper モードで使用する whisper.cpp モデル名。 |
| `WHISPER_LANG` | whisper.cpp の言語引数；通常は `INPUT_LANG` または `auto` に従います。 |
| `VAD_MODEL` | whisper.cpp VAD モデル名。 |
| `WHISPER_CPP_DIR` | ローカルの whisper.cpp チェックアウトパス。 |
| `VIBEVOICE_MODEL` | MLX VibeVoice-ASR モデル。 |
| `VIBEVOICE_MAX_TOKENS` | VibeVoice の最大生成トークン数。 |
| `VIBEVOICE_CONTEXT` | VibeVoice-ASR のオプションコンテキストプロンプト。 |
| `HF_AUTH_TOKEN` | whisper モードの pyannote ダイアライゼーションに必要な Hugging Face トークン。 |
| `PYANNOTE_MODEL` | pyannote ダイアライゼーションパイプラインモデル。 |

### セグメント化と翻訳

| 設定 | 目的 |
|---|---|
| `MIN_SEGMENT_SEC` | これより短いセグメントは TTS をスキップします。 |
| `MERGE_MAX_SEC`・`MERGE_MAX_CHARS`・`MERGE_GAP_SEC` | 初期セグメントマージの上限。 |
| `SPACY_MODEL` | 文分割に使用する spaCy モデル。 |
| `SPACY_CHUNK_MAX_SEC`・`SPACY_CHUNK_MAX_CHARS`・`SPACY_CHUNK_GAP_SEC` | spaCy 入力チャンク構築の上限。 |
| `SPACY_UNIT_MAX_SENTENCES`・`SPACY_UNIT_MERGE_MAX_CHARS`・`SPACY_UNIT_MERGE_MAX_GAP_SEC` | spaCy 分割後の文単位マージ上限。 |
| `CAT_TRANSLATE_*` | CAT-Translate の GGUF モデル・コンテキスト・リトライ・繰り返しペナルティ設定。 |
| `TRANSLATEGEMMA_*` | TranslateGemma の GGUF モデル・コンテキスト・リトライ・繰り返しペナルティ設定。 |
| `OUTPUT_REPEAT_THRESHOLD`・`INPUT_REPEAT_THRESHOLD` | 翻訳の不具合を検出するための繰り返し閾値。 |

### TTS エンジン

| 設定 | 目的 |
|---|---|
| `TTS_ENGINE` | `omnivoice`・`voxcpm2`・または `irodori`。エイリアスとして `irodori-tts` と `irodori_tts` も受け付けます。 |
| `OMNIVOICE_*` | OmniVoice のモデル・dtype・サンプリング・長さ・参照・リトライ設定。 |
| `VOXCPM2_*` | VoxCPM2 のモデル・コントローラブルクローニング・長さ・参照・リトライ設定。 |
| `IRODORI_TTS_BASE_URL` | Irodori-TTS-Server のベース URL。 |
| `IRODORI_TTS_DIR` | ローカルの Irodori-TTS-Server チェックアウト。 |
| `IRODORI_TTS_AUTO_START` | サーバーが正常稼働していない場合に Irodori-TTS-Server を自動起動します。 |
| `IRODORI_TTS_START_COMMAND` | オプションのカスタムサーバー起動コマンド。 |
| `IRODORI_TTS_API_KEY` | Irodori-TTS-Server に送信するオプションの API キー。 |
| `IRODORI_TTS_RESPONSE_FORMAT` | 音声レスポンス形式、デフォルトは `wav`。 |
| `IRODORI_TTS_SPEED` | Irodori 音声 API に送信するスピード値。 |

## エンジンについて

### ASR

| エンジン | 使用場面 | 備考 |
|---|---|---|
| `vibevoice` | 話者情報を統合した多言語またはコードスイッチング ASR を使いたい場合。 | デフォルトモード。Apple Silicon 上で MLX モデルを使用し、pyannote 話者ダイアライゼーションパスをスキップします。 |
| `whisper` | VAD 付きの whisper.cpp ASR を使いたい場合。 | `scripts/setup_whisper.sh` が必要。話者 ID は pyannote.audio で割り当てられ、`HF_AUTH_TOKEN` が必要です。 |

### TTS

| エンジン | 使用場面 | 備考 |
|---|---|---|
| `omnivoice` | デフォルトのプロセス内クローン TTS パスを使いたい場合。 | 話者ごとおよびセグメントごとの参照音声と、利用可能な場合は参照テキストも使用します。 |
| `voxcpm2` | VoxCPM2 のコントローラブルクローニング動作を使いたい場合。 | セグメントごとの `reference_wav_path` のみを渡します。プロンプト音声／テキストは VoxCPM2 に送信されません。 |
| `irodori` | Irodori-TTS-Server を通じて日本語クローン TTS を使いたい場合。 | ここでは英日ジョブに推奨。`OUTPUT_LANG=ja` の場合のみ許可され、セグメントごとの参照音声を `irodori.ref_wav` として送信します。 |

Irodori モードは意図的に Caption・Style Prompt・固定 `seconds` を送信しません。サーバーの長さ予測機能を使用します。

### 翻訳

| 言語ペア | エンジン |
|---|---|
| 英語から日本語 | CAT-Translate |
| 日本語から英語 | CAT-Translate |
| その他のペア | TranslateGemma |

両翻訳エンジンはローカル GGUF 推論で動作します。パイプラインは TTS の前に翻訳モデルを解放してメモリを確保します。

## 再開と出力

最終動画はソース動画と同じ場所に以下の名前で出力されます：

```text
<ソースファイル名><OUTPUT_SUFFIX>
```

デフォルトでは `<ソースファイル名>_xlDub.mp4` となります。

パイプラインは `TEMP_ROOT` 以下にチェックポイントと成果物を保存します：

- `progress.json`
- `segments_src.json`
- `segments_translated.json`
- `subtitles_src.srt`
- `speaker_refs/`
- `seg_audio/`
- `tts_meta.json`
- `retime/`

一時ディレクトリは音声ソースモードによって分離されます：

- Demucs 分離が有効の場合：`temp/<video>`
- Demucs 分離が無効の場合：`temp/<video>_rawaudio`

`progress.json` がソース動画のサイズおよび更新日時と一致する場合、コマンドを再実行すると完了済みの作業が再開されます。特定の動画を最初からやり直したい場合は、その動画の一時ディレクトリと既存の最終出力ファイルを削除してください。

## トラブルシューティング

### `demucs` が見つからない

以下を実行してください：

```bash
uv sync
```

Demucs がプロジェクトの依存関係に追加される前に環境が作成されていた場合、再同期することで更新されます。

### spaCy が `en_core_web_sm` を見つけられない

パイプラインはプリフライトチェックの前に spaCy を初期化します。設定されているモデルをインストールしてください：

```bash
uv run python -m spacy download en_core_web_sm
```

`.env` で `SPACY_MODEL` を変更した場合は、そのモデルをインストールしてください。

### whisper.cpp のファイルが見つからない

以下を実行してください：

```bash
./scripts/setup_whisper.sh
```

次に、`.env` で `WHISPER_CPP_DIR=./whisper.cpp` が設定されていることを確認してください。

### pyannote が `HF_AUTH_TOKEN` が未設定と報告する

`ASR_ENGINE=whisper` を使用する前に、`.env` で `HF_AUTH_TOKEN` を設定してください。VibeVoice モードは pyannote ダイアライゼーションパスを使用しません。

### Irodori-TTS-Server が起動中に終了する

`IRODORI_TTS_DIR` が期待するローカルチェックアウトを使用し、CPU エクストラが同期されていることを確認してください：

```bash
cd Irodori-TTS-Server
uv sync --extra cpu
```

デフォルトのサーバー起動コマンドは以下の通りです：

```bash
uv run python -m irodori_openai_tts --host 0.0.0.0 --port 8088
```

それでも起動に失敗する場合は、以下を確認してください：

```text
/private/tmp/xlanguage_dubbing_irodori_tts.log
```

### 出力がすでに存在する

ターゲットファイルがすでに存在する場合、パイプラインはそのソース動画をスキップします。再生成するには、既存の `<ソースファイル名><OUTPUT_SUFFIX>` ファイルを削除してください。

## 開発

パッケージの構文チェックを実行します：

```bash
uv run python -m compileall src
```

リポジトリにはまだユニットテストスイートが設定されていません。パイプライン・ASR・翻訳・TTS・FFmpeg の動作を変更した後は、短い動画サンプルで `uv run xlanguage-dubbing` を実行して機能を確認してください。

`input_videos/test.mp4` の固定サンプル動画に対して、対応する ASR・音声ソース・TTS の設定マトリクスを検証するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --clean
```

マトリクス実行ツールは `INPUT_LANG=en` と `OUTPUT_LANG=ja` を固定し、`ASR_ENGINE`・`ENABLE_AUDIO_SEPARATION`・`TTS_ENGINE` のすべての組み合わせを実行し、ケースごとのログと `summary.json` を `temp/config_matrix/` 以下に出力します。フルメディアパイプラインを起動せずに予定された組み合わせを確認するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --dry-run
```

## ライセンス

MIT ライセンス。[LICENSE](LICENSE) を参照してください。

外部モデル・モデルウェイト・サードパーティツールはそれぞれ独自のライセンスおよび利用規約に従います。