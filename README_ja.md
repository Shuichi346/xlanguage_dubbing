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

## デモ動画

英語の短い動画を日本語に翻訳・吹き替えした結果です。

<a href="https://youtube.com/shorts/amYVIorgOQQ?si=FEm-2MLur4STlzdu">
  <img src="https://img.youtube.com/vi/amYVIorgOQQ/hqdefault.jpg" alt="英語から日本語への吹き替えデモ動画" width="250">
</a>

## 目次

- [デモ動画](#デモ動画)
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
- NeMo-Speech.cpp による文字起こしと、最大8人の話者分離を選択できます。
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

Irodori TTS を使う場合は、新規・既存環境ともに、このリポジトリのルートでセットアップ用スクリプトを実行してください：

```bash
bash scripts/setup_irodori.sh
```

このスクリプトは [Shuichi346/Irodori-TTS-Server-nonWatermark](https://github.com/Shuichi346/Irodori-TTS-Server-nonWatermark) をクローンし、コミット `ee9fd035cb65223927eb4058d89cb76708756732` に固定します。サーバーに同梱されたロックファイルを `uv sync --locked --extra cpu --inexact` でインストールし、推論ライブラリにも [Irodori-TTS-nonWatermark](https://github.com/Shuichi346/Irodori-TTS-nonWatermark) のコミット `0e984c21294ab699476c4b78e2f117b07a485163` を使用します。サーバー・推論ライブラリともに nonWatermark フォークです。SilentCipher の初期化・ウォーターマーク埋め込み・関連依存パッケージを削除しています。サーバー API と MeanFlow のモデルは変わりません。処理速度の改善量は未測定です。

従来の `./Irodori-TTS-Server` は残します。セットアップ先が別リポジトリの既存チェックアウトだった場合、スクリプトは origin を変更せず終了します。新しいデフォルトの場所、またはサーバーフォークを別途クローンした場所を使用してください。変更のないフォークのチェックアウトは、固定したサーバーコミットに更新します。

サーバーの場所を変更している場合は、`bash scripts/setup_irodori.sh /absolute/path/Irodori-TTS-Server-nonWatermark` のように、`IRODORI_TTS_DIR` と同じパスを第1引数に指定してください。シェルでエクスポートした `IRODORI_TTS_DIR` も使えますが、このスクリプトは `.env` を読み込みません。モデル名を変えるだけでは推論ライブラリは更新されません。[Irodori の移行設定](#irodori-meanflow)と[実行環境の詳細](NOTES.md#2026-10-23)も参照してください。

デフォルトの `IRODORI_TTS_DIR=./Irodori-TTS-Server-nonWatermark` は、このリポジトリルート内にそのチェックアウトがあることを想定しています。`TTS_ENGINE=irodori` では、まずその環境の短命プロセスで未生成の話者参照潜在キャッシュを作成します。`IRODORI_TTS_AUTO_START=true` の場合は、その後に長時間動作するサーバーを自動起動します。

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
| `ASR_ENGINE` | `vibevoice`、`whisper`、`nemo-speech`。 |
| `ENABLE_AUDIO_SEPARATION` | `true` のとき Demucs の `vocals` / `no_vocals` ステムを使用し、`false` のとき元のメディア音声を使用します。 |
| `DEMUCS_MODEL` | Demucs モデル名、デフォルトは `htdemucs`。品質優先の場合は fine-tuned モデルの `htdemucs_ft` も使用できます。 |
| `DEMUCS_DEVICE` | Demucs の推論デバイス。Apple Silicon 向けのデフォルトは `mps`。MPS が使えない、または不安定な場合は `cpu` を指定します。 |
| `WHISPER_MODEL` | `scripts/setup_whisper.sh` と whisper モードで使用する whisper.cpp モデル名。 |
| `WHISPER_LANG` | whisper.cpp の言語引数；通常は `INPUT_LANG` または `auto` に従います。 |
| `VAD_MODEL` | whisper.cpp VAD モデル名。 |
| `WHISPER_CPP_DIR` | ローカルの whisper.cpp チェックアウトパス。 |
| `VIBEVOICE_MODEL` | MLX VibeVoice-ASR モデル。 |
| `VIBEVOICE_MAX_TOKENS` | VibeVoice の最大生成トークン数。 |
| `VIBEVOICE_CONTEXT` | VibeVoice-ASR のオプションコンテキストプロンプト。 |
| `NEMO_SPEECH_BIN` | インストール済みの NeMo-Speech.cpp 実行ファイル。デフォルトは `nemo-speech`。`~/.local/bin` も探索します。 |
| `NEMO_SPEECH_MODEL` | ASR モデルIDまたはローカル GGUF パス。デフォルトは `nvidia/nemotron-3.5-asr-streaming-0.6b`。 |
| `NEMO_SPEECH_DIAR_MODEL` | 話者分離モデルIDまたはローカル GGUF パス。デフォルトは `nvidia/Nemotron-3-Diarization`。最大8人。 |
| `NEMO_SPEECH_DEVICE` | Apple Silicon 向けの `metal`（デフォルト）、または `cpu`。 |
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
| `IRODORI_TTS_DIR` | ローカルのサーバーフォーク。デフォルトは `./Irodori-TTS-Server-nonWatermark`。 |
| `IRODORI_TTS_AUTO_START` | サーバーが正常稼働していない場合に Irodori-TTS-Server を自動起動します。 |
| `IRODORI_TTS_START_COMMAND` | オプションのカスタムサーバー起動コマンド。 |
| `IRODORI_TTS_API_KEY` | Irodori-TTS-Server に送信するオプションの API キー。 |
| `IRODORI_HF_CHECKPOINT` | Irodori-TTS-Server が使う Hugging Face チェックポイント。デフォルトは `Aratako/Irodori-TTS-v4.1-Small-MF`。 |
| `IRODORI_CODEC_REPO` | Irodori 参照潜在の生成とサーバーのデコードに使う DACVAE。デフォルトは `Aratako/Semantic-DACVAE-Japanese-32dim`。 |
| `IRODORI_MODEL_DEVICE` | Irodori モデルの実行デバイス。デフォルトは `cpu`。Pytorchのバグにより`mps`はメモリー増加で処理負荷注意。 |
| `IRODORI_CODEC_DEVICE` | Irodori コーデックの実行デバイス。デフォルトは `cpu`。Pytorchのバグにより`mps`はメモリー増加で処理負荷注意。 |
| `IRODORI_TTS_RESPONSE_FORMAT` | 音声レスポンス形式、デフォルトは `wav`。 |
| `IRODORI_TTS_SPEED` | Irodori 音声 API に送信するスピード値。 |
| `IRODORI_TTS_NUM_STEPS` | Irodori の生成ステップ数。MeanFlow 用にデフォルトは `4`。 |
| `IRODORI_TTS_T_SCHEDULE_MODE` | 従来 RF 専用のスケジュール。デフォルトは `linear`。MeanFlow では送信しません。 |
| `IRODORI_TTS_SWAY_COEFF` | 従来 RF 専用の Sway 係数。デフォルトは `-1.0`。MeanFlow では送信しません。 |
| `IRODORI_REFERENCE_MAX_SEC` | Irodori で話者ごとに連結する参照音声の上限。モデル上限の120秒以下に制限されます。 |

## エンジンについて

### ASR

| エンジン | 使用場面 | 備考 |
|---|---|---|
| `vibevoice` | 話者情報を統合した多言語またはコードスイッチング ASR を使いたい場合。 | デフォルトモード。Apple Silicon 上で MLX モデルを使用し、pyannote 話者ダイアライゼーションパスをスキップします。 |
| `whisper` | VAD 付きの whisper.cpp ASR を使いたい場合。 | `scripts/setup_whisper.sh` が必要。話者 ID は pyannote.audio で割り当てられ、`HF_AUTH_TOKEN` が必要です。 |
| `nemo-speech` | NeMo の文字起こしと話者分離をまとめて使いたい場合。 | インストール済み CLI と指定の Nemotron ASR／話者分離モデルを使用。最大8人で、pyannote は使用しません。 |

NeMo を使う場合は `.env` の `ASR_ENGINE=nemo-speech` を指定します。`nemo-speech transcribe <audio.wav> --diarize --diar-model <model> --model <model> --device metal --json` を実行し、単語ごとの時刻と話者番号を吹き替え用セグメントへ変換します。参照音声の文字起こしも NeMo を使用し、その際は話者分離を省略します。`INPUT_LANG=auto` では言語引数を渡さず、CLI が単一の言語を返した場合はそれを使用し、それ以外はテキストから言語を判定します。

インストール済みなら再インストールは不要です。未導入の場合は[公式インストール手順](https://github.com/NVIDIA/NeMo-Speech.cpp#installation)に従ってください。モデルの取得・キャッシュは CLI が管理します。必要に応じて `NEMO_SPEECH_BIN` に実行ファイルの絶対パスを設定できます。CPU で実行する場合は `NEMO_SPEECH_DEVICE=cpu` を指定します。[既定の話者分離モデル](https://huggingface.co/nvidia/Nemotron-3-Diarization)の話者枠は8人なので、9人以上の音声では同じ話者枠が再利用される可能性があります。

### TTS

| エンジン | 使用場面 | 備考 |
|---|---|---|
| `omnivoice` | デフォルトのプロセス内クローン TTS パスを使いたい場合。 | 話者ごとおよびセグメントごとの参照音声と、利用可能な場合は参照テキストも使用します。 |
| `voxcpm2` | VoxCPM2 のコントローラブルクローニング動作を使いたい場合。 | セグメントごとの `reference_wav_path` のみを渡します。プロンプト音声／テキストは VoxCPM2 に送信されません。 |
| `irodori` | Irodori-TTS-Server を通じて日本語クローン TTS を使いたい場合。 | ここでは英日ジョブに推奨。`OUTPUT_LANG=ja` の場合のみ許可され、話者ごとの参照潜在キャッシュをその話者の全セリフで再利用します。 |

### Irodori MeanFlow

Irodori は [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) を使用し、元音声クローン・初回ボイスデザインともに4ステップで生成します。既存環境では[インストール](#インストール)のコマンドでサーバー内の推論ライブラリを更新し、既存の `.env` の以下の項目を書き換えてください。

```dotenv
TTS_ENGINE=irodori
IRODORI_TTS_DIR=./Irodori-TTS-Server-nonWatermark
OUTPUT_LANG=ja
IRODORI_HF_CHECKPOINT=Aratako/Irodori-TTS-v4.1-Small-MF
IRODORI_TTS_NUM_STEPS=4
```

既存の `.env` は新しいデフォルト値より優先されます。`IRODORI_TTS_NUM_STEPS=8` を残すと8ステップのままです。MeanFlow では `IRODORI_TTS_T_SCHEDULE_MODE`・`IRODORI_TTS_SWAY_COEFF`・CFG の上書き設定を送信しません。Sway 設定は従来の RF チェックポイントを明示的に選んだ場合に使用できます。対応する制御項目は[公式 MeanFlow ガイド](https://github.com/Aratako/Irodori-TTS/blob/main/docs/meanflow.md)を参照してください。

`.env` の変更後、プロジェクト直下で起動スクリプトを再生成します。

```bash
uv run --no-sync xlanguage-dubbing --generate-script
```

手動で起動した旧サーバーがある場合は、次の実行前に停止してください。既定の `IRODORI_TTS_AUTO_START=true` なら、必要な時点でパイプラインが設定済みサーバーを起動します。手動起動する場合は、別のターミナルでプロジェクト直下から `./start_servers.sh` を実行します。異なるチェックポイントを報告するサーバーへの接続は拒否し、外部管理のサーバーは停止しません。生成されるスクリプトはモデルとステップ数の既定値を設定し、サーバー側の `IRODORI_CHECKPOINT` による上書きを解除します。

クローン合成前に、サーバー環境の短命プロセスが DACVAE コーデックだけを読み込みます。同一話者の短い発話を個別に 48 kHz mono・FP32・決定論的エンコード・`normalize_db=-16.0` で潜在化し、時系列順に結合して120秒以下に制限します。検証済みの `.pt` を話者ごとに1つ、`speaker_refs/` に atomic に保存します。元音声・選択範囲・コーデック設定・チェックポイント・サーバー依存関係のロックファイルが変わると、参照潜在キャッシュを再生成します。

各セリフには同じ話者のキャッシュを `irodori.ref_latent` として送り、`ref_wav`・Caption・Style Prompt・固定 `seconds` は送信しません。発話時間はサーバーが予測します。ボイスデザインを有効にした場合だけ、初回の参照音声生成に Caption を使います。生成声と描画キャッシュは有効なモデル・生成設定ごとに分かれ、MeanFlow で使われない Sway 設定の変更では再生成しません。保存先は[再開と出力](#再開と出力)を参照してください。

### ボイスデザインしてからクローンする

`.env` の `ENABLE_VOICE_DESIGN=true` で有効になります。話者ごとに翻訳先言語の声を1つ生成し、その声を全セリフで再利用します。4話者なら参照音声も4つ作ります。言語をまたぐクローンで元言語のアクセントが残る問題を抑えるための機能です。発音の品質はTTSモデルにも依存します。初期設定は従来の元音声クローンです。

```dotenv
ENABLE_VOICE_DESIGN=true
OUTPUT_LANG=ja
TTS_ENGINE=voxcpm2
VOICE_DESIGN_SEED=42
AGE_GENDER_MODEL_DIR=./models/age-gender
```

年齢・性別の推定には、ご指定の[公式audEERING 24層モデル](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)をHugging Faceから直接取得します。保存先はプロジェクト直下の `models/age-gender/hf24/`（`AGE_GENDER_MODEL_DIR` で変更可能）、容量は約1.27GBです。`models/` 全体をGitの管理対象から除外しています。相対パスはプロジェクト直下を基準とし、作業フォルダーが `temp/` でも判定モデルは移りません。生成した参照音声・クローン用潜在キャッシュは `temp/` に保存します。既存のTorch／Transformersで固定リビジョンのsafetensors重みをCPU推論します。Zenodoや第三者のミラーサイトからは取得しません。男女・青年／中年を推定し、不明、低信頼、子供判定、モデル未取得、音声不正の場合は大人の男性を使います。手動で互換性のある `model.onnx` を配置する場合だけ、`uv sync --extra voice-design --inexact` でaudonnxを追加します。既存のローカル6層ONNXキャッシュにも対応します。推定モデルのライセンスは **CC BY-NC-SA 4.0** で、商用利用には提供元の商用ライセンスが必要です。

VoxCPM2は声の説明文、Irodoriは初回だけCaptionを使用します。OmniVoiceは男女・青年／中年・普通の声高を属性から選択します。翻訳先は日本語・英語に限定され、Irodori v4.1 Small MFは日本語のみ対応します。ご指定の天気・駅前・青い屋根の文章を翻訳先言語で読み上げ、その生成音声またはIrodori参照潜在を以後のクローンに使います。モデルの重みを追加学習する方式ではありません。

生成声と途中結果は、音声モードごとの `voice_design/<engine>_<language>_<settings>/` 以下に保存します。出力動画にも `_voice_design_...` が付くため、従来方式の完成動画があっても実行できます。有効な生成声は再開時に再利用されます。推定モデル追加前に作った男性の代替音声も維持するので、声を作り直す場合は `VOICE_DESIGN_SEED` を変更してください。破損した参照音声だけを再生成し、参照音声の指紋でTTS・リタイム結果を分けます。Irodoriでは短命のコーデック処理の前に、このパイプライン自身が起動したサーバーだけを停止します。外部で管理するサーバーは停止しません。

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

`OUTPUT_SUFFIX` のデフォルトは `_xlDub.mp4` です。OmniVoice／VoxCPM2 の元音声クローンでは上記の名前を使い、Irodori の元音声クローンとボイスデザインでは、サフィックスの前に設定ごとの識別子を追加します。

| モード | 最終ファイル名 |
|---|---|
| Irodori 元音声クローン | `<ソースファイル名>_irodori_<設定ハッシュ><OUTPUT_SUFFIX>` |
| ボイスデザイン | `<ソースファイル名>_voice_design_<エンジン>_<言語>_<設定ハッシュ><OUTPUT_SUFFIX>` |

Irodori のモデルやステップ数を変更すると出力名も変わるため、古い動画を残したまま新しい設定で生成できます。

パイプラインは `TEMP_ROOT` 以下にチェックポイントと成果物を保存します：

- `progress.json`
- `segments_src.json`
- `segments_translated.json`
- `subtitles_src.srt`
- `speaker_refs/`（Irodori の話者別 `.pt` 参照潜在キャッシュを含む）
- `seg_audio/`
- `tts_meta.json`
- `retime/`

一時ディレクトリは音声ソースモードによって分離されます：

- Demucs 分離が有効の場合：`temp/<video>`
- Demucs 分離が無効の場合：`temp/<video>_rawaudio`

NeMo のチェックポイントと参照音声は、選択した音声モードのディレクトリ内の `nemo-speech/` に保存します。他の ASR エンジンの結果を誤って再利用しません。検証済みの CLI 出力は、このディレクトリの `nemo_speech.json` に保存します。

Irodori の元音声クローンでは、ASR・翻訳の保存先を維持し、`seg_audio/`・`tts_meta.json`・`retime/` を `render/<指紋>/` 以下に分けます。指紋には有効な生成設定・参照潜在の内容・翻訳済みセグメントを含めます。ボイスデザインでは音声／ASR ディレクトリの下に `voice_design/<エンジン>_<言語>_<設定ハッシュ>/` を追加し、生成声をその `speaker_refs/`、描画結果をその `render/<指紋>/` に保存します。Irodori の設定指紋にはサーバーの `uv.lock` ハッシュも含まれるため、固定した推論フォークを導入すると、生成声・描画キャッシュ・出力名が分かれます。旧モデルの共有 TTS・リタイムキャッシュは残りますが、移行後の Irodori では再利用しません。

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

`ASR_ENGINE=whisper` を使用する前に、`.env` で `HF_AUTH_TOKEN` を設定してください。VibeVoice と NeMo モードは pyannote ダイアライゼーションパスを使用しません。

### Irodori-TTS-Server が起動中に終了する

プロジェクト直下で、固定したサーバーフォークと、そのロックファイルに従う CPU 環境をインストールしてください：

```bash
bash scripts/setup_irodori.sh
```

`IRODORI_TTS_DIR` がデフォルトと異なる場合は、サーバーのパスを第1引数に指定してください（[インストール](#インストール)参照）。続けて[生成した起動スクリプト](#irodori-meanflow)を使ってください。パイプラインのチェックポイント・ステップ数・デバイス設定がサーバーへ渡されます。これらの環境変数を設定せずサーバーモジュールを直接起動すると、サーバー独自のデフォルト値が使われます。

それでも起動に失敗する場合は、以下を確認してください：

```text
/private/tmp/xlanguage_dubbing_irodori_tts.log
```

### 出力がすでに存在する

ターゲットファイルがすでに存在する場合、パイプラインはそのソース動画をスキップします。最終動画の合成をやり直す場合は、現在のモードに対応する完成ファイルを削除してください。Irodori やボイスデザインでは設定ごとの識別子も含む名前です。完成動画だけを削除しても、生成音声やリタイム済みのキャッシュは再利用されます。

## 開発

既存の実行環境を維持しながら、開発用の Ruff と ty をインストールして簡易チェックを実行します：

```bash
uv sync --only-group dev --inexact
uv run --no-sync ruff check src tests scripts
uv run --no-sync ty check
```

パッケージの構文チェックを実行します：

```bash
uv run python -m compileall src
```

リポジトリのユニットテストを実行します：

```bash
uv run python -m unittest discover -s tests -v
```

エンドツーエンドの機能確認では、短い動画サンプルで `uv run xlanguage-dubbing` を実行してください。

`input_videos/test.mp4` の固定サンプル動画に対して、対応する ASR・音声ソース・TTS の設定マトリクスを検証するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --clean
```

マトリクス実行ツールは `INPUT_LANG=en` と `OUTPUT_LANG=ja` を固定し、`ASR_ENGINE`・`ENABLE_AUDIO_SEPARATION`・`TTS_ENGINE` の全18通りを実行し、ケースごとのログと `summary.json` を `temp/config_matrix/` 以下に出力します。フルメディアパイプラインを起動せずに予定された組み合わせを確認するには、以下を実行してください：

```bash
./scripts/run_config_matrix.py --dry-run
```

## ライセンス

MIT ライセンス。[LICENSE](LICENSE) を参照してください。

外部モデル・モデルウェイト・サードパーティツールはそれぞれ独自のライセンスおよび利用規約に従います。
