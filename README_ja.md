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
- 全言語ペアを PLaMo 3 Translate 8B Q4_K_M で翻訳し、公式対応53言語以外は受け付けません。
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
- デフォルトの ASR パスには、公式 VibeVoice-ASR Q8 GGUF と Q8 音声プロジェクターを `llama-server` 経由で使用します。
- オプションの whisper ASR パスには whisper.cpp・Silero VAD・pyannote.audio を使用します。
- ローカルGGUF翻訳には公式の `llama.cpp`／`llama-server` とHugging Face Hubを使用します。
- 音声クローン TTS には OmniVoice・VoxCPM2・Irodori-TTS-Server を使用します。
- 音声／モデルサポートには PyTorch・torchaudio・TorchCodec・soundfile・pydub を使用します。

## 要件

- Apple Silicon 搭載の macOS。
- Python 3.13 以降。
- Homebrew パッケージ：

```bash
brew install ffmpeg cmake uv llama.cpp
```

- PLaMoの `plamo3` トークナイザーに対応する新しい `llama-server`。既存のHomebrew版が古い場合は `brew upgrade llama.cpp` で更新してください。
- 初回の依存関係およびモデルダウンロードのためのインターネット接続。
- pyannote ダイアライゼーションを使う whisper モードおよびゲート付きまたは認証が必要なモデルのダウンロードには、`HF_AUTH_TOKEN` に Hugging Face トークンが必要です。

Linux・Windows・CUDA ワークフローはこのリポジトリ構成ではサポートされていません。

依存関係の解決対象は macOS arm64 です。PyTorch `2.15.0`・TorchAudio `2.11.1`・TorchCodec `0.17.0`・Demucs `4.1.0` を固定し、FFmpeg `9.0.2` でネイティブデコードと合成メディアの検証を通過しています。FFmpeg 9 を維持してください。MLX Audio・MLX・MLX LM は不要になりました。Irodori は後述する独立した固定環境で動作します。

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

`TTS_ENGINE=irodori` の場合、生成された `start_servers.sh` は設定されたホストとポートで Irodori-TTS-Server を起動します。OmniVoiceとVoxCPM2はプロセス内で動作します。翻訳用のローカル `llama-server` はパイプラインが自動で起動・停止するため、手動起動は不要です。

Irodori のデフォルトの待受先は `127.0.0.1` です。パイプラインや生成スクリプトからループバック以外で起動する場合は `IRODORI_TTS_API_KEY` が必須で、API リクエストにも同じキーを送信します。Irodori 用スクリプトの権限は所有者限定（`0700`）で、キーを設定した場合はその値も含まれます。設定変更後は `uv run xlanguage-dubbing --generate-script` で再生成してください。既存のスクリプトや外部で管理するサーバーは自動更新しません。

## 設定

すべての実行時設定は `.env` から読み込まれます。完全なリストは [.env.example](.env.example) を参照してください。

### コア設定

| 設定 | 目的 |
|---|---|
| `VIDEO_FOLDER` | ソース動画を再帰的にスキャンするフォルダ。 |
| `TEMP_ROOT` | チェックポイントと中間出力のルート。 |
| `OUTPUT_SUFFIX` | 生成された動画に付加されるサフィックス。 |
| `KEEP_TEMP` | 合成後に各動画の一時作業ディレクトリを保持するか削除するか。 |
| `INPUT_LANG` | PLaMo対応の入力言語コード、または `auto`。 |
| `OUTPUT_LANG` | PLaMo対応の出力言語コード。TTS側の言語制限も適用されます。 |
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
| `VIBEVOICE_SERVER_BIN` | 対応版 `llama-server` のパス。空欄は一時的なプロジェクト内ビルドを優先。`llama-server` と指定すると PATH／Homebrew 版を選択。 |
| `VIBEVOICE_N_CTX` | コンテキスト長。既定 `65536`。生成上限と入力トークンを収める必要があります。 |
| `VIBEVOICE_N_GPU_LAYERS` | GPU 層数。既定 `-1` は Metal。`0` は音声プロジェクターを含む CPU 実行。 |
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
| `PLAMO_TRANSLATE_SERVER_BIN` | `llama-server` の実行ファイル名またはパス。デフォルトは `llama-server`。`/opt/homebrew/bin/llama-server` も探索します。 |
| `PLAMO_TRANSLATE_N_GPU_LAYERS` | GPUに割り当てる層数。デフォルトは全層の `-1`。 |
| `PLAMO_TRANSLATE_N_CTX` | 入出力を合わせたコンテキスト長。デフォルトは `8192`。 |
| `PLAMO_TRANSLATE_RETRIES` | 翻訳の試行回数。デフォルトは `3`。 |
| `PLAMO_TRANSLATE_RETRY_BACKOFF_SEC` | リトライ待機時間の基準秒数。デフォルトは `1.5`。 |
| `PLAMO_TRANSLATE_REPEAT_PENALTY` | 繰り返しペナルティ。デフォルトは `1.0`。 |
| `OUTPUT_REPEAT_THRESHOLD`・`INPUT_REPEAT_THRESHOLD` | 翻訳の不具合を検出するための繰り返し閾値。 |

### TTS エンジン

| 設定 | 目的 |
|---|---|
| `TTS_ENGINE` | `omnivoice`・`voxcpm2`・または `irodori`。エイリアスとして `irodori-tts` と `irodori_tts` も受け付けます。 |
| `OMNIVOICE_*` | OmniVoice のモデル・dtype・サンプリング・長さ・参照・リトライ設定。 |
| `VOXCPM2_*` | VoxCPM2 のモデル・コントローラブルクローニング・長さ・参照・リトライ設定。 |
| `IRODORI_TTS_BASE_URL` | Irodori-TTS-Server のベース URL。 |
| `IRODORI_TTS_SERVER_HOST` | サーバーの待受先。デフォルトは `127.0.0.1`。ループバック以外での起動には API キーが必須です。 |
| `IRODORI_TTS_SERVER_PORT` | サーバーのポート。デフォルトは `8088`。 |
| `IRODORI_TTS_DIR` | ローカルのサーバーフォーク。デフォルトは `./Irodori-TTS-Server-nonWatermark`。 |
| `IRODORI_TTS_AUTO_START` | サーバーが正常稼働していない場合に Irodori-TTS-Server を自動起動します。 |
| `IRODORI_TTS_START_COMMAND` | オプションのカスタムサーバー起動コマンド。 |
| `IRODORI_TTS_API_KEY` | Irodori-TTS-Server に送信する API キー。パイプラインや生成スクリプトからループバック以外で起動する場合は必須です。 |
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
| `vibevoice` | 話者情報を統合した多言語またはコードスイッチング ASR を使いたい場合。 | デフォルトモード。Apple Silicon 上で公式 Q8 GGUF と Q8 プロジェクターを使用し、pyannote をスキップします。 |
| `whisper` | VAD 付きの whisper.cpp ASR を使いたい場合。 | `scripts/setup_whisper.sh` が必要。話者 ID は pyannote.audio で割り当てられ、`HF_AUTH_TOKEN` が必要です。 |
| `nemo-speech` | NeMo の文字起こしと話者分離をまとめて使いたい場合。 | インストール済み CLI と指定の Nemotron ASR／話者分離モデルを使用。最大8人で、pyannote は使用しません。 |

VibeVoice は公式の [`VibeVoice-ASR-Q8_0.gguf` と `mmproj-VibeVoice-ASR-Q8_0.gguf`](https://huggingface.co/ggml-org/VibeVoice-ASR-GGUF/tree/main)（合計約8.94 GB）を、プロジェクト直下の `models/VibeVoice-ASR-GGUF/` に自動取得します。音声プロジェクターも必須です。両方を同じ固定リビジョンで使用します。旧 `VIBEVOICE_MODEL` 環境設定と MLX 専用のチャンク／メモリ設定は廃止しました。旧 MLX の作業フォルダーは保存され、新しい設定では再利用しません。

現在インストールされている Homebrew の `llama.cpp` 1.0.0 は [VibeVoice 対応](https://github.com/ggml-org/llama.cpp/pull/32329)より前の版です。Homebrew が対応するまで、次の一時的な専用ビルドを使用します。

```bash
bash scripts/setup_vibevoice.sh
```

公式コミット `fc504f91d8cfdfcd9b0fedcf14600fd2b0e63420` を `models/llama.cpp-vibevoice/` に固定し、Homebrew や PLaMo の実行環境は変更しません。`VIBEVOICE_SERVER_BIN` が空欄の場合はこのビルドを優先します。将来の Homebrew 版で Q8 の文字起こしと参照音声処理が実際に通ることを確認したら、`.env` に `VIBEVOICE_SERVER_BIN=llama-server` を指定して切り替えます。AI エージェント向けの移行条件は `AGENTS.md` に記載しています。

音声全体と実際の長さを `/v1/chat/completions` に渡します。24 kHz への変換と音声エンコーダーのチャンク処理はネイティブ側で行い、話者 ID を途中でリセットしません。時刻・話者 ID を検証し、トークン上限などで途切れた結果は完了扱いにしません。認証付きローカル専用サーバーは、本編 ASR 後・参照音声処理後・失敗時に終了し、翻訳や TTS の前にメモリを解放します。

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

生成声と途中結果は、翻訳設定別ディレクトリの `input_<20桁ハッシュ>/voice_design/<engine>_<language>_<settings>/` 以下に保存します。出力動画にも `_voice_design_...` が付くため、従来方式の完成動画があっても実行できます。有効な生成声は再開時に再利用されます。推定モデル追加前に作った男性の代替音声も維持するので、声を作り直す場合は `VOICE_DESIGN_SEED` を変更してください。破損した参照音声だけを再生成し、参照音声の指紋でTTS・リタイム結果を分けます。Irodoriでは短命のコーデック処理の前に、このパイプライン自身が起動したサーバーだけを停止します。外部で管理するサーバーは停止しません。

### 翻訳

言語ペアに関わらず、[公式GGUF](https://huggingface.co/pfnet/plamo-3-translate-8b-GGUF) の `plamo-3-translate-8b-Q4_K_M.gguf` を使います。リポジトリとファイル名は固定です。初回使用時に約5.28GBのモデルをプロジェクト直下の `models/plamo-3-translate-8b-GGUF/` に取得します。必要になった時点で専用の `llama-server` をlocalhostの空きポートに自動起動し、Python標準ライブラリで `/v1/completions` にPFML3プロンプトを送信します。このプロセスはTTS前と処理失敗時に停止します。

対応する言語コードは以下の53種類です。入力・出力ともにこの一覧に限定します。`INPUT_LANG=auto` は使用できますが、検出された未対応言語はエラーになります。TTSエンジンとボイスデザインの言語制限は別途適用されます。

```text
ja en zh-hans zh-hant ko ar az be bg bn ca ceb de es et eu fa fr gu he
hi hr hu hy id it jv ka kk km lt lv ml mr ms my ne nl pa pl ru si sk sl
sr sw ta te th tl ur uz vi
```

中国語は簡体字 `zh-hans` と繁体字 `zh-hant` を区別します。`zh`・`zh-cn` は簡体字、`zh-tw`・`zh-hk`・`zh-mo` は繁体字として扱います。その他の地域付きコードは基本言語コードに正規化します。

## 再開と出力

最終動画はソース動画と同じ場所に出力します。`OUTPUT_SUFFIX` のデフォルトは `_xlDub.mp4` です。全モードで翻訳設定の識別子 `plamo3_<出力言語>_<12桁ハッシュ>` と、入力・有効な解析設定・音声生成設定・合成／出力設定から決まる16桁の識別子を付けます。

| モード | 最終ファイル名 |
|---|---|
| OmniVoice／VoxCPM2 元音声クローン | `<拡張子なしソース名>_<翻訳識別子>_<16桁識別子><OUTPUT_SUFFIX>` |
| Irodori 元音声クローン | `<拡張子なしソース名>_<翻訳識別子>_<16桁識別子>_irodori_<設定ハッシュ><OUTPUT_SUFFIX>` |
| ボイスデザイン | `<拡張子なしソース名>_<翻訳識別子>_<16桁識別子>_voice_design_<エンジン>_<言語>_<設定ハッシュ><OUTPUT_SUFFIX>` |

一時ディレクトリの基点は音声ソースモードによって分離します。

- Demucs有効：`temp/<video>`
- Demucs無効：`temp/<video>_rawaudio`

NeMoでは基点の下に `nemo-speech/` を追加します。その下、または他のASRの基点の下に `<翻訳識別子>/input_<20桁ハッシュ>/` を追加し、ボイスデザインではさらに `voice_design/<エンジン>_<言語>_<設定ハッシュ>/` を追加します。入力ハッシュには解決済みのソースパス・内容・ファイル情報と、有効なASR・セグメント化・音声分離設定を含めます。同名の別動画や内容を変更した動画は別の保存先になり、以前の作業ディレクトリと出力は維持します。

この作業ディレクトリに `progress.json`・`segments_src.json`・`segments_translated.json`・`subtitles_src.srt`・`speaker_refs/` を保存します。NeMoの検証済みCLI出力 `nemo_speech.json` もここに保存します。翻訳は元セグメントに紐付く途中結果を `segments_translated.records/` に1件ずつアトミックに保存し、完了時に集約JSONを書き出します。

全TTSエンジンの元音声クローン・ボイスデザインで、`render/<指紋>/` の下に `seg_audio/`・発話別の `tts_records/`・集約した `tts_meta.json` を保存します。指紋には翻訳文・時刻・有効な生成設定と、使用する参照音声／潜在の内容や参照テキストを含めます。Irodoriの設定指紋にはサーバーの `uv.lock` ハッシュも含まれます。リタイム結果はその下の `retime/<パーツとフレーム周期の指紋>/` に分けるため、合計時間が同じでも個々の発話時間が変われば古い結合動画を再利用しません。

翻訳識別子はモデル・プロンプト形式・入力言語設定・出力言語・コンテキスト長・繰り返しペナルティから決まります。旧翻訳モデルの途中結果・完成動画は再利用せず、新しい作業ディレクトリでASRから実行します。翻訳設定を変更すると別の保存先と出力名になります。再開時には翻訳キャッシュの元文・時刻・話者・言語も照合します。

再開時はファイルの存在だけでなく、入力とキャッシュの識別情報を検証します。完成MP4をスキップするには、一致する完了マニフェストが必要です。`example.mp4` の場合は `example.complete.json` です。入力・有効な設定・出力ファイルの署名と、保持されている参照を照合します。`KEEP_TEMP=false` で意図的に一時ファイルを削除した後も、完成動画のスキップに対応します。フォルダースキャンでは生成済み出力を次の吹き替え対象から除外します。

メディアは一時保存先で検証してからアトミックに公開します。必須のTTS音声が未生成・失敗・破損した場合は完成動画を公開せず、成功済みの他の発話を保持して再試行できます。以前の結果を残して最初からやり直すには、該当する作業ディレクトリ・最終動画・完了マニフェストを入力と作業領域の外へ移してから再実行してください。最終動画とマニフェストだけを移す場合は、検証済みの音声・リタイム結果を再利用して最終合成をやり直します。

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

既存の動画は完了マニフェストが一致する場合だけスキップします（[再開と出力](#再開と出力)参照）。以前の結果を残して最終合成をやり直す場合は、該当する最終動画と `.complete.json` マニフェストを `VIDEO_FOLDER` の外へ移して再実行してください。検証済みの音声とリタイム結果は再利用します。マニフェストがない、または一致しない場合は検証し、必要な処理をやり直します。

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
