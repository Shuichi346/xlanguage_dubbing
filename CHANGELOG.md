# Changelog

## Unreleased

### Added

- Added optional Demucs voice/background separation before ASR and TTS reference extraction.
- Added `ENABLE_AUDIO_SEPARATION`, `DEMUCS_MODEL`, and `DEMUCS_DEVICE` environment settings; disabled mode keeps the previous raw-audio workflow.
- Added `TTS_ENGINE=irodori` for Japanese Irodori-TTS-500M-v3 voice cloning through Irodori-TTS-Server.
- Added `IRODORI_MODEL_DEVICE` and `IRODORI_CODEC_DEVICE` settings, both defaulting to `cpu`.
- Added Irodori Sway Sampling settings: `IRODORI_TTS_NUM_STEPS`, `IRODORI_TTS_T_SCHEDULE_MODE`, and `IRODORI_TTS_SWAY_COEFF`.
- Added `scripts/run_config_matrix.py` for developer verification across all supported `ASR_ENGINE`, `ENABLE_AUDIO_SEPARATION`, and `TTS_ENGINE` combinations on `input_videos/test.mp4`.

### Changed

- Changed the default Demucs model from `htdemucs_ft` to faster `htdemucs`.
- Changed `TTS_ENGINE=voxcpm2` synthesis from Ultimate Cloning to Controllable Cloning by using per-segment `reference_wav_path` without prompt audio/text.
- Separated background audio is now mixed at full volume; `ORIGINAL_VOLUME` only attenuates raw original audio when separation is disabled.
- Replaced supported `TTS_ENGINE=kokoro-fastapi` selection with `TTS_ENGINE=irodori`.
- Irodori mode uses per-segment reference audio via `irodori.ref_wav` and sends Sway Sampling options by default; caption/style prompt and fixed `seconds` options are not sent.

### Fixed

- Fixed CAT-Translate-7b GGUF inference adding a duplicate BOS token, which could reduce translation quality and cause wrong-language or prompt-contaminated output.
- Store VoxCPM2 reference metadata and segment clips under `voxcpm2_*` names instead of `omnivoice_*` names.

## [9.0.2] - 2026-04-16

### Fixed

- OmniVoice TTS エンジンで全セグメントが `'numpy.ndarray' object has no attribute 'detach'` エラーで失敗するバグを修正
  - OmniVoice の `model.generate()` は `list[np.ndarray]` を返すが、旧コードは `torch.Tensor` を前提に `.detach().cpu().float().numpy()` を呼んでいた
  - 型安全な `_to_numpy()` ヘルパーを追加し、`torch.Tensor` / `np.ndarray` / その他すべてに対応
  - 複数波形の結合を `torch.cat` から `np.concatenate` に修正
  - モデルの `sampling_rate` 属性からサンプルレートを動的に取得するよう改善

## [9.0.1] - 2026-04-14

### Fixed

- VoxCPM2 TTS エンジンで全セグメントが失敗するバグを修正
  - `VoxCPM.__init__() got an unexpected keyword argument 'device'` エラーの原因は
    voxcpm v2.0.0 の `__init__` が `device` 引数を受け付けないためだった
  - `voxcpm>=2.0.2` に最小バージョンを引き上げ（v2.0.2 で `device` 対応）
  - 古いバージョンとの互換性のためフォールバック処理も追加
- サンプルレートをモデルオブジェクトから動的に取得するよう改善
- Python バージョン要件を `>=3.13` に統一
  - `audioop-lts>=0.2.1` は Python 3.13+ 専用（3.12 では標準ライブラリの `audioop` が存在）
  - `pydub` が内部で `audioop` を使用するため、Python 3.13 環境では `audioop-lts` が必須

### Changed

- バージョンを 9.0.0 → 9.0.1 に更新

## [9.0.0] - 2026-04-14

### Added

- VoxCPM2 — 2B モデルによる TTS エンジンを追加
  - 30言語対応・48kHz ネイティブ出力
  - Ultimate Cloning モード（参照音声＋参照テキストで最高品質クローン）
  - MPS / CPU 対応（Apple Silicon Mac で動作可能）
- `TTS_ENGINE` 環境変数を追加（`omnivoice` または `voxcpm2` を選択）
- `src/xlanguage_dubbing/voxcpm2_tts.py` を新規追加
- VoxCPM2 関連の設定値を `.env.example` および `config.py` に追加

### Changed

- `pipeline.py` を TTS エンジン切り替え対応に拡張
- `cli.py` に TTS エンジン表示を追加
- `pyproject.toml` に `voxcpm` パッケージを依存に追加
- バージョンを 8.0.0 → 9.0.0 に更新

## [8.0.0] - 2026-04-09

### Added

- 多言語対応: 他言語音声 → 他言語音声の吹き替えに対応
  - `INPUT_LANG` 設定（デフォルト: `auto`）を追加。ASR の自動言語判定に対応
  - `OUTPUT_LANG` 設定を追加。出力言語を明示的に指定
  - 各セグメントに検出言語 `detected_lang` フィールドを追加
- TranslateGemma-12b-it (GGUF) による多言語翻訳エンジンを追加
  - 日英/英日ペア以外の翻訳で自動選択される（55言語対応）
  - CAT-Translate は日英/英日ペアのみに特化して継続使用
- `src/xlanguage_dubbing/lang_utils.py` を新規追加（言語検出・言語コード変換）
- `src/xlanguage_dubbing/translation/translategemma.py` を新規追加

### Changed

- プロジェクト名を `ja-dubbing` → `xlanguage-dubbing` に変更
  - パッケージ名: `ja_dubbing` → `xlanguage_dubbing`
  - CLI コマンド: `ja-dubbing` → `xlanguage-dubbing`
  - 出力サフィックス: `_jaDub.mp4` → `_xlDub.mp4`
- デフォルト ASR エンジンを `whisper` → `vibevoice` に変更
  - VibeVoice-ASR はコードスイッチング対応で多言語混在音声に最適
- 音量設定を汎用化: `ENGLISH_VOLUME` → `ORIGINAL_VOLUME`、`JAPANESE_VOLUME` → `DUBBED_VOLUME`
- whisper.cpp の言語設定を `INPUT_LANG` 連動に変更（`auto` 対応）
- セグメント JSON に `detected_lang` フィールドを追加
- バージョンを 7.0.0 → 8.0.0 に更新

## [7.0.0] - 2026-04-09

### Removed

- Kokoro TTS エンジンのサポートを完全に削除

### Changed

- TTS エンジンを OmniVoice に一本化
- バージョンを 6.0.0 → 7.0.0 に更新

## [6.0.0]

- OmniVoice + Kokoro TTS のデュアルエンジン対応
- CAT-Translate-7b によるローカル翻訳
- VibeVoice-ASR 対応
