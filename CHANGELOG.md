# Changelog

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
