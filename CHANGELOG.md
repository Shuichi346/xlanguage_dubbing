# Changelog

## [7.0.0] - 2026-04-09

### Removed

- Kokoro TTS エンジンのサポートを完全に削除
  - `src/ja_dubbing/tts/kokoro_tts.py` を削除
  - `pyproject.toml` から `kokoro`、`misaki[ja]`、`unidic` の依存を削除
  - `.env.example` から `TTS_ENGINE` 設定と Kokoro 関連の設定を削除
  - `config.py` から `TTS_ENGINE`、`KOKORO_MODEL`、`KOKORO_VOICE`、`KOKORO_SPEED`、`KOKORO_SAMPLE_RATE` を削除
  - `cli.py` から Kokoro 用 preflight チェック（`ensure_unidic_downloaded`）を削除
  - `pipeline.py` から `_run_tts_kokoro()` と Kokoro 分岐ロジックを削除
  - `servers/health.py` から Kokoro 分岐を削除
  - `README.md` から Kokoro TTS への全言及を削除

### Changed

- TTS エンジンを OmniVoice に一本化（エンジン選択機能を廃止）
- 話者分離（pyannote.audio）は常に実行されるように変更（Kokoro 用の省略ロジックを削除）
- バージョンを 6.0.0 → 7.0.0 に更新

## [6.0.0]

- OmniVoice + Kokoro TTS のデュアルエンジン対応
- CAT-Translate-7b によるローカル翻訳
- VibeVoice-ASR 対応