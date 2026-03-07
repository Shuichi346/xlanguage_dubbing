#!/bin/bash
# ================================================================
# whisper.cpp + VADモデル セットアップスクリプト
# ja-dubbing プロジェクト用
#
# 実行方法:
#   chmod +x scripts/setup_whisper.sh
#   ./scripts/setup_whisper.sh
#
# 前提:
#   - cmake がインストール済み（brew install cmake）
#   - git がインストール済み
# ================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WHISPER_DIR="$PROJECT_ROOT/whisper.cpp"

# デフォルトモデル
WHISPER_MODEL="${WHISPER_MODEL:-large-v3-turbo}"
VAD_MODEL="${VAD_MODEL:-silero-v6.2.0}"

echo "=== whisper.cpp セットアップ ==="
echo "プロジェクトルート: $PROJECT_ROOT"
echo "Whisper モデル: $WHISPER_MODEL"
echo "VAD モデル: $VAD_MODEL"
echo ""

# ---- 1. whisper.cpp クローン ----
if [ -d "$WHISPER_DIR" ]; then
    echo "1/4: whisper.cpp ディレクトリが既に存在します。更新します..."
    cd "$WHISPER_DIR"
    git pull --ff-only || echo "  (git pull に失敗しましたが続行します)"
else
    echo "1/4: whisper.cpp をクローン中..."
    git clone https://github.com/ggml-org/whisper.cpp.git "$WHISPER_DIR"
    cd "$WHISPER_DIR"
fi

# ---- 2. CMake ビルド ----
echo ""
echo "2/4: CMake ビルド中..."

# Apple Silicon 用に Metal を有効化
cmake -B build -DWHISPER_METAL=ON
cmake --build build -j --config Release

# ビルド結果の確認
WHISPER_CLI="$WHISPER_DIR/build/bin/whisper-cli"
if [ ! -f "$WHISPER_CLI" ]; then
    echo "エラー: whisper-cli のビルドに失敗しました。"
    echo "  $WHISPER_CLI が見つかりません。"
    exit 1
fi
echo "  ビルド完了: $WHISPER_CLI"

# ---- 3. Whisper モデルダウンロード ----
echo ""
echo "3/4: Whisper モデルをダウンロード中 ($WHISPER_MODEL)..."

MODEL_FILE="$WHISPER_DIR/models/ggml-${WHISPER_MODEL}.bin"
if [ -f "$MODEL_FILE" ]; then
    echo "  モデルは既にダウンロード済みです: $MODEL_FILE"
else
    bash "$WHISPER_DIR/models/download-ggml-model.sh" "$WHISPER_MODEL"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "エラー: モデルのダウンロードに失敗しました。"
        exit 1
    fi
    echo "  ダウンロード完了: $MODEL_FILE"
fi

# ---- 4. VAD モデルダウンロード ----
echo ""
echo "4/4: VAD モデルをダウンロード中 ($VAD_MODEL)..."

VAD_FILE="$WHISPER_DIR/models/ggml-${VAD_MODEL}.bin"
if [ -f "$VAD_FILE" ]; then
    echo "  VAD モデルは既にダウンロード済みです: $VAD_FILE"
else
    bash "$WHISPER_DIR/models/download-vad-model.sh" "$VAD_MODEL"
    if [ ! -f "$VAD_FILE" ]; then
        echo "エラー: VAD モデルのダウンロードに失敗しました。"
        exit 1
    fi
    echo "  ダウンロード完了: $VAD_FILE"
fi

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "whisper-cli: $WHISPER_CLI"
echo "Whisper モデル: $MODEL_FILE"
echo "VAD モデル: $VAD_FILE"
echo ""
echo "テスト実行:"
echo "  $WHISPER_CLI -m $MODEL_FILE --vad -vm $VAD_FILE -f samples/jfk.wav"
