#!/bin/bash
# Build an isolated official llama.cpp runtime with VibeVoice-ASR support.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
runtime_dir="${1:-$project_root/models/llama.cpp-vibevoice}"
runtime_source="https://github.com/ggml-org/llama.cpp.git"
runtime_revision="fc504f91d8cfdfcd9b0fedcf14600fd2b0e63420"
build_jobs="${VIBEVOICE_BUILD_JOBS:-4}"

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/setup_vibevoice.sh [runtime-directory]" >&2
    exit 1
fi
if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This setup helper requires macOS on Apple Silicon." >&2
    exit 1
fi
if [[ ! "$build_jobs" =~ ^[1-9][0-9]*$ ]]; then
    echo "VIBEVOICE_BUILD_JOBS must be a positive integer." >&2
    exit 1
fi
if [[ "$runtime_dir" != /* ]]; then
    runtime_dir="$project_root/$runtime_dir"
fi
command -v git >/dev/null
command -v cmake >/dev/null

# The runtime is independent of every Python environment in this project.
unset VIRTUAL_ENV

if [[ ! -e "$runtime_dir" ]]; then
    mkdir -p "$(dirname "$runtime_dir")"
    git clone --depth 1 --no-checkout "$runtime_source" "$runtime_dir"
    git -C "$runtime_dir" fetch --depth 1 origin "$runtime_revision"
    git -C "$runtime_dir" checkout --detach "$runtime_revision"
fi

# Never repurpose another repository or discard an existing checkout's edits.
if [[ ! -e "$runtime_dir/.git" ]]; then
    echo "Not a llama.cpp checkout; preserving $runtime_dir. Choose a new directory." >&2
    exit 1
fi
runtime_origin="$(git -C "$runtime_dir" remote get-url origin)"
if [[ "${runtime_origin%.git}" != "${runtime_source%.git}" ]]; then
    echo "Unexpected repository origin; preserving $runtime_dir. Choose a new directory." >&2
    exit 1
fi
if [[ -n "$(git -C "$runtime_dir" status --porcelain --untracked-files=no)" ]]; then
    echo "Runtime has local changes; preserving $runtime_dir. Choose a new directory." >&2
    exit 1
fi
if [[ "$(git -C "$runtime_dir" rev-parse HEAD)" != "$runtime_revision" ]]; then
    git -C "$runtime_dir" fetch --depth 1 origin "$runtime_revision"
    git -C "$runtime_dir" checkout --detach "$runtime_revision"
fi

# Models are downloaded by the Python pipeline; this server needs only local HTTP.
cmake -S "$runtime_dir" -B "$runtime_dir/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_APP=OFF \
    -DLLAMA_BUILD_UI=OFF \
    -DLLAMA_USE_PREBUILT_UI=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=ON
cmake --build "$runtime_dir/build" --target llama-server \
    --config Release --parallel "$build_jobs"

runtime_bin="$runtime_dir/build/bin/llama-server"
if [[ ! -x "$runtime_bin" ]]; then
    echo "Built llama-server not found: $runtime_bin" >&2
    exit 1
fi
"$runtime_bin" --version
echo "VibeVoice-ASR runtime ready: $runtime_bin"
echo "This isolated runtime is a temporary fallback for older Homebrew builds."
echo "After upgrading Homebrew llama.cpp to a VibeVoice-capable release,"
echo "set VIBEVOICE_SERVER_BIN=llama-server in .env to use that runtime."
