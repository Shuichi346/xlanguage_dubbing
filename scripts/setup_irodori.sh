#!/bin/bash
# Install the pinned server fork and its locked inference fork dependencies.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
server_dir="${1:-${IRODORI_TTS_DIR:-$project_root/Irodori-TTS-Server-nonWatermark}}"
if [[ "$server_dir" != /* ]]; then
    server_dir="$project_root/$server_dir"
fi

server_source="https://github.com/Shuichi346/Irodori-TTS-Server-nonWatermark.git"
server_revision="ee9fd035cb65223927eb4058d89cb76708756732"

command -v git >/dev/null
command -v uv >/dev/null
if [[ ! -e "$server_dir" ]]; then
    git clone "$server_source" "$server_dir"
    git -C "$server_dir" checkout --detach "$server_revision"
fi

# Preserve existing upstream checkouts and local edits instead of replacing them.
server_origin="$(git -C "$server_dir" remote get-url origin)"
if [[ "${server_origin%.git}" != "${server_source%.git}" ]]; then
    echo "Choose a new server directory for $server_source; preserving $server_dir" >&2
    exit 1
fi
if [[ "$(git -C "$server_dir" rev-parse HEAD)" != "$server_revision" ]]; then
    if [[ -n "$(git -C "$server_dir" status --porcelain --untracked-files=no)" ]]; then
        echo "Server has local changes; preserving $server_dir. Choose a new directory." >&2
        exit 1
    fi
    git -C "$server_dir" fetch origin "$server_revision"
    git -C "$server_dir" checkout --detach "$server_revision"
fi
if [[ ! -f "$server_dir/pyproject.toml" ]]; then
    echo "Irodori server pyproject.toml not found: $server_dir" >&2
    exit 1
fi

# Keep installation in the server environment even from an activated project venv.
unset VIRTUAL_ENV
uv sync --project "$server_dir" --locked --extra cpu --inexact
