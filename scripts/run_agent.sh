#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
if [[ -z "${PYTHON_BIN:-}" && -x "$repo_dir/.venv/bin/python" ]]; then
    python_bin="$repo_dir/.venv/bin/python"
fi
export PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m phoenix.cli "$@"
