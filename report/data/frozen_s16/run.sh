#!/bin/bash
# Portable rerun of the frozen experiments; outputs are kept separate from saved evidence.
set -eu
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi
cd "$ROOT"
exec "$PY" research/frozen_s16.py rerun "$@"
