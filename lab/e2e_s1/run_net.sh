#!/bin/bash
# One network end to end: chain B prefix to layer 4 (metered, writes the snapshot), then the compact K=4
# suffix under flopscope. usage: bash lab/e2e_s1/run_net.sh <index> <pair_rank> <tag> [extra chain.py args]
set -euo pipefail
cd "${WORKDIR:-$PWD}"
IDX=$1; RANK=$2; TAG=$3; shift 3
export AW_DATA="$PWD/data"
SNAP="$PWD/out/snap$IDX"
OUT="$PWD/out/$TAG"
mkdir -p "$SNAP" "$OUT"
if [ ! -f "$SNAP/B_mini${IDX}_pre4.npz" ]; then
  /usr/bin/time -v venv/bin/python lab/cand_k4x/r4/prefix_meter.py --index "$IDX" --out "$SNAP" > "$SNAP/prefix.log" 2>&1 || true
  test -f "$SNAP/B_mini${IDX}_pre4.npz"
fi
set +e
/usr/bin/time -v venv/bin/python lab/e2e_s1/chain.py --index "$IDX" --snapshots "$SNAP" --out "$OUT" \
  --pair-rank "$RANK" --meter "$@" > "$OUT/chain.log" 2>&1
echo "$?" > "$OUT/exit"
