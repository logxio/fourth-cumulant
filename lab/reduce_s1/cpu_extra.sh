#!/bin/bash
# reduce_s1 CPU extras on the same box: same-machine controls for the fed-back pieces (B alone, the in-budget best),
# so every chain-B-plus-pieces ratio on the curve is against chain B on this machine.
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data PYTHONUNBUFFERED=1
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
for c in base:1:all combo:4:all star:4:all mixshort:4:all:2 combo:2:all; do
  name=$(echo "$c" | tr ':' '_')
  [ -f "outputs/net0/fed_$name.json" ] && continue
  OPENBLAS_NUM_THREADS=7 OMP_NUM_THREADS=7 timeout 1800 $PY -u lab/reduce_s1/probe_feedback.py "$c" 0 "outputs/net0/fed_$name.json" > "logs/net0_fed_$name.log" 2>&1
  echo "CPU_DONE net0 $c exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
done
date -u +%FT%TZ > CPU_EXTRA_DONE
