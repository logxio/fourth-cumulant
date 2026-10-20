#!/bin/bash
# reduce_s1 GPU lane 4 (after lane 3): consistent windows. Every family of the births s0..e at full life,
# without and with C.k3 births; the 20,451-unit control (early non-tree, tree and star) was such a window.
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
for i in $(seq 1440); do [ -f GPU_LANE3_DONE ] && break; sleep 10; done
gpu() {
  local net=$1 plan=$2; shift 2
  local name=$(basename "$plan" .json)
  [ -f "outputs/net$net/$name.json" ] && return 0
  timeout 2400 $PY -u lab/reduce_s1/select_run.py --plan "$plan" --index "$net" --snapshots snapshots \
    --output "outputs/net$net/$name.json" "$@" > "logs/net${net}_$name.log" 2>&1
  echo "GPU_DONE net$net $name exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}
for e in 4 5 6 7 8 9; do $PY lab/reduce_s1/plans.py window 4 $e 0 plans/window_s4_b$e.json; done
for e in 4 5 6; do $PY lab/reduce_s1/plans.py window 4 $e 1 plans/window_s4_b${e}_ck3.json; done
for p in window_s4_b6 window_s4_b9 window_s4_b5_ck3 window_s4_b4 window_s4_b7 window_s4_b4_ck3 window_s4_b6_ck3 window_s4_b5 window_s4_b8; do
  gpu 0 plans/$p.json
done
for p in window_s4_b9 window_s4_b6; do gpu 1 plans/$p.json; done
date -u +%FT%TZ > GPU_LANE4_DONE
