#!/bin/bash
# reduce_s1 GPU lane 5 (after lane 4): windows of the gain families (non-tree, star) for births 4..e with tree
# slices born at every layer (the 20,451-unit control kept late tree slices; the clean 4..9 window without them is 2.6x worse).
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
for i in $(seq 1440); do [ -f GPU_LANE4_DONE ] && break; sleep 10; done
gpu() {
  local net=$1 plan=$2; shift 2
  local name=$(basename "$plan" .json)
  [ -f "outputs/net$net/$name.json" ] && return 0
  timeout 2400 $PY -u lab/reduce_s1/select_run.py --plan "$plan" --index "$net" --snapshots snapshots \
    --output "outputs/net$net/$name.json" "$@" > "logs/net${net}_$name.log" 2>&1
  echo "GPU_DONE net$net $name exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}
for e in 4 6 9; do $PY lab/reduce_s1/plans.py window 4 $e 0 plans/window_s4_b${e}_treeall.json --tree-all; done
for e in 9 6 4; do gpu 0 plans/window_s4_b${e}_treeall.json; done
gpu 1 plans/window_s4_b9_treeall.json
date -u +%FT%TZ > GPU_LANE5_DONE
