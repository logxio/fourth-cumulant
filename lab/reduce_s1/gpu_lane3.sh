#!/bin/bash
# reduce_s1 GPU lane 3 (after lane 2): whole states at full life. 'state'/'stateunit' rank states by the summed
# v2 read weight (alone / per counted unit); 'early' keeps tree, non-tree and star states in birth order, no C.k3.
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
for i in $(seq 1440); do [ -f GPU_LANE2_DONE ] && break; sleep 10; done
gpu() {
  local net=$1 plan=$2; shift 2
  local name=$(basename "$plan" .json)
  [ -f "outputs/net$net/$name.json" ] && return 0
  timeout 2400 $PY -u lab/reduce_s1/select_run.py --plan "$plan" --index "$net" --snapshots snapshots \
    --output "outputs/net$net/$name.json" "$@" > "logs/net${net}_$name.log" 2>&1
  echo "GPU_DONE net$net $name exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}
for net in 0 1; do
  I=outputs/net$net/full_s4_b14_v2.json
  [ -f $I ] || I=outputs/net$net/full_s4_b14.json
  [ -f $I ] || continue
  for r in early state stateunit; do
    for b in 3010 6000 12000 20453; do $PY lab/reduce_s1/plans.py select $I $r $b plans/n${net}_${r}_$b.json; done
  done
done
for p in n0_early_6000 n0_early_12000 n0_state_12000 n0_stateunit_12000 n0_early_20453 n0_state_6000 n0_stateunit_6000 n0_early_3010 n0_state_20453 n0_stateunit_20453; do
  [ -f plans/$p.json ] && gpu 0 plans/$p.json
done
for p in n1_early_12000 n1_early_6000 n1_early_20453; do [ -f plans/$p.json ] && gpu 1 plans/$p.json; done
date -u +%FT%TZ > GPU_LANE3_DONE
