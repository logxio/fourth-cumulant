#!/bin/bash
# reduce_s1 GPU lane 2: read weights that also carry the (2,1,1) -> kappa3 path (select_run.py v2), states kept by them.
# Starts after lane 1 has completed; the two lanes share one GPU.
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
for i in $(seq 1080); do [ -f GPU_LANE_DONE ] && break; sleep 10; done
[ -f GPU_LANE_DONE ] || { echo 'GPU lane 1 did not complete' >&2; exit 75; }
gpu() {
  local net=$1 plan=$2; shift 2
  local name=$(basename "$plan" .json)
  [ -f "outputs/net$net/$name.json" ] && return 0
  timeout 2400 $PY -u lab/reduce_s1/select_run.py --plan "$plan" --index "$net" --snapshots snapshots \
    --output "outputs/net$net/$name.json" "$@" > "logs/net${net}_$name.log" 2>&1
  echo "GPU_DONE net$net $name exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}
sed 's/"full_s4_b14"/"full_s4_b14_v2"/' plans/full_s4_b14.json > plans/full_s4_b14_v2.json
gpu 0 plans/full_s4_b14_v2.json --instrument
I0=outputs/net0/full_s4_b14_v2.json
if [ -f $I0 ]; then
  for b in 6000 20453 12000 33611 3010; do $PY lab/reduce_s1/plans.py select $I0 weight $b plans/n0_w2_weight_$b.json; done
  $PY lab/reduce_s1/plans.py select $I0 reverse 20453 plans/n0_w2_reverse_20453.json
fi
for p in n0_w2_weight_6000 n0_w2_weight_20453 n0_w2_reverse_20453 n0_w2_weight_12000 n0_w2_weight_33611 n0_w2_weight_3010; do
  [ -f plans/$p.json ] && gpu 0 plans/$p.json
done
gpu 1 plans/full_s4_b14_v2.json --instrument
I1=outputs/net1/full_s4_b14_v2.json
if [ -f $I1 ]; then
  for b in 6000 20453; do $PY lab/reduce_s1/plans.py select $I1 weight $b plans/n1_w2_weight_$b.json; done
fi
for p in n1_w2_weight_20453 n1_w2_weight_6000; do [ -f plans/$p.json ] && gpu 1 plans/$p.json; done
date -u +%FT%TZ > GPU_LANE2_DONE
