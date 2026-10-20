#!/bin/bash
# reduce_s1 one-machine batch (cloud, one 96 GB GPU, >= 256 GB host RAM). Research chains only, no submission.
# GPU lane: chain B prefix snapshots, instrumented full s0 = 4 run (read weights), then states kept by read weight
# at fixed unit budgets, the reverse-order control, and full later starts. CPU lane, alongside: chain B plus
# K=4 pieces fed back (flopscope-metered B + counted sidecar), ages and ranks past the 1,024 budget.
# Net 0 first, then net 1. Each stage writes outputs/<net>/<name>.json + .npy; DONE marks the end.
set -uo pipefail
RR=${REDUCE_ROOT:-"$(pwd)"}
cd $RR
export AW_DATA=$RR/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$RR/tools:$RR/refs/mlp_cumulant_propagation/src:$RR/lab:$RR/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs/net0 outputs/net1 snapshots plans logs
if [ ! -x ${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python ]; then
  python -m pip install -q uv
  uv venv -q -p 3.12 ${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv
  uv pip install -q --python ${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  uv pip install -q --python ${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python einops jaxtyping joblib opt-einsum scipy tqdm numpy pandas psutil flopscope==0.12.1 whestbench==0.16.1
fi
PY=${REDUCE_ROOT:-"$(pwd)"}/.reduce-venv/bin/python
$PY -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())' || exit 3
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,memory.total --format=csv | tee logs/gpu.txt
free -g | tee logs/mem.txt

gpu() {  # gpu <net> <plan> [--instrument]
  local net=$1 plan=$2; shift 2
  local name=$(basename "$plan" .json)
  [ -f "outputs/net$net/$name.json" ] && return 0
  timeout 2400 $PY -u lab/reduce_s1/select_run.py --plan "$plan" --index "$net" --snapshots snapshots \
    --output "outputs/net$net/$name.json" "$@" > "logs/net${net}_$name.log" 2>&1
  echo "GPU_DONE net$net $name exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}
cpu() {  # cpu <net> <case>
  local net=$1 case=$2 name=$(echo "$2" | tr ':' '_')
  [ -f "outputs/net$net/fed_$name.json" ] && return 0
  OPENBLAS_NUM_THREADS=7 OMP_NUM_THREADS=7 timeout 1800 $PY -u lab/reduce_s1/probe_feedback.py "$case" "$net" \
    "outputs/net$net/fed_$name.json" > "logs/net${net}_fed_$name.log" 2>&1
  echo "CPU_DONE net$net $case exit=$? $(date -u +%FT%TZ)" | tee -a logs/progress.txt
}

# chain B snapshots (CPU): net 0 at 4, 10, 12, 13; net 1 at 4 and 10
[ -f snapshots/B_mini0_pre13.npz ] || $PY -u lab/reduce_s1/b_prefix_more.py --index 0 --out snapshots --layers 4,10,12,13 > logs/prefix0.log 2>&1
cat snapshots/B_mini0.json
( [ -f snapshots/B_mini1_pre10.npz ] || $PY -u lab/reduce_s1/b_prefix_more.py --index 1 --out snapshots --layers 4,10 > logs/prefix1.log 2>&1 ) &
PREFIX1=$!

# CPU lane, net 0 then net 1 (after the net-1 prefix export)
(
  for c in combo:6:all combo:8:all combo:15:all mixshort:8:all:2 mix:8:all:8 starcross:4:all:16 every:4:all:16 every:8:all:16; do cpu 0 $c; done
  for i in $(seq 360); do [ -f snapshots/B_mini1.json ] && break; sleep 5; done
  for c in base:1:all combo:4:all combo:8:all combo:15:all every:8:all:16; do cpu 1 $c; done
  date -u +%FT%TZ > CPU_DONE
) &
CPU=$!

# GPU lane, net 0
$PY lab/reduce_s1/plans.py full 4 plans/full_s4_b14.json --last-birth 14
gpu 0 plans/full_s4_b14.json --instrument
I0=outputs/net0/full_s4_b14.json
if [ -f $I0 ]; then
  for b in 6000 20453 12000 3010 33611 1204 602; do $PY lab/reduce_s1/plans.py select $I0 weight $b plans/n0_weight_$b.json; done
  $PY lab/reduce_s1/plans.py select $I0 reverse 20453 plans/n0_reverse_20453.json
fi
for s in 10 12 13; do $PY lab/reduce_s1/plans.py full $s plans/full_s$s.json; done
for p in n0_weight_6000 n0_weight_20453 full_s10 n0_reverse_20453 n0_weight_12000 n0_weight_3010 n0_weight_33611 full_s12 full_s13 n0_weight_1204 n0_weight_602; do
  [ -f plans/$p.json ] && gpu 0 plans/$p.json
done
date -u +%FT%TZ > NET0_DONE

# GPU lane, net 1
wait $PREFIX1
gpu 1 plans/full_s4_b14.json --instrument
I1=outputs/net1/full_s4_b14.json
if [ -f $I1 ]; then
  for b in 6000 20453; do $PY lab/reduce_s1/plans.py select $I1 weight $b plans/n1_weight_$b.json; done
fi
for p in n1_weight_20453 full_s10 n1_weight_6000; do
  [ -f plans/$p.json ] && gpu 1 plans/$p.json
done
wait $CPU
date -u +%FT%TZ > DONE
