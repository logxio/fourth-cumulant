#!/bin/bash
# Frozen local checks of PREREG.md. Sequential, one process at a time, each under timeout 295.
set -u
cd "$(dirname "$0")/../../.."
OUT=report/data/frozen_s16
export AW_DATA=$OUT/truth
mkdir -p $AW_DATA
cp -n data/kscan_128_16_3_10000000.npz data/kscan_128_16_4_10000000.npz data/kscan_128_16_5_10000000.npz $AW_DATA/
PY=.venv/bin/python
run() {  # name, command...
  local name=$1; shift
  { echo "### $name $(date -u +%FT%TZ)"; /usr/bin/time -l timeout 295 "$@" 2>&1 | grep -v -i warn; } >> $OUT/$name.log 2>&1
}
# E1
for shape in "64 8" "96 12"; do
  for seed in 301 302 303; do
    set -- $shape
    run e1_${1}_${2}_${seed} $PY lab/kscan.py $1 $2 $seed 16777216 k3a,k4s
    run e1_${1}_${2}_${seed}_k4a $PY lab/kscan.py $1 $2 $seed 16777216 k4a
  done
done
# E2: truth for the new 192 networks, then four arms per network
for seed in 601 602 603; do
  run e2_192_16_${seed}_truth $PY lab/kscan.py 192 16 $seed 20000000 k3a
done
for cfg in "128 16 3 10000000" "128 16 4 10000000" "128 16 5 10000000" "192 16 601 20000000" "192 16 602 20000000" "192 16 603 20000000"; do
  set -- $cfg
  for arm in full no_k3c no_path no_slicex; do
    case $arm in
      full) spec=full ;;
      no_k3c) spec="path+star,slice4+slice31+slice22+slice211" ;;
      no_path) spec="star+three2,slice4+slice31+slice22+slice211" ;;
      no_slicex) spec="path+star+three2,slice4+slice31,!slice22+slice211" ;;
    esac
    export AW_SPEC="$spec"; run e2_${1}_${2}_${3}_${arm} $PY lab/k4full.py kscan $1 $2 $3 $4 cpu f64
  done
done
echo "done $(date -u +%FT%TZ)" > $OUT/DONE
