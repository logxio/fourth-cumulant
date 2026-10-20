#!/bin/bash
# Build the code package for a cloud box (everything end-to-end study needs, ~7 MB incl. mini truth).
# usage: bash lab/e2e_s1/pack.sh <out.tgz>
# then on the box: mkdir -p ${WORKDIR:-$PWD} && tar xzf <out.tgz> -C ${WORKDIR:-$PWD} && bash ${WORKDIR:-$PWD}/lab/e2e_s1/remote_setup.sh
#   network i, r4 config (the run in runs/cloud/e2e_s1/m1_net<i>):
#     bash ${WORKDIR:-$PWD}/lab/e2e_s1/run_net.sh <i> 8 m1_net<i> --slice-mode dense --feedback-rank 128 --m0f 1 --core-rank 128 --m0 2 --threads 16
set -euo pipefail
cd "$(dirname "$0")/../.."
tar czf "$1" --exclude='__pycache__' \
  lab/k4full.py lab/e2e_s1/*.py lab/e2e_s1/*.sh \
  lab/cand_k4x/base_b.py lab/cand_k4x/b_suffix.py lab/cand_k4x/delayed_reference.py lab/cand_k4x/slice_age.py \
  lab/cand_k4x/tree_birth.py lab/cand_k4x/r4/prefix_meter.py tools/mlpdata.py \
  refs/mlp_cumulant_propagation/src/mlp_kprop data/mini_truth.npz
ls -la "$1"
