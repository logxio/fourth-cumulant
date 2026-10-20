#!/bin/bash
set -euo pipefail
cd "${WORKDIR:-$PWD}"
export AW_DATA=$PWD/data AW_STORE=host AW_CHUNK=64 PYTHONUNBUFFERED=1
export AW_SPEC='star,three2,slice4+slice31,!slice22+slice211'
export PYTHONPATH=$PWD:$PWD/tools:$PWD/refs/mlp_cumulant_propagation/src:$PWD/lab:$PWD/lab/cand_k4x
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs/net0 outputs/net1 logs snapshots plans
# CUDA and torch are supplied by the verified image; all additional wheels <100 MB.
python -m pip install --disable-pip-version-check -q einops jaxtyping joblib opt-einsum scipy tqdm numpy pandas psutil flopscope==0.12.1 whestbench==0.16.1
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.cuda.get_device_name())'
python lab/reduce_s2/smoke.py smoke
for b in 6 9 13; do
  for c in 0 1; do python lab/reduce_s1/plans.py window 4 "$b" "$c" "plans/w${b}c${c}.json" --tree-all; done
done
# Real recurrence, then package and receiver checks, before the wide sweep.
for n in 20 21 22 23 24 25 26 27 28 29; do
  python lab/reduce_s2/run.py --plan plans/w6c1.json --synthetic "$n" --pair-rank 4 --ck3-rank 4 --output "smoke/chain$n.json" > "logs/smoke$n.log" 2>&1
done
python - <<'PY'
import tarfile,tempfile,pathlib,numpy as np
with tempfile.TemporaryDirectory() as td:
 p=pathlib.Path(td)
 with tarfile.open(p/'out.tgz','w:gz') as t:
  for f in pathlib.Path('smoke').glob('chain*'):t.add(f,arcname=f.name)
 with tarfile.open(p/'out.tgz') as t:
  assert all(m.name.startswith('chain') and m.name.endswith(('.json','.npy')) and '/' not in m.name for m in t.getmembers())
  t.extractall(p/'received',filter='data')
 for f in (p/'received').glob('*.npy'):
  a=np.load(f);assert a.shape[0]==16 and np.isfinite(a).all();assert np.array_equal(a,np.load(pathlib.Path('smoke')/f.name))
print('TEN_CHAIN_PACK_RECEIVE_RESTORE_OK',flush=True)
PY
if [ ! -s snapshots/B_mini0_pre4.npz ]; then
  python lab/reduce_s1/b_prefix_more.py --index 0 --layers 4 --out snapshots > logs/prefix0.log 2>&1
fi
point() {
 local net=$1 plan=$2 nr=$3 cr=$4 name=$5
 if [ -s "outputs/net$net/$name.json" ]; then return; fi
 timeout 900 python lab/reduce_s2/run.py --plan "plans/$plan.json" --index "$net" --snapshots snapshots --pair-rank "$nr" --ck3-rank "$cr" --output "outputs/net$net/$name.json" > "logs/$name.log" 2>&1
 echo "POINT_DONE $name $(date -u +%FT%TZ)"
}
point 0 w9c0 0 0 baseline_w9
point 0 w9c0 1 0 w9_n1
point 0 w9c0 4 0 w9_n4
point 0 w9c0 16 0 w9_n16
point 0 w6c1 4 4 w6_n4_c4
point 0 w6c1 8 8 w6_n8_c8
point 0 w13c1 4 4 full_n4_c4
point 0 w13c1 8 8 full_n8_c8
python lab/reduce_s2/restore.py .
date -u +%FT%TZ > DONE
