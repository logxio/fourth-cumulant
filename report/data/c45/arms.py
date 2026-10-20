"""K=4 chain (lab/k4full.py, unchanged reference step) on a seeded kscan MLP with one birth-type arm; saves per-layer means.
usage: arms.py <n> <L> <seed> <arm> <f64|f32> <out.npz>
arms: full, no_path, no_k3c, no_slicex, no_star          (AW_SPEC strings as report/data/frozen_s16/run.sh)
      pathA<alpha>         path births scaled by alpha at every layer (e.g. pathA0.5)
      pathLT<b> / pathGE<b> path births kept only for birth layers < b / >= b
      k3cA<alpha>          C.kappa3 (three2) births scaled by alpha
Also records, per layer, the rms off-diagonal correlation of the pre-activations seen by the chain.
Run from the repository root with the pinned ARC reference on PYTHONPATH (REPRODUCE.md); report/data/c45/preds/ holds
the saved outputs of the four arms used in Section 3. Released copy: only the lab/ path below is made relative.
"""
import sys, os, math, time, re
n, L, seed, arm, prec, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
SPECS = {"full": "full",
         "no_path": "star+three2,slice4+slice31+slice22+slice211",
         "no_k3c": "path+star,slice4+slice31+slice22+slice211",
         "no_slicex": "path+star+three2,slice4+slice31,!slice22+slice211",
         "no_star": "path+three2,slice4+slice31+slice22+slice211"}
scale = lambda types, layer: 1.0
if arm in SPECS:
    spec = SPECS[arm]
elif arm.startswith("path"):
    spec = "path,star+three2,slice4+slice31+slice22+slice211"
    m = re.fullmatch(r"path(A|LT|GE)([0-9.]+)", arm); kind, v = m.group(1), float(m.group(2))
    f = {"A": lambda l: v, "LT": lambda l: 1.0 if l < v else 0.0, "GE": lambda l: 1.0 if l >= v else 0.0}[kind]
    scale = lambda types, layer: f(layer) if types == ("path",) else 1.0
elif arm.startswith("k3cA"):
    spec = "three2,path+star,slice4+slice31+slice22+slice211"
    v = float(arm[4:])
    scale = lambda types, layer: v if types == ("three2",) else 1.0
else:
    raise SystemExit(f"unknown arm {arm}")
os.environ["AW_SPEC"] = spec
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "lab"))
import numpy as np, torch
torch.set_num_threads(10)
import k4full as K4
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import linear_kprop, coerce_input, Kind
from mlp_kprop.wick import WICK_COEF_D

LAYER = [0]


def add_two_leg_scaled(pK, X, groups):
    nn = pK.n
    Q = torch.eye(nn, device=X.device, dtype=X.dtype)
    new = []
    for types, crossed, buf in groups:
        s = scale(types, LAYER[0])
        if s == 0.0:
            continue
        B = (buf + buf.transpose(0, 1)) / 2
        if s != 1.0:
            B = B * s
        new.append(K4.Group(types, crossed, X.clone(), None, K4._store(B), Q.clone()))
        del B
    groups.clear()
    pK.add_groups_(new)


K4.ns["add_two_leg"] = add_two_leg_scaled
dt = torch.float32 if prec == "f32" else torch.float64
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(dtype=dt)
for i in range(L):
    mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(dtype=dt).contiguous()
Kt = coerce_input({1: torch.zeros(n, dtype=dt), 2: torch.eye(n, dtype=dt)}, k_max=4, kind=Kind.SIMPLE)
wick = WICK_COEF_D["relu"]
means, rho2 = [], []
t0 = time.time()
for l in range(L):
    LAYER[0] = l
    K4.LAST_LAYER = l == L - 1
    Kt = linear_kprop(Kt, mlp.Ws[l].weight, k_max=4, set_metric=mlp.init_scale[l])
    C = Kt[2].core.double()
    sd = C.diagonal().sqrt()
    R = C / sd[:, None] / sd[None, :]
    rho2.append(float(((R ** 2).sum() - n) / (n * (n - 1))))
    Kt = K4.nonlin(Kt, nonlin_wick_coef=wick)
    means.append(Kt[1].to_tensor().double().numpy())
pred = np.stack(means)
np.savez(out, pred=pred, rho2=np.array(rho2), arm=arm, spec=spec, prec=prec, t=time.time() - t0)
print(f"arms n={n} L={L} seed={seed} arm={arm} spec={spec} {prec} t={time.time() - t0:.0f}s {K4._mem()}", flush=True)
