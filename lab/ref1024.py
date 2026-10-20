"""ARC reference kprop at the competition shape (1024x16) on public MLPs vs dataset ground truth.
usage: ref1024.py <idx list> <variant: k3s|k3a|k2s> [device] [dtype]
"""
import sys, time, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import numpy as np, torch
torch.set_grad_enabled(False)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlpdata import weights_from_seed

idx = [int(x) for x in sys.argv[1].split(",")]
variant = sys.argv[2]
dev = sys.argv[3] if len(sys.argv) > 3 else ("cuda" if torch.cuda.is_available() else "cpu")
dt = {"f32": torch.float32, "f64": torch.float64}[sys.argv[4] if len(sys.argv) > 4 else "f32"]
t = np.load(os.environ.get("AW_DATA", "data") + "/mini_truth.npz")
k = int(variant[1]); kind = Kind.SIMPLE if variant[2] in "sS" else Kind.AUGMENT
for i in idx:
    W = weights_from_seed(t["mlp_seed"][i])
    n, L = 1024, 16
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=dt)
    for l in range(L):
        mlp.Ws[l].weight.data = torch.from_numpy(W[l].T.copy()).to(device=dev, dtype=dt).contiguous()
    K_in = {1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}
    t0 = time.time()
    means = []
    # run layer by layer via output_all=False per layer would recompute; use up_to_layer per call is O(L^2): use output_all but only means kept
    fac = (k >= 3) and variant[2] in "sa"
    out = mlp_kprop(mlp, K_in, k_max=k, kind=kind, factor=fac, output_all=True, up_to_layer=f"act{L-1}", output_d_max=1)
    pred = np.stack([out[f"act{l}"][1].to_tensor().double().cpu().numpy() for l in range(L)])
    gt = t["all_layer_means"][i].astype(np.float64)
    per = np.mean((pred - gt) ** 2, axis=1)
    print(f"mlp {i} {variant} {dev} {dt}: final {per[-1]:.4e} t={time.time()-t0:.0f}s | " + " ".join(f"{x:.1e}" for x in per[1::3]), flush=True)
