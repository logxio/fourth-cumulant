"""Per-layer ARC reference K=3 augmented errors on official 1024x16 MLPs.

The propagation setup is copied from lab/ref1024.py.  Output keeps full
floating-point precision so the final-layer check can use its raw value.
Usage: AW_DATA=data python lab/depth_s1/ref_depth.py 0,1,2,3,4 cuda f32
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
torch.set_grad_enabled(False)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlpdata import weights_from_seed

idx = [int(x) for x in sys.argv[1].split(",")]
dev = sys.argv[2] if len(sys.argv) > 2 else ("cuda" if torch.cuda.is_available() else "cpu")
dt = {"f32": torch.float32, "f64": torch.float64}[sys.argv[3] if len(sys.argv) > 3 else "f32"]
t = np.load(os.environ.get("AW_DATA", "data") + "/mini_truth.npz")
print("net,layer,mse_ref_k3a", flush=True)
for i in idx:
    W = weights_from_seed(t["mlp_seed"][i])
    n, L = 1024, 16
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=dt)
    for l in range(L):
        mlp.Ws[l].weight.data = torch.from_numpy(W[l].T.copy()).to(device=dev, dtype=dt).contiguous()
    K_in = {1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}
    t0 = time.time()
    out = mlp_kprop(mlp, K_in, k_max=3, kind=Kind.AUGMENT, factor=True,
                    output_all=True, up_to_layer=f"act{L-1}", output_d_max=1)
    pred = np.stack([out[f"act{l}"][1].to_tensor().double().cpu().numpy() for l in range(L)])
    gt = t["all_layer_means"][i].astype(np.float64)
    per = np.mean((pred - gt) ** 2, axis=1)
    for layer, mse in enumerate(per, 1):
        print(f"{i},{layer},{mse:.17g}", flush=True)
    print(f"net={i} final={per[-1]:.4e} elapsed={time.time()-t0:.2f}s", file=sys.stderr, flush=True)
