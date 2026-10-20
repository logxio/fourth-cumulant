"""Scan cumulant order K on small-width deep MLPs with MC ground truth (ARC reference kprop).
usage: kscan.py <n> <L> <seed> <mc_samples> <variants comma list>
variants: k2s, k3s, k3a, k4s, k4a  (s=SIMPLE factor, a=AUGMENT factor)
"""
import sys, time, math, os
import numpy as np, torch
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get('AW_THREADS', '8')))
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind

n, L, seed, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
variants = sys.argv[5].split(",")
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]

# --- MC ground truth (antithetic pairs), float32 forward, float64 accumulation (cached)
import os
DATA = os.environ.get("AW_DATA", "data")
cache = f"{DATA}/kscan_{n}_{L}_{seed}_{N}.npz"
t0 = time.time()
if os.path.exists(cache):
    z = np.load(cache); gt, var_final, done = z["gt"], z["var_final"], int(z["done"]); N = -1
dev = "cuda" if torch.cuda.is_available() else "cpu"
Wt = [torch.from_numpy(w).to(dev) for w in W]
g = torch.Generator(device=dev).manual_seed(seed + 12345)
acc = torch.zeros(L, n, dtype=torch.float64, device=dev); acc2 = torch.zeros(n, dtype=torch.float64, device=dev)
B = 65536; done = 0
while N > 0 and done < N:
    x = torch.randn(B, n, generator=g, device=dev)
    for sgn in (1.0, -1.0):
        h = x * sgn
        for l in range(L):
            h = torch.relu(h @ Wt[l])
            acc[l] += h.sum(0, dtype=torch.float64)
            if l == L - 1:
                acc2 += (h.double() ** 2).sum(0)
    done += 2 * B
if N > 0:
    gt = (acc / done).cpu().numpy()
    var_final = (acc2 / done).cpu().numpy() - gt[-1] ** 2
    np.savez(cache, gt=gt, var_final=var_final, done=done)
print(f"MC N={done:.2e} t={time.time()-t0:.0f}s avg final var {var_final.mean():.4f} noise-floor MSE~{var_final.mean()/done:.2e}", flush=True)

def run(variant):
    k = int(variant[1]); kind = Kind.SIMPLE if variant[2] in "sS" else Kind.AUGMENT
    dev_k = os.environ.get("AW_KDEV", "cpu"); dt_k = torch.float32 if os.environ.get("AW_KF32") == "1" else torch.float64
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev_k, dtype=dt_k)
    for i in range(L):
        mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(device=dev_k, dtype=dt_k).contiguous()
    K_in = {1: torch.zeros(n, device=dev_k, dtype=dt_k), 2: torch.eye(n, device=dev_k, dtype=dt_k)}
    factor = k >= 3 and variant[2] in "sa"   # capital S/A = unfactored
    out = mlp_kprop(mlp, K_in, k_max=k, kind=kind, factor=factor, output_all=True, up_to_layer=f"act{L-1}", output_d_max=1)
    return np.stack([out[f"act{i}"][1].to_tensor().double().cpu().numpy() for i in range(L)])

for v in variants:
    t0 = time.time()
    try:
        pred = run(v)
    except Exception as e:
        print(f"{v}: ERROR {type(e).__name__}: {e}", flush=True); continue
    per = np.mean((pred - gt) ** 2, axis=1)
    print(f"{v}: final MSE {per[-1]:.3e} (norm {per[-1]/var_final.mean():.2e}) t={time.time()-t0:.1f}s | layers " +
          " ".join(f"{x:.1e}" for x in per[::3]), flush=True)
