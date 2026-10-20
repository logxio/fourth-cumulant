"""504aldo's float64 lean chain (lean_k3_aug) on a chosen device, on public MLPs vs dataset GT.
usage: run_lean_gpu.py <idx list> <device> [--aug] [--f32]
The module source is exec'd with the weight upload moved to the device; every other tensor
is created on torch's default device, which is set here.
"""
import sys, os, time, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "refs", "504aldo", "lean"))
LEAN_DIR = os.environ.get("AW_LEAN", os.path.join(os.path.dirname(__file__), "..", "refs", "504aldo", "lean"))
# the lean chain builds its term tables from the ARC reference at import; it matches the reference as of
# 2026-07-09 (6e80f7f) to 1e-16, while f4325dd (07-31, new truncation) breaks it (aug mode 6.9e-8 instead of 2.1e-8)
REF0709 = os.environ.get("AW_REF0709", os.path.join(os.path.dirname(__file__), "..", "refs", "ref_0709", "src"))
sys.path.insert(0, REF0709)
import numpy as np, torch

idx = [int(x) for x in sys.argv[1].split(",")]
dev = sys.argv[2]
aug = "--aug" in sys.argv
dt = torch.float32 if "--f32" in sys.argv else torch.float64
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)   # F51: reference coefficient builders need f64 defaults
torch.set_default_device(dev)

src = open(os.path.join(LEAN_DIR, "lean_k3_aug.py")).read()
old = "Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(dtype) for w in weights]"
assert old in src
src = src.replace(old, "Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(device=torch.get_default_device(), dtype=dtype) for w in weights]")
src = src.replace("torch.set_num_threads(22)", "pass")
import mlp_kprop
_kp = getattr(mlp_kprop, "__file__", None) or list(mlp_kprop.__path__)[0]   # namespace package: no __file__
assert os.path.realpath(_kp).startswith(os.path.realpath(REF0709)), _kp
mod = types.ModuleType("lean_k3_aug_dev")
mod.__file__ = os.path.join(LEAN_DIR, "lean_k3_aug.py")
exec(compile(src, mod.__file__, "exec"), mod.__dict__)

from mlpdata import weights_from_seed, load_truth
t = load_truth("mini")
statics = None
for i in idx:
    W = np.stack(weights_from_seed(t["mlp_seed"][i]))
    if statics is None:
        if dev == "mps":
            torch.set_default_device("cpu")
        statics = mod.build_static_coefs(W.shape[1])
        if dev == "mps":
            torch.set_default_dtype(torch.float32)
            torch.set_default_device(dev)
    gt = t["all_layer_means"][i].astype(np.float64)
    t0 = time.time()
    out = mod.lean_k3_predict(W, dtype=dt, statics=statics, k4_aug=aug)
    pred = (out[0] if isinstance(out, tuple) else out).cpu().numpy().astype(np.float64)
    per = np.mean((pred - gt) ** 2, axis=1)
    print(f"lean aug={aug} mlp {i} {dev} {dt}: final {per[-1]:.4e} all {per.mean():.3e} t {time.time()-t0:.1f}s | " +
          " ".join(f"{x:.1e}" for x in per[1::3]), flush=True)
