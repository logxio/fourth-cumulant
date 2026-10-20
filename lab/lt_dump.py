"""Dump the LT corrector's training inputs from the released estimator (estimator.py).

usage: lt_dump.py <split> <a-b|i,j,...> <out_dir> [--mem-gb 12]
Per MLP writes <out_dir>/<split>_<idx:04d>.npz with X (16, n, LT_F) float32 corrector inputs,
Phi (16, n), pred (16, n) = the chain's own rows (corrector off), all in natural unit order,
plus flops/wall. Truth stays in data/<split>_truth.npz (joined by index at training time).
Set V26_STRASSEN=0 for dense products (same values up to f32 rounding, much faster).
"""
import argparse, importlib.util, os, sys, threading, time
os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import numpy as np
import psutil
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext
from mlpdata import weights_from_seed, estimator_seed, load_truth

B = 2 ** 41


def watchdog(limit_gb: float) -> None:
    proc = psutil.Process()
    while True:
        rss = proc.memory_info().rss / 2**30
        if rss > limit_gb:
            print(f"WATCHDOG: RSS {rss:.2f} GB > {limit_gb} GB, killing", flush=True)
            os._exit(3)
        time.sleep(0.1)


def load_est(path: str):
    spec = importlib.util.spec_from_file_location("est_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["est_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def parse_idx(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("idx"); ap.add_argument("out")
    ap.add_argument("--est", default=os.path.join(ROOT, "estimator.py"))
    ap.add_argument("--mem-gb", type=float, default=12.0)
    a = ap.parse_args()
    threading.Thread(target=watchdog, args=(a.mem_gb,), daemon=True).start()
    os.makedirs(a.out, exist_ok=True)
    t = load_truth(a.split)
    est = load_est(a.est)
    mod = sys.modules["est_mod"]
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1",
                           submission_dir=os.path.dirname(os.path.abspath(a.est))))
    for k in parse_idx(a.idx):
        out = os.path.join(a.out, f"{a.split}_{k:04d}.npz")
        if os.path.exists(out):
            continue
        W = [fnp.array(w, dtype=fnp.float32) for w in weights_from_seed(t["mlp_seed"][k])]
        mlp = MLP(width=1024, depth=16, weights=W, seed=estimator_seed(t["mlp_seed"][k]), name=str(t["mlp_name"][k]))
        mod.LT_DUMP.clear()
        t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float32)
        wall = time.time() - t0
        assert len(mod.LT_DUMP) == 16, len(mod.LT_DUMP)
        X = np.stack([np.asarray(x, dtype=np.float32) for _, x, _ in mod.LT_DUMP])
        Phi = np.stack([np.asarray(p, dtype=np.float32) for _, _, p in mod.LT_DUMP])
        truth = t["all_layer_means"][k].astype(np.float64)
        raw = float(np.mean((pred[-1] - truth[-1]) ** 2))
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, Phi=Phi, pred=pred, seed=int(t["mlp_seed"][k]), idx=k,
                 flops=float(ctx.flops_used), wall=wall)
        os.replace(tmp, out)
        rss = psutil.Process().memory_info().rss / 2**30
        print(f"{a.split} {k:4d} raw {raw:.4e} C/B {ctx.flops_used / B:.4f} wall {wall:.1f}s rss {rss:.2f}GB "
              f"Xfinite {bool(np.isfinite(X).all())}", flush=True)


if __name__ == "__main__":
    main()
