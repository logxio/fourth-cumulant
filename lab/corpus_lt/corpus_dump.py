"""Dump networks of Keenan Pepper's public corpus through the released chain's corrector hooks.

Run from the repository root, with the shards fetched by lab/corpus_lt/fetch.py in a separate data directory.  The
only objects read from the downloaded files are numeric NumPy arrays.
"""
import argparse
import os
import sys
import time

os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")

import importlib.util

import numpy as np
import torch
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_est(path):
    spec = importlib.util.spec_from_file_location("est_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["est_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def parse_idx(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out

B = 2**41
SCALE = (2.0 / 1024.0) ** 0.5


def weights(idx):
    gen = torch.Generator(device="cpu").manual_seed(770000 + idx)
    w = torch.randn(16, 1024, 1024, generator=gen, dtype=torch.float32)
    w *= SCALE
    return [fnp.array(x.numpy(), dtype=fnp.float32) for x in w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indices", help="a-b or comma-separated indices, within 0..2047")
    ap.add_argument("shards")
    ap.add_argument("out")
    ap.add_argument("--est", default=os.path.join(ROOT, "estimator.py"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    est = load_est(a.est)
    mod = sys.modules["est_mod"]
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1",
                           submission_dir=os.path.dirname(os.path.abspath(a.est))))
    cache_shard = None
    moments = None
    for idx in parse_idx(a.indices):
        assert 0 <= idx < 2048
        out = os.path.join(a.out, f"corpus_{idx:04d}.npz")
        if os.path.exists(out):
            continue
        shard = idx % 16
        row = idx // 16
        if shard != cache_shard:
            path = os.path.join(a.shards, f"shard{shard:02d}.npz")
            with np.load(path, allow_pickle=False) as z:
                moments = np.asarray(z["moments"], dtype=np.float32)
                shard_indices = np.asarray(z["idx"], dtype=np.int64)
            assert moments.shape == (128, 4, 16, 1024), moments.shape
            cache_shard = shard
        assert shard_indices[row] == idx, (idx, shard, row, shard_indices[row])
        truth = moments[row, 0]
        w = weights(idx)
        mlp = MLP(width=1024, depth=16, weights=w, seed=770000 + idx, name=f"corpus_{idx}")
        mod.LT_DUMP.clear()
        start = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float32)
        assert len(mod.LT_DUMP) == 16, len(mod.LT_DUMP)
        X = np.stack([np.asarray(x, dtype=np.float32) for _, x, _ in mod.LT_DUMP])
        Phi = np.stack([np.asarray(p, dtype=np.float32) for _, _, p in mod.LT_DUMP])
        assert X.shape[:2] == (16, 1024) and np.isfinite(X).all()
        raw = np.mean((pred[-1].astype(np.float64) - truth[-1]) ** 2)
        assert raw < 1e-6, f"corpus weight/truth mismatch at {idx}: {raw}"
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, Phi=Phi, pred=pred, idx=idx, seed=770000 + idx,
                 flops=float(ctx.flops_used), wall=time.time() - start)
        os.replace(tmp, out)
        print(f"corpus {idx} raw {raw:.5e} C/B {ctx.flops_used/B:.4f} wall {time.time()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
