"""Merge official full dumps and public corpus dumps into LT2 memmaps.

The official mini split is packed solely for evaluation. It never enters the
training split. Downloaded NPZs are loaded with pickle disabled.
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mlpdata import load_truth, weights_from_seed

L, N = 16, 1024
G = {}


def init(prep, split, shards):
    G["shards"] = shards
    G["shard_cache"] = {}
    for key in ("X", "W16", "Phi", "P", "T", "E"):
        G[key] = np.lib.format.open_memmap(f"{prep}/{split}_{key}.npy", mode="r+")


def one(item):
    row, source, idx, path, official_truth = item
    with np.load(path, allow_pickle=False) as z:
        X, phi, pred = z["X"], z["Phi"], z["pred"]
        seed = int(z["seed"])
    if source == "corpus":
        assert seed == 770000 + idx
        shard = idx % 16
        shard_row = idx // 16
        if shard not in G["shard_cache"]:
            shard_path = os.path.join(G["shards"], f"shard{shard:02d}.npz")
            with np.load(shard_path, allow_pickle=False) as z:
                G["shard_cache"][shard] = (
                    np.asarray(z["idx"], dtype=np.int64),
                    np.asarray(z["moments"], dtype=np.float32),
                )
        shard_indices, moments = G["shard_cache"][shard]
        assert shard_indices[shard_row] == idx, (idx, shard, shard_row)
        truth = moments[shard_row, 0]
        gen = torch.Generator(device="cpu").manual_seed(seed)
        W = (torch.randn(L, N, N, generator=gen, dtype=torch.float32) * (2.0 / N) ** 0.5).numpy()
    else:
        truth = official_truth
        W = np.stack(weights_from_seed(seed))
    delta = truth.astype(np.float64) - pred.astype(np.float64)
    eps = delta.copy()
    for l in range(1, L):
        eps[l] = delta[l] - phi[l] * (delta[l - 1] @ W[l])
    G["X"][row] = X
    G["W16"][row] = W.astype(np.float16)
    G["Phi"][row] = phi
    G["P"][row] = pred
    G["T"][row] = truth
    G["E"][row] = eps.astype(np.float32)
    return source, idx, seed, bool(np.isfinite(X).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("shards")
    ap.add_argument("prep")
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.prep, exist_ok=True)
    for split in ("full", "mini"):
        jobs = []
        truth = load_truth(split)
        for path in sorted(glob.glob(f"{a.dump}/{split}_*.npz")):
            if path.endswith(".tmp.npz"):
                continue
            idx = int(os.path.basename(path)[len(split) + 1:-4])
            jobs.append((len(jobs), split, idx, path, truth["all_layer_means"][idx].astype(np.float32)))
        if split == "full":
            for path in sorted(glob.glob(f"{a.dump}/corpus_*.npz")):
                if path.endswith(".tmp.npz"):
                    continue
                idx = int(os.path.basename(path)[7:-4])
                jobs.append((len(jobs), "corpus", idx, path, None))
        assert jobs, split
        with np.load(jobs[0][3], allow_pickle=False) as z:
            F = z["X"].shape[-1]
        shapes = {"X": (len(jobs), L, N, F), "W16": (len(jobs), L, N, N)}
        shapes.update({key: (len(jobs), L, N) for key in ("Phi", "P", "T", "E")})
        for key, shape in shapes.items():
            np.lib.format.open_memmap(f"{a.prep}/{split}_{key}.npy", mode="w+",
                                      dtype=np.float16 if key == "W16" else np.float32, shape=shape)
        with Pool(a.workers, initializer=init, initargs=(a.prep, split, a.shards)) as pool:
            rows = pool.map(one, jobs, chunksize=1)
        assert all(r[3] for r in rows)
        with open(f"{a.prep}/{split}_meta.json", "w") as f:
            json.dump({"sources": [r[:3] for r in rows], "F": F}, f)
        print(f"{split}: {len(rows)} networks, F={F}, corpus={sum(r[0]=='corpus' for r in rows)}", flush=True)


if __name__ == "__main__":
    main()
