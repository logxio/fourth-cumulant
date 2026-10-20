"""Pack lt_dump outputs (lab/cand_lt2 extended dump) into memmaps the LT2 trainers share.

usage: lt2_prep.py <dump_dir> <prep_dir> [--workers 128]
Writes, split s in {full, mini} (natural index order, files sorted by idx):
  <s>_X.npy (M,16,1024,F) f32 corrector inputs [own27 | W prev8 | WW prev8 | ext8]
  <s>_Phi.npy, <s>_P.npy (pred), <s>_T.npy (truth), <s>_E.npy (local eps) (M,16,1024) f32
  <s>_W16.npy (M,16,1024,1024) f16 weights rebuilt from seed, <s>_meta.json (seeds, idx)
eps_l = d_l - Phi_l * (d_{l-1} @ W_l), d = truth - pred, in float64 with float32 W (as lt_train.py).
"""
import argparse, glob, json, os, sys, time
from multiprocessing import Pool
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from mlpdata import weights_from_seed, load_truth

L, N = 16, 1024
G = {}


def init(prep, split, M, F):
    G["X"] = np.lib.format.open_memmap(f"{prep}/{split}_X.npy", mode="r+")
    G["W"] = np.lib.format.open_memmap(f"{prep}/{split}_W16.npy", mode="r+")
    for k in ("Phi", "P", "T", "E"):
        G[k] = np.lib.format.open_memmap(f"{prep}/{split}_{k}.npy", mode="r+")


def one(args):
    k, path, truth = args
    d = np.load(path)
    X, Phi, pred = d["X"], d["Phi"], d["pred"]
    W = weights_from_seed(int(d["seed"]))
    dl = truth.astype(np.float64) - pred.astype(np.float64)
    eps = dl.copy()
    for l in range(1, L):
        eps[l] = dl[l] - Phi[l] * (dl[l - 1] @ W[l])
    G["X"][k] = X
    G["Phi"][k] = Phi
    G["P"][k] = pred
    G["T"][k] = truth
    G["E"][k] = eps.astype(np.float32)
    G["W"][k] = np.stack(W).astype(np.float16)
    return int(d["seed"]), int(d["idx"]), bool(np.isfinite(X).all())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("prep"); ap.add_argument("--workers", type=int, default=128)
    a = ap.parse_args()
    os.makedirs(a.prep, exist_ok=True)
    t0 = time.time()
    for split in ("mini", "full"):
        files = sorted(glob.glob(os.path.join(a.dump, f"{split}_*.npz")))
        files = [f for f in files if not f.endswith(".tmp.npz")]
        M = len(files)
        F = np.load(files[0])["X"].shape[-1]
        t = load_truth(split)
        idx = [int(os.path.basename(f)[len(split) + 1:-4]) for f in files]
        T = t["all_layer_means"][idx].astype(np.float32)
        np.lib.format.open_memmap(f"{a.prep}/{split}_X.npy", mode="w+", dtype=np.float32, shape=(M, L, N, F))
        np.lib.format.open_memmap(f"{a.prep}/{split}_W16.npy", mode="w+", dtype=np.float16, shape=(M, L, N, N))
        for k in ("Phi", "P", "T", "E"):
            np.lib.format.open_memmap(f"{a.prep}/{split}_{k}.npy", mode="w+", dtype=np.float32, shape=(M, L, N))
        with Pool(a.workers, initializer=init, initargs=(a.prep, split, M, F)) as pool:
            rows = pool.map(one, [(k, f, T[k]) for k, f in enumerate(files)], chunksize=1)
        seeds = [r[0] for r in rows]; idd = [r[1] for r in rows]
        assert idd == idx and all(int(t["mlp_seed"][i]) == s for i, s in zip(idx, seeds))
        assert all(r[2] for r in rows), "non-finite X"
        json.dump({"seeds": seeds, "idx": idx, "F": F}, open(f"{a.prep}/{split}_meta.json", "w"))
        print(f"{split}: {M} MLPs F={F} t={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
