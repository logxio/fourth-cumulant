"""Rebuild the final-layer vectors behind the star-birth weights (report Section 3) from the released predictions.

usage: python3 report/data/c45/build.py [--out outputs/c45]
Reads the saved K=4 predictions of four arms (full, no_path, no_star, no_k3c) and their truths:
  widths 64-192: report/data/c45/preds/p<n>_<seed>_<arm>_f64.npz (f32 where no f64 run exists), written by
                 report/data/c45/arms.py; truths report/data/frozen_s16/truth/kscan_<n>_<L>_<seed>_*.npz (key gt);
  width 1024:    official mini 5, 6, 7: full, no_path, no_k3c from data/frozen_1024/birth<i>_<arm>.npy (the frozen
                 Table 3 runs), no_star from report/data/c45/preds/birth<i>_no_star.npy; truth data/mini_truth.npz
                 all_layer_means.
Writes <out>/arms_last.npz (key "<n>_<seed>_<arm>" and "<n>_<seed>_gt", final layer only) and <out>/sources.json (SHA-256
of every input file), then checks both against report/data/c45/arms_last.npz and sources.json. Layer 0 of every arm must
match the full arm: a mismatch means a mixed-up file.
"""
import argparse
import glob
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARMS = ("full", "no_path", "no_star", "no_k3c")


def collect():
    out, src = {}, {}

    def sha(p):
        src[str(Path(p).relative_to(ROOT))] = hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def put(n, s, arms, gt):
        for a in ARMS:
            assert np.allclose(arms[a][0], arms["full"][0], rtol=1e-4, atol=1e-12), (n, s, a)
            out[f"{n}_{s}_{a}"] = arms[a][-1].astype(np.float64)
        out[f"{n}_{s}_gt"] = gt[-1].astype(np.float64)

    nets = {}
    for f in sorted(glob.glob(str(HERE / "preds" / "p*_f??.npz")), key=lambda x: x.endswith("_f32.npz")):
        n, s, a = re.match(r"p(\d+)_(\d+)_(.+)_f(?:64|32)\.npz", os.path.basename(f)).groups()
        if a in ARMS and a not in nets.setdefault((int(n), int(s)), {}):
            nets[(int(n), int(s))][a] = f
    for (n, s), files in sorted(nets.items()):
        if set(files) != set(ARMS):
            continue
        arms = {a: np.load(f)["pred"] for a, f in files.items()}
        L = arms["full"].shape[0]
        (tf,) = glob.glob(str(ROOT / f"report/data/frozen_s16/truth/kscan_{n}_{L}_{s}_*.npz"))
        for f in (*files.values(), tf):
            sha(f)
        put(n, s, arms, np.load(tf)["gt"])

    truth_file = ROOT / "data" / "mini_truth.npz"
    truth = np.load(truth_file)["all_layer_means"]
    sha(truth_file)
    for i in (5, 6, 7):
        files = {a: ROOT / "data" / "frozen_1024" / f"birth{i}_{a}.npy" for a in ("full", "no_path", "no_k3c")}
        files["no_star"] = HERE / "preds" / f"birth{i}_no_star.npy"
        arms = {a: np.load(f) for a, f in files.items()}
        assert all(p.shape == (16, 1024) and np.isfinite(p).all() for p in arms.values()), i
        for f in files.values():
            sha(f)
        put(1024, i, arms, truth[i])
    return out, src


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(ROOT / "outputs" / "c45"))
    out_dir = Path(parser.parse_args().out)
    out, src = collect()
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "arms_last.npz", **out)
    (out_dir / "sources.json").write_text(json.dumps(src, indent=1, sort_keys=True) + "\n")
    with np.load(HERE / "arms_last.npz") as saved:
        assert sorted(saved.files) == sorted(out) and all(np.array_equal(saved[k], out[k]) for k in out)
    assert json.loads((HERE / "sources.json").read_text()) == src
    print(f"{len(out)} vectors from {len(src)} files -> {out_dir / 'arms_last.npz'}; "
          "equal to report/data/c45/arms_last.npz, inputs match sources.json")


if __name__ == "__main__":
    main()
