"""Run one ablation arm on seeded public networks.

The estimator source is shared by both learner settings. Each invocation loads
it after applying one arm's environment switches, so module constants stay
fixed for the entire invocation.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench import MLP, SetupContext

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 2**41
SWITCHES = {
    "b": {},
    "active_off": {"V30_PRUNE": "0"},
    "col_off": {"AW_NOCOL": "1"},
    "young_off": {"AW_Y_EPS": "0"},
    "repr_off": {"V21_R_OLD": "384", "AW_R_RES": "16", "AW_R_FB": "16",
                 "V26_STRASSEN": "6", "V26_STRASSEN_HUB": "6", "V28_STRASSEN_SB": "6",
                 "V29_CPRE_LEV": "6", "V28_STRASSEN_FIRST": "6"},
    "all_off": {"V30_PRUNE": "0", "AW_Y_EPS": "0", "V21_R_OLD": "384",
                "AW_R_RES": "16", "AW_R_FB": "16", "V26_STRASSEN": "6",
                "V26_STRASSEN_HUB": "6", "V28_STRASSEN_SB": "6",
                "V29_CPRE_LEV": "6", "V28_STRASSEN_FIRST": "6"},
}
FIELDS = ("arm", "learner", "index", "seed", "width", "depth", "raw", "cb", "adj", "flops", "wall_s")


def weights_from_seed(seed, width, depth):
    weight_seed, _, estimator_seed = np.random.SeedSequence(int(seed)).spawn(3)
    rng = np.random.default_rng(weight_seed)
    scale = float(np.sqrt(2.0 / width))
    weights = [(rng.standard_normal((width, width)) * scale).astype(np.float32)
               for _ in range(depth)]
    return weights, int(estimator_seed.generate_state(1)[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=SWITCHES, required=True)
    parser.add_argument("--learner", choices=("plain", "lt2"), required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--indices", default="0-7")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ablation.csv")
    args = parser.parse_args()
    if "-" in args.indices:
        lo, hi = [int(x) for x in args.indices.split("-")]
        indices = list(range(lo, hi + 1))
    else:
        indices = [int(x) for x in args.indices.split(",")]
    seeds = json.loads((ROOT / "data" / "seeds.json").read_text())
    if not indices or any(i < 0 or i >= len(seeds) for i in indices):
        parser.error("indices must be in 0-7")
    for key in {key for switch in SWITCHES.values() for key in switch} | {"LT_OFF"}:
        os.environ.pop(key, None)
    os.environ.update(SWITCHES[args.arm])
    os.environ["LT_OFF"] = "1" if args.learner == "plain" else "0"
    source = Path(__file__).with_name("chain.py")
    spec = importlib.util.spec_from_file_location("release_ablation_chain", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    estimator = module.Estimator()
    estimator.setup(SetupContext(width=args.width, depth=args.depth,
                                 flop_budget=BUDGET, api_version="1",
                                 submission_dir=str(source.parent)))
    truth = np.load(ROOT / "data" / "mini_truth_final.npy", allow_pickle=False) if args.width == 1024 and args.depth == 16 else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    existing = list(csv.DictReader(args.out.open())) if args.out.exists() else []
    keyset = {(x["arm"], x["learner"], int(x["index"]), int(x["width"]), int(x["depth"])) for x in existing}
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if not existing:
            writer.writeheader()
        for index in indices:
            key = (args.arm, args.learner, index, args.width, args.depth)
            if key in keyset:
                continue
            weights, est_seed = weights_from_seed(seeds[index]["seed"], args.width, args.depth)
            mlp = MLP(width=args.width, depth=args.depth,
                      weights=[fnp.array(w, dtype=fnp.float32) for w in weights],
                      seed=est_seed, name=seeds[index]["name"])
            estimator._ncall = index
            context = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
            start = time.monotonic()
            with context:
                prediction = np.asarray(estimator.predict(mlp, BUDGET), dtype=np.float64)
            wall = time.monotonic() - start
            if prediction.shape[-1] != args.width or not np.isfinite(prediction).all():
                raise RuntimeError("prediction shape or values are invalid")
            raw = float(np.mean((prediction[-1] - truth[index])**2)) if truth is not None else ""
            cb = float(context.flops_used / BUDGET)
            row = dict(arm=args.arm, learner=args.learner, index=index,
                       seed=seeds[index]["seed"], width=args.width, depth=args.depth,
                       raw=raw, cb=cb, adj=raw * max(0.1, cb) if truth is not None else "",
                       flops=int(context.flops_used), wall_s=wall)
            writer.writerow(row)
            handle.flush()
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
