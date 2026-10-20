"""One ablation arm on the eight public mini networks; truth is opened after prediction."""
import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import time
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import flopscope as flops
import flopscope.numpy as f
import numpy as np
from whestbench import MLP, SetupContext

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "tools"))
from mlpdata import estimator_seed, weights_from_seed

BUDGET = 2 ** 41
ARMS = {
    "b": {},
    "active_off": {"V30_PRUNE": "0"},
    "young_off": {"AW_Y_EPS": "0"},
    "old384": {"V21_R_OLD": "384"},
    "res16": {"AW_R_RES": "16"},
    "fb16": {"AW_R_FB": "16"},
    "strassen6": {"V26_STRASSEN": "6", "V26_STRASSEN_HUB": "6", "V28_STRASSEN_SB": "6", "V29_CPRE_LEV": "6", "V28_STRASSEN_FIRST": "6"},
    "repr_off": {"V21_R_OLD": "384", "AW_R_RES": "16", "AW_R_FB": "16", "V26_STRASSEN": "6", "V26_STRASSEN_HUB": "6", "V28_STRASSEN_SB": "6", "V29_CPRE_LEV": "6", "V28_STRASSEN_FIRST": "6"},
    "all_off": {"V30_PRUNE": "0", "AW_Y_EPS": "0", "V21_R_OLD": "384", "AW_R_RES": "16", "AW_R_FB": "16", "V26_STRASSEN": "6", "V26_STRASSEN_HUB": "6", "V28_STRASSEN_SB": "6", "V29_CPRE_LEV": "6", "V28_STRASSEN_FIRST": "6"},
}


def load_est(path):
    spec = importlib.util.spec_from_file_location("ablate_estimator", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS)
    ap.add_argument("learner", choices=("plain", "lt2"))
    ap.add_argument("out", type=Path)
    ap.add_argument("--indices", default="0,1,2,3,4,5,6,7")
    a = ap.parse_args()
    for k in ("V30_PRUNE", "AW_Y_EPS", "V21_R_OLD", "AW_R_RES", "AW_R_FB", "V26_STRASSEN", "V26_STRASSEN_HUB", "V28_STRASSEN_SB", "V29_CPRE_LEV", "V28_STRASSEN_FIRST"):
        os.environ.pop(k, None)
    os.environ.update(ARMS[a.arm])
    os.environ["LT_OFF"] = "0" if a.learner == "lt2" else "1"
    source = Path(__file__).with_name("chain_b.py" if a.learner == "plain" else "chain_b_lt2.py")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    seeds = json.loads((ROOT / "lab/ablate_s1/seeds.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    est = load_est(source)
    t0 = time.monotonic()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=BUDGET, api_version="1", submission_dir=str(source.parent)))
    setup_s = time.monotonic() - t0
    for idx in [int(x) for x in a.indices.split(",")]:
        path = a.out / f"mini{idx}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["source_sha256"] != source_hash or old["params"] != ARMS[a.arm] or old["seed"] != seeds[idx]["seed"]:
                raise RuntimeError(f"stale checkpoint {path}")
            continue
        seed = seeds[idx]["seed"]
        mlp = MLP(width=1024, depth=16, weights=[f.array(w, dtype=f.float32) for w in weights_from_seed(seed)], seed=estimator_seed(seed), name=seeds[idx]["name"])
        ctx = flops.BudgetContext(flop_budget=BUDGET, quiet=True)
        # The chain changes its first-call Strassen level. A resumed mini index
        # must see the same call ordinal as an uninterrupted eight-network run.
        est._ncall = idx
        signal.alarm(600)
        start = time.monotonic()
        with ctx:
            pred = np.asarray(est.predict(mlp, BUDGET), dtype=np.float64)
        wall = time.monotonic() - start
        signal.alarm(0)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"nonfinite prediction {a.arm}/{a.learner}/{idx}")
        # The true means enter only after the model has returned its prediction.
        with np.load(ROOT / "data/mini_truth.npz") as truth_file:
            truth = np.asarray(truth_file["all_layer_means"][idx, -1], dtype=np.float64)
        raw = float(np.mean((pred[-1] - truth) ** 2))
        cb = float(ctx.flops_used / BUDGET)
        row = {"arm": a.arm, "learner": a.learner, "index": idx, "seed": seed,
               "raw": raw, "cb": cb, "adj": raw * max(0.1, cb),
               "flops": int(ctx.flops_used), "setup_s": setup_s, "wall_s": wall,
               "residual_s": ctx.residual_wall_time_s,
               "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576,
               "source_sha256": source_hash, "params": ARMS[a.arm]}
        np.save(path.with_suffix(".npy"), pred)
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
