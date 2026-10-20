"""Run metered exact K=4 and augmented K=3 on three fixed 128x16 MLPs."""

from __future__ import annotations

import os

# Fix BLAS before NumPy or the research modules enter this process.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import multiprocessing
from pathlib import Path
import resource
import sys
import time
import warnings


ROOT = Path(__file__).resolve().parent
WEIGHT_HASH = {
    3: "eeaa2243c7c57d0155692f4e136fa7caa51d9807f37a5dc840f46bde1286b5e3",
    4: "6c01182958971fb6c63da14c64da20048e5f237b6c008e237c73bafaffad0fa3",
    5: "a920e669d7d3679d8cb3d533dabbd2a30ca45195a90fcbc43b7d56a24e625855",
}
TRUTH_HASH = {
    3: "033bda54ff4c8e7cbaa0e6cc95131d2456825b425735f99fe696307041378081",
    4: "3e66c6a6adb1ea5709a206aeb3822d2b7e08f915da4870e7f2b496a1f36bc23d",
    5: "135b93738cc4a1fba92c82286e208ed57300639eae02b14766f8db7840719112",
}


def _rss_mib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (2**20 if sys.platform == "darwin" else 2**10)


def _inputs(seed):
    import numpy as np

    rng = np.random.default_rng(seed)
    weights = np.stack([(rng.standard_normal((128, 128)) * np.sqrt(2 / 128))
                        .astype(np.float32) for _ in range(16)])
    assert hashlib.sha256(weights.tobytes()).hexdigest() == WEIGHT_HASH[seed]
    truth_path = ROOT / f"data/laptop_truth_128/seed{seed}.npz"
    assert hashlib.sha256(truth_path.read_bytes()).hexdigest() == TRUTH_HASH[seed]
    with np.load(truth_path) as saved:
        truth = saved["gt"]
    return weights, truth


def _run_arm(spec):
    import numpy as np
    import flopscope as flops
    warnings.filterwarnings("ignore", message=r"flops\.configure\(\) configures")
    flops.configure(symmetry_warnings=False)

    sys.path.insert(0, str(ROOT / "tools"))
    seed, arm = spec
    weights, truth = _inputs(seed)
    warnings.filterwarnings("ignore", message="stats.norm.cdf promoted its float32 input")
    if arm == "k4":
        from k4_exact import run_exact
        prediction, units = run_exact(weights, dtype="f32")
    else:
        from k3_aug import predict as run_k3_aug
        with flops.BudgetContext(flop_budget=10**18, quiet=True) as budget:
            prediction = run_k3_aug(weights)
        units = budget.flops_used / 2**31
    mse = float(np.mean((np.asarray(prediction[-1], dtype=np.float64) - truth[-1]) ** 2))
    return dict(seed=seed, arm=arm, mse=mse, units=units,
                cpu=time.process_time(), rss=_rss_mib())


def _run_seed(seed):
    import numpy as np
    import flopscope as flops
    warnings.filterwarnings("ignore", message=r"flops\.configure\(\) configures")
    flops.configure(symmetry_warnings=False)

    sys.path.insert(0, str(ROOT / "tools"))
    from k4_exact import run_exact
    from k3_aug import predict as run_k3_aug

    weights, truth = _inputs(seed)
    warnings.filterwarnings("ignore", message="stats.norm.cdf promoted its float32 input")

    p4, u4 = run_exact(weights, dtype="f32")
    with flops.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        p3 = run_k3_aug(weights)
    u3 = budget.flops_used / 2**31
    e4 = float(np.mean((p4[-1] - truth[-1]) ** 2))
    e3 = float(np.mean((np.asarray(p3[-1], dtype=np.float64) - truth[-1]) ** 2))
    return dict(seed=seed, k4=e4, k3=e3, u4=u4, u3=u3,
                cpu=time.process_time(),
                rss=_rss_mib())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(3, 4, 5),
                        help="run one network in this process; default runs all three")
    args = parser.parse_args()
    seeds = [args.seed] if args.seed is not None else [3, 4, 5]
    start = time.monotonic()
    if len(seeds) == 1:
        rows = [_run_seed(seeds[0])]
        rss_upper = rows[0]["rss"]
    else:
        with ProcessPoolExecutor(max_workers=3,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            k4 = list(pool.map(_run_arm, [(seed, "k4") for seed in seeds]))
        with ProcessPoolExecutor(max_workers=3,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            k3 = list(pool.map(_run_arm, [(seed, "k3") for seed in seeds]))
        rows = [dict(seed=seed, k4=a["mse"], k3=b["mse"],
                     u4=a["units"], u3=b["units"], cpu=a["cpu"] + b["cpu"],
                     rss=max(a["rss"], b["rss"]))
                for seed, a, b in zip(seeds, k4, k3)]
        rss_upper = max(sum(row["rss"] for row in k4),
                        sum(row["rss"] for row in k3)) + _rss_mib()
    wall = time.monotonic() - start
    for row in rows:
        print(f"seed {row['seed']}: K4 raw={row['k4']:.9e} units={row['u4']:.6f}; "
              f"K3a raw={row['k3']:.9e} units={row['u3']:.6f}; "
              f"K3a/K4={row['k3']/row['k4']:.3f}x; "
              f"CPU={row['cpu']:.2f}s RSS={row['rss']:.1f}MiB")
    if len(rows) == 3:
        k4_mean = sum(row["k4"] for row in rows) / 3
        k3_mean = sum(row["k3"] for row in rows) / 3
        print(f"mean: K4 raw={k4_mean:.9e}; K3a raw={k3_mean:.9e}; "
              f"K3a/K4={k3_mean/k4_mean:.3f}x")
    print(f"run: wall={wall:.2f}s RSS<={rss_upper:.1f}MiB")


if __name__ == "__main__":
    main()
