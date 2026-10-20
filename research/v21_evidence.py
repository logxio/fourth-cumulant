"""Report v21: the star-birth weights of Section 3 and the corrector seed check of Section 4.2, from saved files.

  python3 research/v21_evidence.py
      Seconds, NumPy only. Checks report/data/c45/PREDICT_1024.en.md against its release SHA-256, rebuilds the 75
      final-layer vectors of report/data/c45/arms_last.npz from the released predictions (report/data/c45/build.py),
      fits per network the weights of path, star and C.k3 births, rebuilds every c45_ value and the three 1024
      predictions, rescores the three seed reruns of #335351's corrector, and compares each value with data/numbers.csv.

Per network: A = [full - no_path, full - no_star, full - no_k3c] on the final layer, least squares c of truth - full on A,
weights w = 1 + c (1 is the recursion as written); cos(path, star) from A^T A; leave-one-out scales the network's star
contribution by the mean star weight of the other networks of its shape. Twelve frozen networks at 64x8, 96x12, 128x16
and 192x16 and official mini 5, 6, 7 at 1024. PREDICT_1024.en.md records the three 1024
predictions and their measured outcomes. runs/cloud/lt-seeds/ holds #335351's corrector configuration
retrained with seeds 0, 1 and 2 (training logs and mini results, paths removed).
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
C45 = ROOT / "report" / "data" / "c45"
PREDICT_SHA = "ab26ed8e67351262922356ab70dac4e8b2b928c0205fb00623f9b16aa2fe2c52"
SEEDS = ROOT / "runs" / "cloud" / "lt-seeds"
SMALL = ROOT / "runs" / "cloud" / "lt-s5" / "runs" / "Eg"
TOL = 1e-5  # numbers.csv keeps six significant digits
sys.path.insert(0, str(C45))
from build import collect  # noqa: E402


def _numbers():
    with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
        return {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}


def _close(got, want, key):
    assert abs(got - want) <= TOL * max(abs(want), 1e-300), (key, got, want)


def star_weights(numbers):
    assert hashlib.sha256((C45 / "PREDICT_1024.en.md").read_bytes()).hexdigest() == PREDICT_SHA
    vectors, sources = collect()
    assert json.loads((C45 / "sources.json").read_text()) == sources
    with np.load(C45 / "arms_last.npz") as saved:
        assert sorted(saved.files) == sorted(vectors)
        assert all(np.array_equal(saved[k], vectors[k]) for k in vectors)
    nets = sorted({tuple(int(x) for x in k.split("_")[:2]) for k in vectors})
    res = {}
    for n, s in nets:
        g, full = vectors[f"{n}_{s}_gt"], vectors[f"{n}_{s}_full"]
        A = np.stack([full - vectors[f"{n}_{s}_{a}"] for a in ("no_path", "no_star", "no_k3c")], 1)
        c, *_ = np.linalg.lstsq(A, g - full, rcond=None)
        G = A.T @ A
        mse = lambda p: float(((p - g) ** 2).mean())
        res[n, s] = dict(g=g, full=full, A=A, w=1 + c, cos=G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]),
                         R=mse(vectors[f"{n}_{s}_no_path"]) / mse(full), star=mse(vectors[f"{n}_{s}_no_star"]) / mse(full),
                         fit3=mse(full + A @ c) / mse(full))
    for (n, s), q in res.items():
        ws = float(np.mean([v["w"][1] for (n2, s2), v in res.items() if n2 == n and s2 != s]))
        q["loo"] = float(((q["full"] + q["A"][:, 1] * (ws - 1) - q["g"]) ** 2).mean()
                         / ((q["full"] - q["g"]) ** 2).mean())
    sm = [q for (n, _), q in res.items() if n < 1024]
    bg = [q for (n, _), q in res.items() if n == 1024]
    assert len(sm) == 12 and len(bg) == 3
    w, loo = [q["w"][1] for q in sm], [q["loo"] for q in sm]
    out = {"c45_small_networks": len(sm),
           "c45_small_wstar_min": min(w), "c45_small_wstar_max": max(w),
           "c45_small_wstar_median": float(np.median(w)), "c45_small_wstar_above1": sum(x > 1 for x in w),
           "c45_small_loo_min": min(loo), "c45_small_loo_max": max(loo),
           "c45_small_loo_geomean": float(np.exp(np.mean(np.log(loo)))), "c45_small_loo_wins": sum(x < 1 for x in loo),
           "c45_small_path_helps": sum(q["R"] < 1 for q in sm),
           "c45_small_cos_rule": sum((q["cos"] < 0) == (q["R"] < 1) for q in sm),
           "c45_upto128_cos_rule": sum((q["cos"] < 0) == (q["R"] < 1) for (n, _), q in res.items() if n <= 128)}
    for key, ix in (("wpath", 0), ("wstar", 1), ("wk3c", 2)):
        v = [q["w"][ix] for q in bg]
        out[f"c45_1024_{key}_min"], out[f"c45_1024_{key}_max"] = min(v), max(v)
    for key, field in (("cos", "cos"), ("loo", "loo"), ("fit3", "fit3"), ("nostar", "star")):
        v = [q[field] for q in bg]
        out[f"c45_1024_{key}_min"], out[f"c45_1024_{key}_max"] = min(v), max(v)
    out["c45_1024_pred1_holds"] = sum(q["w"][1] > 1 for q in bg)
    out["c45_1024_pred2_holds"] = sum(q["loo"] < 1 for q in bg)
    out["c45_1024_pred3_holds"] = sum(q["cos"] > 0 for q in bg)
    for key, value in out.items():
        _close(value, numbers[key], key)
    assert (out["c45_1024_pred1_holds"], out["c45_1024_pred2_holds"], out["c45_1024_pred3_holds"]) == (3, 3, 0)
    return out


def corrector_seeds(numbers):
    small = json.loads((SMALL / "result.json").read_text())
    out, ratios = {}, []
    for seed in (0, 1, 2):
        run = SEEDS / f"Eg_s{seed}"
        result = json.loads((run / "result.json").read_text())
        args = json.loads((run / "train.log").read_text().splitlines()[0][len("args "):].rsplit(" t=", 1)[0])
        assert args == result["args"] and args["seed"] == seed and args["ntrain"] == 1000
        assert {k: v for k, v in args.items() if k != "seed"} == {k: v for k, v in small["args"].items() if k != "seed"}
        corr, raw = np.array(result["mini_corr"]), np.array(result["mini_raw"])
        assert len(corr) == len(raw) == 100 and np.allclose(raw, small["mini_raw"], rtol=1e-12)
        ratios.append(float(corr.sum() / raw.sum()))
        out[f"lt2_seed{seed}_holdout_ratio"] = ratios[-1]
        out[f"lt2_seed{seed}_holdout_better"] = int((corr < raw).sum())
    out["lt2_seed_spread"] = max(ratios) - min(ratios)
    assert abs(ratios[0] - sum(small["mini_corr"]) / sum(small["mini_raw"])) < 1e-4  # the seed-0 rerun reproduces Eg
    for key, value in out.items():
        _close(value, numbers[key], key)
    return out


def check():
    numbers = _numbers()
    return {**star_weights(numbers), **corrector_seeds(numbers)}


if __name__ == "__main__":
    v = check()
    print(f"v21: {len(v)} values match data/numbers.csv; star weight > 1 on {v['c45_small_wstar_above1']}/12 small "
          f"networks ({v['c45_small_wstar_min']:.3g}-{v['c45_small_wstar_max']:.3g}), leave-one-out lowers the error on "
          f"{v['c45_small_loo_wins']}/12 (geometric mean {v['c45_small_loo_geomean']:.3g}); 1024 predictions hold on "
          f"{v['c45_1024_pred1_holds']}, {v['c45_1024_pred2_holds']} and {v['c45_1024_pred3_holds']} of 3; corrector "
          f"seeds 0-2 give {v['lt2_seed0_holdout_ratio']:.4f}, {v['lt2_seed1_holdout_ratio']:.4f}, "
          f"{v['lt2_seed2_holdout_ratio']:.4f}.")
