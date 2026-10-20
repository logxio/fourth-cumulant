"""Report v19: the frozen 1024 column of Table 3 and the column-cut row of Table 5, rescored from saved predictions.

  python3 research/v19_evidence.py
      Seconds, NumPy only. Checks data/frozen_1024/PREREG.md and config.json against PREREG.sha256 and the hash the
      report quotes, rescores the twelve saved 1024 x 16 birth-arm predictions of networks 5, 6 and 7 against
      data/mini_truth.npz, rebuilds every fz1024_ value and the preregistered order on each network, rescores the 100
      saved final layers of the column-cut panel (chain B, and chain B with only our column cut off, mini 0-49), rebuilds
      the paired ratios with the same 20,000 network resamples, and compares each value with data/numbers.csv.

data/frozen_1024/ holds PREREG.md and config.json as frozen (networks 5, 6 and 7, four arms, one order), the run logs
and the 16 x 1024 predictions of all twelve arms. data/colsplit/ holds one row per network and arm
(raw, FLOPs, adj, peak memory), the final-layer predictions, and nocol.diff, the two-hunk switch between chain B's source
and the column-cut arm; ablation/chain.py carries the same switch as AW_NOCOL (ablation/run.py --arm col_off).
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FZ = ROOT / "data" / "frozen_1024"
CS = ROOT / "data" / "colsplit"
PREREG_SHA = "0f5c0d7e392eb7b7d3410d3c39596c0579f0b73211b2fea3a1856afeb514cf34"
ARMS = ("full", "no_k3c", "no_path", "no_slicex")
SPECS = {"full": "full",
         "no_k3c": "path+star,slice4+slice31+slice22+slice211",
         "no_path": "star+three2,slice4+slice31+slice22+slice211",
         "no_slicex": "path+star+three2,slice4+slice31,!slice22+slice211"}
B = 2 ** 41
TOL = 1e-5  # numbers.csv keeps six significant digits


def _numbers():
    with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
        return {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}


def _close(got, want, key):
    assert abs(got - want) <= TOL * max(abs(want), 1e-300), (key, got, want)


def frozen_1024(numbers, truth):
    recorded = dict(line.split()[::-1] for line in (FZ / "PREREG.sha256").read_text().splitlines())
    for name, digest in recorded.items():
        assert hashlib.sha256((FZ / name).read_bytes()).hexdigest() == digest, name
    assert recorded["PREREG.md"] == PREREG_SHA
    config = json.loads((FZ / "config.json").read_text())
    assert config["birth_indices"] == [5, 6, 7] and config["birth_arms"] == SPECS
    prereg = (FZ / "PREREG.md").read_text()
    assert "Official mini indices 5,6,7" in prereg and "no_k3c MSE > no_path MSE > no_slicex MSE" in prereg
    out, mse = {}, {}
    nets = sorted({int(m.group(1)) for p in FZ.glob("birth*_full.npy") for m in [re.match(r"birth(\d+)_", p.name)]})
    assert nets == config["birth_indices"]
    for i in nets:
        for arm in ARMS:
            pred = np.load(FZ / f"birth{i}_{arm}.npy", allow_pickle=False)
            assert pred.shape == (16, 1024) and np.isfinite(pred).all()
            per_layer = np.mean((pred - truth[i]) ** 2, axis=1)
            log = (FZ / f"birth{i}_{arm}.log").read_text()
            assert f"spec={SPECS[arm]} mlp {i} " in log
            assert abs(float(re.search(r"final ([0-9.e+-]+)", log).group(1)) / per_layer[-1] - 1) < 1e-3
            mse[i, arm] = out[f"fz1024_mini{i}_{arm}"] = float(per_layer[-1])
        for arm in ARMS[1:]:
            out[f"fz1024_mini{i}_{arm}_over_full"] = mse[i, arm] / mse[i, "full"]
    for arm in ARMS[1:]:
        q = [mse[i, arm] / mse[i, "full"] for i in nets]
        out[f"fz1024_{arm}_over_full_mean"] = float(np.mean(q))
        out[f"fz1024_{arm}_over_full_min"], out[f"fz1024_{arm}_over_full_max"] = min(q), max(q)
    out["fz1024_full_min"] = min(mse[i, "full"] for i in nets)
    out["fz1024_full_max"] = max(mse[i, "full"] for i in nets)
    out["fz1024_networks_preregistered"] = len(config["birth_indices"])
    out["fz1024_networks_run"] = len(nets)
    out["fz1024_order_holds"] = sum(mse[i, "no_k3c"] > mse[i, "no_path"] > mse[i, "no_slicex"] for i in nets)
    out["fz1024_k3c_largest"] = sum(mse[i, "no_k3c"] == max(mse[i, a] for a in ARMS[1:]) for i in nets)
    for key, value in out.items():
        _close(value, numbers[key], key)
    assert out["fz1024_order_holds"] == 3 and out["fz1024_k3c_largest"] == 3
    return out


def colsplit(numbers, truth_final):
    with (CS / "rows.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 100
    final = np.load(CS / "final_layer.npz", allow_pickle=False)
    idx = [int(i) for i in final["index"]]
    assert idx == list(range(50))
    arm = {}
    for name in ("b", "col_off"):
        recs = {int(r["index"]): r for r in rows if r["arm"] == name}
        assert sorted(recs) == idx
        raw = np.array([np.mean((final[name][j] - truth_final[i]) ** 2) for j, i in enumerate(idx)])
        flops = np.array([float(recs[i]["flops"]) for i in idx])
        adj = raw * np.maximum(0.1, flops / B)
        for j, i in enumerate(idx):
            assert abs(raw[j] / float(recs[i]["raw"]) - 1) < 1e-12 and abs(adj[j] / float(recs[i]["adj"]) - 1) < 1e-12
        arm[name] = {"raw": raw, "cb": flops / B, "adj": adj,
                     "peak": max(float(recs[i]["peak_rss_gib"]) for i in idx)}
    rng = np.random.default_rng(20261017)  # the panel's own resampling: 20,000 paired draws of the 50 networks
    boot = rng.integers(0, len(idx), size=(20000, len(idx)))
    out = {}
    for q in ("adj", "raw", "cb"):
        num, den = arm["col_off"][q], arm["b"][q]
        out[f"colcut50_{q}_ratio_of_means"] = float(num.sum() / den.sum())
        lo, hi = np.percentile(num[boot].sum(1) / den[boot].sum(1), [2.5, 97.5])
        out[f"colcut50_{q}_ratio_of_means_ci_lo"], out[f"colcut50_{q}_ratio_of_means_ci_hi"] = float(lo), float(hi)
    out["colcut50_cb_B"], out["colcut50_cb_off"] = float(arm["b"]["cb"].mean()), float(arm["col_off"]["cb"].mean())
    per = arm["col_off"]["adj"] / arm["b"]["adj"]
    out["colcut50_adj_median"] = float(np.median(per))
    out["colcut50_adj_net_min"], out["colcut50_adj_net_max"] = float(per.min()), float(per.max())
    out["colcut50_worse"] = int((per > 1).sum())
    out["colcut50_raw_B_lower"] = int((arm["b"]["raw"] < arm["col_off"]["raw"]).sum())
    out["colcut46_adj_ratio_of_means"] = float(arm["col_off"]["adj"][4:].sum() / arm["b"]["adj"][4:].sum())
    for key, value in out.items():
        _close(value, numbers[key], key)
    for key, name in (("colcut50_peak_gib_B", "b"), ("colcut50_peak_gib_off", "col_off")):
        assert abs(arm[name]["peak"] - numbers[key]) < 1e-3, key  # rows keep four decimals
    assert out["colcut50_worse"] == 50 and out["colcut50_adj_ratio_of_means_ci_lo"] > 1
    return out


def check():
    numbers = _numbers()
    with np.load(ROOT / "data" / "mini_truth.npz") as handle:
        truth = handle["all_layer_means"]
    a = frozen_1024(numbers, truth)
    b = colsplit(numbers, truth[:, -1])
    return {**a, **b}


if __name__ == "__main__":
    values = check()
    print(f"v19: {len(values)} values match data/numbers.csv; frozen 1024 order holds on "
          f"{values['fz1024_order_holds']} of {values['fz1024_networks_run']} networks run "
          f"({values['fz1024_networks_preregistered']} preregistered); column cut off "
          f"{values['colcut50_adj_ratio_of_means']:.4f}x adj, worse on {values['colcut50_worse']}/50.")
