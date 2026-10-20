"""Recompute the 102 added report values from released per-network evidence."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ablation.summarize import metric

DATA = ROOT / "data"
ABL = DATA / "ablate_s3"


def grader_steps():
    rows = list(csv.DictReader((DATA / "grader_step_scores.csv").open(newline="")))
    ids = (332957, 334990, 335127, 335162, 335185, 335300, 335301, 336537)   # Table 4, steps 1-8
    corpus = (335351, 336536)   # the 1,000-network corrector and the wider corrector on 3,048 networks
    scores = {sid: np.array([float(r["adjusted_final_layer_score"]) for r in rows
                             if int(r["submission"]) == sid]) for sid in ids + corpus}
    raws = {sid: np.array([float(r["final_layer_mse"]) for r in rows
                           if int(r["submission"]) == sid]) for sid in ids + corpus}
    assert len(rows) == 500
    for sid in ids + corpus:
        assert [int(r["network"]) for r in rows if int(r["submission"]) == sid] == list(range(50))
    draws = np.random.default_rng(20261014).integers(0, 50, (20000, 50))
    result = {}

    def compare(name, pairs):
        ratio = 1.0
        resamples = np.ones(len(draws))
        per_network = np.ones(50)
        for old, new in pairs:
            a, b = scores[old], scores[new]
            ratio *= b.mean() / a.mean()
            resamples *= b[draws].mean(axis=1) / a[draws].mean(axis=1)
            per_network *= b / a
        lo, hi = np.quantile(resamples, (0.025, 0.975))
        prefix = f"lad50_{name}"
        result[f"{prefix}_adj_ratio_of_means"] = ratio
        result[f"{prefix}_ci_lo"] = lo
        result[f"{prefix}_ci_hi"] = hi
        result[f"{prefix}_better"] = int((per_network < 1).sum())
        result[f"{prefix}_net_min"] = per_network.min()
        result[f"{prefix}_net_max"] = per_network.max()

    for step in range(1, len(ids)):
        compare(f"step{step + 1}", ((ids[step - 1], ids[step]),))
    compare("own_steps", ((334990, 335162), (335185, 335301)))
    compare("v29_to_b", ((332957, 335301),))
    compare("corpus", ((335351, 336537),))
    compare("wide", ((335351, 336536),))
    for name, (old, new) in (("corpus", (335351, 336537)), ("wide", (335351, 336536))):
        a, b = raws[old], raws[new]
        resamples = b[draws].mean(axis=1) / a[draws].mean(axis=1)
        result[f"lad50_{name}_raw_ratio_of_means"] = b.mean() / a.mean()
        result[f"lad50_{name}_raw_ci_lo"], result[f"lad50_{name}_raw_ci_hi"] = np.quantile(resamples, (0.025, 0.975))
        result[f"lad50_{name}_raw_better"] = int((b < a).sum())
    return result


def ablations():
    rows = list(csv.DictReader((ABL / "all_rows.csv").open(newline="")))
    assert len(rows) == 150 and set(rows[0]) == {
        "batch", "arm", "index", "seed", "raw", "cb", "adj", "peak_rss_gib", "wall_s"}
    indexed = {(r["batch"], r["arm"], int(r["index"])): r for r in rows}
    assert len(indexed) == 150
    saved = {r["arm"]: r for r in json.loads((ABL / "summary.json").read_text())
             if r["learner"] == "plain"}
    assert set(saved) >= {"young_off", "repr_off"}
    result = {}
    with np.load(DATA / "mini_truth.npz") as data:
        truth = data["all_layer_means"]
        for arm in ("young_off", "repr_off"):
            pairs = [(indexed[("s1" if i < 8 else "s3", "b", i)],
                      indexed[("s1" if i < 8 else "s3", arm, i)]) for i in range(50)]
            assert all(b["seed"] == a["seed"] for b, a in pairs)
            measured = metric(pairs)
            expected = saved[arm]
            assert np.isclose(measured["ratio_of_means"], expected["ratio_of_means"], rtol=1e-12)
            assert np.allclose(measured["ratio_of_means_ci95"], expected["ratio_of_means_ci95"], rtol=1e-12)
            assert measured["worse_count"] == expected["worse_count"]
            for index in range(8, 50):
                prediction = np.load(ABL / f"{arm}-plain" / f"mini{index}.npy")
                assert prediction.shape == (16, 1024)
                raw = np.mean((prediction[-1].astype(np.float64) - truth[index, -1]) ** 2)
                assert np.isclose(raw, float(pairs[index][1]["raw"]), rtol=1e-6)
            prefix = f"abl50_plain_{arm}"
            result[f"{prefix}_adj_ratio_of_means"] = measured["ratio_of_means"]
            result[f"{prefix}_adj_ratio_of_means_ci_lo"], result[f"{prefix}_adj_ratio_of_means_ci_hi"] = measured["ratio_of_means_ci95"]
            result[f"{prefix}_adj_median"] = measured["median_network_ratio"]
            result[f"{prefix}_adj_ratio_of_means_ex6"] = (
                sum(float(a["adj"]) for i, (_, a) in enumerate(pairs) if i != 6)
                / sum(float(b["adj"]) for i, (b, _) in enumerate(pairs) if i != 6))
            result[f"{prefix}_worse"] = measured["worse_count"]
            result[f"{prefix}_cb"] = np.mean([float(a["cb"]) for _, a in pairs])
            result[f"{prefix}_peak_gib"] = max(float(a["peak_rss_gib"]) for _, a in pairs)
    return result


def network_six():
    mechanism = json.loads((ABL / "mechanism.json").read_text())
    parity = json.loads((ABL / "trace_parity.json").read_text())
    assert mechanism["trace_parity"] == parity
    assert parity["usable_for_old_s1_arm_table"] is None
    with np.load(DATA / "mini_truth.npz") as file:
        truth15 = file["all_layer_means"][6, 14]
    old = np.load(ROOT / "runs/cloud/ablate-s1/results/b-plain/mini6.npy")
    old_active = np.load(ROOT / "runs/cloud/ablate-s1/results/active_off-plain/mini6.npy")
    old_mse = np.mean((old[14] - truth15) ** 2)
    excess = (old[14] - truth15) ** 2 - (old_active[14] - truth15) ** 2
    top20 = np.sort(excess)[-20:].sum() / excess.sum()
    neurons = list(csv.DictReader((ABL / "old_machine_neurons.csv").open(newline="")))
    assert len([r for r in neurons if r["step"] == "active_off"]) == 20
    new = ABL / "new_machine"
    control, trace, trace2 = (np.load(new / f"{name}.npy") for name in ("control", "trace", "trace2"))
    assert control.shape == trace.shape == trace2.shape == old.shape
    assert np.array_equal(control, trace) and np.array_equal(control, trace2)
    new_mse = np.mean((control[14] - truth15) ** 2)
    with np.load(new / "gate.npz") as gate:
        kept = int(gate["kept"])
        dropped = gate["w1_prev"][kept:]
        assert kept == 800 and int(gate["out_kept"]) == 768
        assert len(dropped) == 224 and dropped.max() < 0.014
        assert np.isclose(np.median(dropped), mechanism["new_machine_trace"]["dropped_w1_median"], rtol=1e-5)
    return {
        "abl_net6_old_b_layer15": old_mse,
        "abl_net6_new_b_layer15": new_mse,
        "abl_net6_cross_machine_max_abs": np.max(np.abs(control - old)),
        "abl_net6_same_machine_trace_max_abs": np.max(np.abs(control - trace2)),
        "abl_net6_cut_columns": len(dropped),
        "abl_net6_top20_old_share": top20,
    }


def check_report():
    result = {**grader_steps(), **ablations(), **network_six()}
    for row in csv.DictReader((DATA / "scored_young_settings.csv").open(newline="")):
        assert int(row["young_round"]) == 32
        result[f"young_keep_mass_{row['submission']}"] = 1 - float(row["young_eps"])
    report = {row["key"]: float(row["value"]) for row in
              csv.DictReader((DATA / "numbers.csv").open(newline=""))}
    for shape in ("256x4", "512x8"):
        lo = report[f"nl64_{shape}_rel_1024_ci90_lo"]
        hi = report[f"nl64_{shape}_rel_1024_ci90_hi"]
        result[f"nl64_{shape}_inside_080_125"] = int(0.8 <= lo and hi <= 1.25)
    assert len(result) == 102
    for key, actual in result.items():
        if key.endswith("_inside_080_125"):
            assert actual == int(key.startswith("nl64_512x8"))
            continue
        assert np.isclose(actual, report[key], rtol=5e-6, atol=1e-12), (key, actual, report[key])
    return result


if __name__ == "__main__":
    values = check_report()
    target = ROOT / "outputs" / "v15_evidence.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps({key: float(value) for key, value in values.items()}, indent=2, sort_keys=True) + "\n")
    print(f"{len(values)} added measurements match data/numbers.csv")
