"""Recompute eight-network and 50-network paired ablation summaries."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fifty_rows(source, new_results=None, arms=("b", "active_off", "all_off"), batch="s2"):
    rows = list(csv.DictReader(source.open()))
    if new_results is not None:
        rows = [r for r in rows if r["batch"] == "s1"]
        seeds = json.loads((ROOT / "lab/ablate_s2/seeds.json").read_text())
        source_hashes = set()
        for index in range(8, 50):
            for arm in arms:
                value = json.loads((new_results / f"{arm}-plain" / f"mini{index}.json").read_text())
                assert value["arm"] == arm and value["index"] == index and value["seed"] == seeds[index]["seed"]
                source_hashes.add(value["source_sha256"])
                rows.append({"batch": batch, "arm": arm, "index": str(index),
                             **{key: value[key] for key in ("raw", "cb", "adj", "peak_rss_gib", "wall_s")}})
        assert len(source_hashes) == 1
    return rows


def metric(pairs):
    base = np.array([float(b["adj"]) for b, _ in pairs])
    arm = np.array([float(a["adj"]) for _, a in pairs])
    ratios = arm / base
    rng = np.random.default_rng(20261014)
    draw = rng.integers(0, len(pairs), size=(20000, len(pairs)))
    boot = (arm[draw].mean(axis=1) / base[draw].mean(axis=1),
            ratios[draw].mean(axis=1), np.median(ratios[draw], axis=1))
    return {"n": len(pairs), "indices": [int(b["index"]) for b, _ in pairs],
            "base_mean_adj": float(base.mean()), "arm_mean_adj": float(arm.mean()),
            "ratio_of_means": float(arm.mean() / base.mean()),
            "ratio_of_means_ci95": np.quantile(boot[0], [0.025, 0.975]).tolist(),
            "mean_network_ratio": float(ratios.mean()),
            "mean_network_ratio_ci95": np.quantile(boot[1], [0.025, 0.975]).tolist(),
            "median_network_ratio": float(np.median(ratios)),
            "median_network_ratio_ci95": np.quantile(boot[2], [0.025, 0.975]).tolist(),
            "worse_count": int((ratios > 1).sum()),
            "better_count": int((ratios < 1).sum()),
            "equal_count": int((ratios == 1).sum()),
            "raw_lower_count": sum(float(a["raw"]) < float(b["raw"]) for b, a in pairs)}


def summarize_fifty(source, source_s3, output, new_results=None, new_s3_results=None):
    panels = {
        "s2": fifty_rows(source, new_results),
        "s3": fifty_rows(source_s3, new_s3_results,
                           ("b", "young_off", "repr_off"), "s3"),
    }
    for panel in panels.values():
        assert len(panel) == 150 and len({(r["batch"], r["arm"], int(r["index"])) for r in panel}) == 150
    summary = []
    for arm in ("all_off", "active_off", "young_off", "repr_off"):
        indexed = {(r["batch"], r["arm"], int(r["index"])): r
                   for r in panels["s3" if arm in ("young_off", "repr_off") else "s2"]}
        batch = "s3" if arm in ("young_off", "repr_off") else "s2"
        for cohort, indices in (("s1_0_7", range(8)), ("s2_8_49", range(8, 50)),
                                ("pooled_0_49", range(50))):
            pairs = [(indexed[("s1" if i < 8 else batch, "b", i)],
                      indexed[("s1" if i < 8 else batch, arm, i)]) for i in indices]
            raw_max = max(pairs, key=lambda p: float(p[0]["raw"]))
            summary.append({"cohort": cohort, "arm": arm,
                            "raw_max_base_index": int(raw_max[0]["index"]),
                            "all": metric(pairs),
                            "without_raw_max_base": metric([p for p in pairs if p is not raw_max])})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    for item in summary:
        if item["cohort"] == "pooled_0_49":
            m = item["all"]
            print(item["arm"], f'{m["ratio_of_means"]:.4f}',
                  '[%0.4f, %0.4f]' % tuple(m["ratio_of_means_ci95"]),
                  f'{m["worse_count"]}/50')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=("8", "50"), default="8")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--s3-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--new-results", type=Path, help="replace the 42 new networks with a fresh matrix run")
    parser.add_argument("--new-s3-results", type=Path, help="replace the young and representation cells with a fresh grid")
    args = parser.parse_args()
    if args.panel == "50":
        summarize_fifty(args.input or ROOT / "data/ablate_s2/all_rows.csv",
                        args.s3_input or ROOT / "data/ablate_s3/all_rows.csv",
                        args.output or ROOT / "outputs/ablation50_summary.json",
                        args.new_results, args.new_s3_results)
        return
    if args.new_results is not None or args.new_s3_results is not None:
        parser.error("new results require --panel 50")
    args.input = args.input or ROOT / "data/ablation_observations.csv"
    args.output = args.output or ROOT / "outputs/ablation_summary.csv"
    rows = list(csv.DictReader(args.input.open()))
    by_key = {(r["arm"], r["learner"], int(r["index"])): r for r in rows}
    fields = ("arm", "learner", "n", "raw", "cb", "adj", "paired_adj_ratio", "median_ratio", "worse")
    out = []
    for learner in ("plain", "lt2"):
        for arm in ("b", "active_off", "young_off", "repr_off", "all_off"):
            data = [by_key[(arm, learner, i)] for i in range(8)]
            ratios = []
            for i, row in enumerate(data):
                control = ("b_same_machine" if arm == "all_off" and learner == "lt2" and i == 6
                           and ("b_same_machine", "lt2", 6) in by_key else "b")
                base = by_key[(control, learner, i)]
                if row["seed"] != base["seed"]:
                    raise ValueError("paired seeds differ")
                ratios.append(float(row["adj"]) / float(base["adj"]))
            out.append(dict(arm=arm, learner=learner, n=8,
                            raw=mean(float(r["raw"]) for r in data),
                            cb=mean(float(r["cb"]) for r in data),
                            adj=mean(float(r["adj"]) for r in data),
                            paired_adj_ratio=mean(ratios), median_ratio=median(ratios),
                            worse=sum(x > 1 for x in ratios)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    for row in out:
        print(row["learner"], row["arm"], f'{row["paired_adj_ratio"]:.6f}', f'{row["worse"]}/8')


if __name__ == "__main__":
    main()
