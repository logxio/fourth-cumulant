"""Rebuild the compressed-state comparison from the released result JSON.

The original run separately rescored saved prediction arrays. This portable
readout uses their validated JSON receipts and checks every value against the
frozen curve.csv; it does not claim a fresh prediction rescore.
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs/cloud/reduce_s2"
OLD = list(csv.DictReader((ROOT / "runs/cloud/reduce_s1/curve.csv").open()))
FROZEN = {row["point"]: row for row in csv.DictReader((DATA / "curve.csv").open())}
FILES = sorted((DATA / "outputs/net0").glob("*.json"))
assert len(FILES) == len(FROZEN) == 8
OLD_FRONTIER = [v for v in OLD if int(v["net"]) == 0 and v["frontier"] == "True"]
CHAIN_B = next(v for v in OLD if int(v["net"]) == 0 and v["group"] == "chain B")

rows = []
for file in FILES:
    item = json.loads(file.read_text())
    name = file.stem
    assert name in FROZEN and item["index"] == 0
    leg_units = 0.0
    if item["ranks"]["ck3"] > 0:
        for birth, life in item["plan"]["life"].get("ck3", {}).items():
            birth = int(birth)
            if item["start"] <= birth <= 13:
                leg_units += max(0, min(int(life), 14 - birth)) * (63 * birth - 7)
    units = item["units"] + leg_units
    raw = item["mse"]
    ref = min((v for v in OLD_FRONTIER if float(v["units"]) <= units + 1e-8), key=lambda v: float(v["raw"]))
    same_raw = min((float(v["units"]) for v in OLD_FRONTIER if float(v["raw"]) <= raw), default=float("nan"))
    ratio = raw / float(ref["raw"])
    row = FROZEN[name]
    for key, value in {"units": units, "kernel_available_units": item["units"],
                       "retained_kappa3_leg_units": leg_units, "raw": raw,
                       "s1_budget_units": float(ref["units"]),
                       "s1_budget_raw": float(ref["raw"]),
                       "raw_ratio_to_s1": ratio,
                       "ratio_B": raw / float(CHAIN_B["raw"]),
                       "s1_same_raw_units": same_raw,
                       "seconds": item["seconds"]}.items():
        assert math.isclose(float(row[key]), value, rel_tol=1e-9, abs_tol=1e-10), (name, key)
    assert row["s1_budget_point"] == ref["label"]
    assert (row["is_control"] == "True") == (item["ranks"]["nontree"] == item["ranks"]["ck3"] == 0)
    assert (ROOT / row["source"]).exists() and (ROOT / row["s1_source"]).exists()
    rows.append((name, item, row, units, ratio))

for name, item, row, units, _ in rows:
    competitors = [(float(v["units"]), float(v["raw"])) for v in OLD_FRONTIER]
    competitors += [(u, other["mse"]) for other_name, other, _, u, _ in rows if other_name != name]
    frontier = not any(u <= units and raw <= item["mse"] and (u < units or raw < item["mse"])
                       for u, raw in competitors)
    assert (row["on_combined_frontier"] == "True") == frontier, name

print("| point | network | non-tree / C-kappa3 rank | units | kernel-available units | final raw MSE | raw / B | old envelope raw | new / old | seconds |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for name, item, row, units, ratio in sorted(rows, key=lambda x: x[3]):
    print(f'| {name} | 0 | {item["ranks"]["nontree"]}/{item["ranks"]["ck3"]} | {units:.2f} | {item["units"]:.2f} | {item["mse"]:.8e} | {item["mse"]/float(CHAIN_B["raw"]):.4f} | {float(row["s1_budget_raw"]):.8e} | {ratio:.4f} | {item["seconds"]:.1f} |')

frontier = [name for name, _, row, _, _ in rows if row["on_combined_frontier"] == "True" and row["is_control"] != "True"]
assert sorted(frontier) == ["w9_n1", "w9_n16", "w9_n4"]
print(f"\nValidated {len(rows)} released JSON receipts; new combined-frontier points: {', '.join(sorted(frontier))}.")
