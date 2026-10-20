"""Inputs for Figure 5 from the released reduction measurements."""

import csv
import json
from pathlib import Path


def inputs(root: Path, numbers: dict[str, float]) -> dict:
    out = root / "runs/cloud/reduce_s1"
    recorded = out / "outputs"
    units = {
        "ck3": (numbers["ledger_s4_C_k3_births_at_real_kappa3_width_beyond_n_wide_legs"]
                + numbers["ledger_s4_C_k3_crossing_at_rank_16"]),
        "nontree": numbers["ledger_s4_non_tree_first_reads_dense_classes_merged"],
        "tree": numbers["ledger_s4_old_slice_reads"] + numbers["ledger_s4_feedback"],
        "star": (numbers["ledger_s4_K_4_sources_kappa3_legs_n_wide"]
                 + numbers["ledger_s4_young_leg_transport"]),
        "chainb": numbers["ledger_s4_chain_B_metered"],
    }
    weights = {}
    for tag, net, name in (("v1_n0", 0, "full_s4_b14"),
                           ("v2_n0", 0, "full_s4_b14_v2"),
                           ("v2_n1", 1, "full_s4_b14_v2")):
        result = json.loads((recorded / f"net{net}" / f"{name}.json").read_text())
        counted = {}
        for read in result["reads"]:
            if read["family"] != "seed" and read["birth"] <= 13 and read["birth"] < read["layer"] <= 14:
                family = read["family"]
                counted[family] = counted.get(family, 0.0) + read["w"]
        total = sum(counted.values())
        weights[tag] = {family: value / total for family, value in counted.items()}
    full = json.loads((recorded / "net0/full_s4_b14.json").read_text())
    removed = json.loads((recorded / "net0/n0_weight_33611.json").read_text())
    exact_gain = json.loads((root / "data/figure5_exact.json").read_text())
    gain = {
        "ck3": exact_gain["no_C_kappa3_with_star_mse"] / exact_gain["late_start4_mse"],
        "nontree": removed["mse"] / full["mse"],
        "nontree_alt": numbers["late4_tree_star0_age0"] / numbers["late_start4_noCk3_slices_net0"],
        "star": numbers["late_start4_noCk3_slices_net0"] / numbers["late_start4_noCk3_star_net0"],
        "tree_alone_over_B": numbers["late4_tree_star0_age0"] / numbers["late_B_alone_net0"],
    }
    with (out / "curve.csv").open(newline="") as handle:
        points = [dict(group=row["group"], label=row["label"], net=int(row["net"]),
                       units=float(row["units"]), raw=float(row["raw"]),
                       ratio=float(row["ratio_B"]), frontier=row["frontier"] == "True",
                       src=row["src"], note=row["note"]) for row in csv.DictReader(handle)]
    base = {net: next(point for point in points if point["net"] == net and point["group"] == "chain B")
            for net in (0, 1)}
    adjusted = {net: point["raw"] * max(0.1, point["units"] / 1024) for net, point in base.items()}

    def tie(point):
        return adjusted[point["net"]] / point["raw"] * 1024

    compressed = []
    with (root / "runs/cloud/reduce_s2/curve.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            point = {
                "label": row["point"], "net": int(row["network"]),
                "nt": int(row["nontree_rank"]), "ck": int(row["ck3_rank"]),
                "units": float(row["units"]), "kunits": float(row["kernel_available_units"]),
                "legs": float(row["retained_kappa3_leg_units"]),
                "raw": float(row["raw"]), "ratio": float(row["ratio_B"]),
                "control": row["is_control"] == "True",
                "frontier": row["on_combined_frontier"] == "True",
                "s1_same": float(row["s1_same_raw_units"]), "src": row["source"],
            }
            point["tie"] = tie(point)
            receipt = json.loads((root / row["source"]).read_text())
            assert abs(point["raw"] - receipt["mse"]) < 1e-15
            compressed.append(point)
    e2e = {}
    for tag in ("m1_net0", "p8_net0", "p32_net0"):
        path = (root / "data/e2e_p32/result.json") if tag == "p32_net0" else (root / "runs/cloud/e2e_s1" / tag / "result.json")
        result = json.loads(path.read_text())
        e2e[tag] = {"units": result["total_units"], "raw": result["final_mse"],
                    "ratio": result["final_mse"] / base[0]["raw"]}

    return {"pts": points, "tie": tie, "adjB": adjusted, "units": units,
            "rw": weights, "gain": gain, "pts2": compressed, "e2e": e2e}
