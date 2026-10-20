"""Check the 32-pair complete-chain readings against its saved meter receipts."""

import json
from pathlib import Path


def check_p32(root: Path, numbers: dict[str, float]) -> None:
    source = root / "data/e2e_p32"
    result = json.loads((source / "result.json").read_text())
    layers = json.loads((source / "per_layer.json").read_text())
    parts = {}
    for layer in layers:
        for key, (_, units) in layer["seconds_units_by_part"].items():
            parts[key] = parts.get(key, 0.0) + units

    units = result["total_units"]
    raw = result["final_mse"]
    linear = sum(layer["units_linear"] for layer in layers)
    nonlinear = sum(layer["units_nonlinear"] + layer["units_projection"] for layer in layers)
    outside = (result["phase_units"]["B_switch_expand"] + linear + nonlinear - sum(parts.values())
               + parts["birth_pairs_compress"] + parts["birth_two_leg_store"]
               + result["phase_units"]["B_K4_seed"])
    rows = {
        "prefix": result["prefix_units"],
        "twoleg": parts["read_dense_two_leg"] + parts["read_dense_age0"] + parts["project_old"] + parts["read_core"],
        "feedback": parts.get("feedback_projection", 0.0),
        "oldslice": parts["read_slice_legs"],
        "nontree": parts["read_slice_pairs"],
        "outside": outside,
    }
    assert abs(sum(rows.values()) - units) < 1.0
    expected = {
        "e2e_p32_net0_raw": raw,
        "e2e_p32_net0_units": units,
        "e2e_p32_net0_cb": result["c_over_b"],
        "e2e_p32_net0_hours": result["suffix_wall_s"] / 3600,
        "e2e_p32_net0_peak_gib": result["host_peak_rss_gib"],
        "e2e_p32_over_m1_units": units / numbers["e2e_m1_net0_units"],
        "e2e_p32_over_m1_raw": raw / numbers["e2e_m1_net0_raw"],
        "e2e_p32_over_p8_units": units / numbers["e2e_p8_net0_units"],
        "e2e_p32_over_p8_raw": raw / numbers["e2e_p8_net0_raw"],
        "e2e_p32_over_tie": units / numbers["ledger_s4_tie_units_compressed"],
        "e2e_p32_over_counted": units / numbers["ledger_s4_total"],
        "e2e_p32_pair_units": parts["read_slice_pairs"] + parts["birth_pairs_compress"],
        "e2e_p32_linear_units": linear,
        "e2e_p32_slice_twoleg_share": (parts["read_slice_pairs"] + parts["read_slice_legs"]
                                         + parts["read_dense_two_leg"] + parts["read_dense_age0"]) / units,
    }
    expected.update({f"e2e_p32_rec_{key}_metered": value for key, value in rows.items()})
    expected.update({f"e2e_p32_l{layer['layer']}_raw": layer["mse"] for layer in layers})
    assert len(expected) == 32
    for key, actual in expected.items():
        saved = numbers[key]
        assert abs(saved - actual) <= 6e-6 * max(abs(actual), 1e-12), (key, saved, actual)
