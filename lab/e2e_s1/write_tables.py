"""Rebuild the end-to-end readout from the released metered run JSON.

Full rescoring from prediction.npy was performed during the original run. The
large prediction arrays are not shipped; this script checks the saved layer
rows against each complete receipt and prints the published tables.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs/cloud/e2e_s1"
RUNS = ("m1_net0", "p8_net0", "m1_net1", "p8_net1")


def load(name):
    folder = DATA / name
    config = json.loads((folder / "config.json").read_text())
    layers = json.loads((folder / "per_layer.json").read_text())
    assert len(layers) in (3, 11, 12) and [r["layer"] for r in layers] == list(range(4, 4 + len(layers)))
    result_file = folder / "result.json"
    result = json.loads(result_file.read_text()) if result_file.exists() else None
    if result:
        assert result["config"] == config
        assert result["layers"] == layers
        assert abs(result["final_mse"] - layers[-1]["mse"]) < 1e-20
        prefix = json.loads((DATA / f'prefix_meter_net{config["index"]}.json').read_text())["units"]
        assert abs(result["total_units"] - prefix - layers[-1]["units_total_so_far"]) < 1e-5
    prefix = json.loads((DATA / f'prefix_meter_net{config["index"]}.json').read_text())["units"]
    assert prefix > 0 and layers[-1]["units_total_so_far"] > 0
    return config, layers, result


data = {name: load(name) for name in RUNS}
print("| run | network | through layer | raw MSE | metered units | suffix wall | peak RSS |")
print("|---|---:|---:|---:|---:|---:|---:|")
for name, (cfg, layers, result) in data.items():
    prefix = json.loads((DATA / f'prefix_meter_net{cfg["index"]}.json').read_text())["units"]
    units = result["total_units"] if result else prefix + layers[-1]["units_total_so_far"]
    wall = result["suffix_wall_s"] if result else sum(x["seconds"] for x in layers)
    rss = result["host_peak_rss_gib"] if result else max(x["host_peak_rss_gib"] for x in layers)
    print(f'| {name} | {cfg["index"]} | {layers[-1]["layer"]} | {layers[-1]["mse"]:.4e} | {units:,.0f} | {wall/3600:.2f} h | {rss:.1f} GiB |')

print("\n| metered component | m1_net0 units | p8_net0 units |")
print("|---|---:|---:|")
full = [(name, data[name][2]) for name in ("m1_net0", "p8_net0")]
parts = {}
for name, result in full:
    counted = {}
    for layer in result["layers"]:
        for key, (_, units) in layer.get("seconds_units_by_part", {}).items():
            counted[key] = counted.get(key, 0) + units
    parts[name] = counted
keys = ["prefix", "expand", "seed", "linear"] + sorted(set(parts["m1_net0"]) | set(parts["p8_net0"])) + ["other_nonlinear"]
for key in keys:
    vals = []
    for name, result in full:
        if key == "prefix":
            value = result["prefix_units"]
        elif key == "expand":
            value = result["phase_units"]["B_switch_expand"]
        elif key == "seed":
            value = result["phase_units"]["B_K4_seed"]
        elif key == "linear":
            value = sum(row["units_linear"] for row in result["layers"])
        elif key == "other_nonlinear":
            value = sum(row["units_nonlinear"] for row in result["layers"]) - sum(v for k, v in parts[name].items() if k != "project_old")
        else:
            value = parts[name].get(key, 0)
        vals.append(f"{value:,.0f}")
    print(f"| {key} | " + " | ".join(vals) + " |")

print("\n| layer | " + " | ".join(RUNS) + " |")
print("|---:|" + "---:|" * len(RUNS))
for layer in range(4, 16):
    cells = []
    for _, layers, _ in data.values():
        cells.append(next((f'{x["mse"]:.4e}' for x in layers if x["layer"] == layer), ""))
    print(f"| {layer} | " + " | ".join(cells) + " |")

print("\n| layer | " + " | ".join(f"{x} units" for x in RUNS) + " |")
print("|---:|" + "---:|" * len(RUNS))
for layer in range(4, 16):
    cells = []
    for _, layers, _ in data.values():
        cells.append(next((f'{sum(x[f"units_{k}"] for k in ("linear", "nonlinear", "projection")):,.0f}' for x in layers if x["layer"] == layer), ""))
    print(f"| {layer} | " + " | ".join(cells) + " |")
