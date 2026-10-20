"""Reprice the saved complete dense-chain operation trace for Table 7."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab/cand_k4x/r4"))
from trace_cost import cost_trace


def main():
    trace_dir = ROOT / "data/dense_trace"
    files = [trace_dir / "trace_seed.json", *(trace_dir / f"trace_layer{layer}.json" for layer in range(4, 16))]
    results = [cost_trace(json.loads(path.read_text())) for path in files]
    assert all(result["complete_mapping"] for result in results)
    calls = sum(result["total_calls"] for result in results)
    units = sum(result["complete_modeled_units"] for result in results)
    with (ROOT / "data/numbers.csv").open(newline="") as handle:
        values = {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}
    units += values["exec_s4_prefix_units"]
    assert calls == 203289
    assert abs(units - values["exec_s4_total_units"]) < 10
    print(json.dumps({"mapped_calls": calls, "complete_units": units, "raw_MSE": values["exec_s4_raw"]}))


if __name__ == "__main__":
    main()
