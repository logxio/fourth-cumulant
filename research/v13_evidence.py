"""Recompute the corrector protocol (#336537, against #335351's corrector) and network-six layer measurements."""
import argparse
import ast
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from lt2_embed import embed  # noqa: E402

RUN = ROOT / "runs/cloud/corpus-s6/runs/E32"   # #336537's corrector: full split 0-999 plus corpus 0-2047
SMALL = ROOT / "runs/cloud/lt-s5/runs/Eg"      # #335351's corrector: the same configuration, full split only
ABLATION = ROOT / "runs/cloud/ablate-s1/results"
CONFIG = ("emb", "hidden", "depth", "act", "batch", "seq_epochs", "seq_lr", "seq_beta", "epochs", "e2e_epochs",
          "seed", "fm", "ext", "uin", "xall", "embw")


def parse(run):
    lines = (run / "train.log").read_text().splitlines()
    args = json.loads(lines[0].split(" ", 1)[1].rsplit(" t=", 1)[0])
    start = next(i for i, line in enumerate(lines) if line.startswith("local eps rms"))
    rms = [float(x) for x in " ".join(lines[start:start + 3]).split("[", 1)[1].split("]", 1)[0].split()]
    epochs = [line for line in lines if line.startswith("seq ep ")]
    holdout = [float(line.split("holdout ratio ")[1].split()[0]) for line in epochs]
    return lines, args, rms, holdout


def blob_of(path):
    estimator = path.read_text()
    start = estimator.index("_LT2_BLOB = {")
    return ast.literal_eval(estimator[start + len("_LT2_BLOB = "):estimator.index("\n}", start) + 2])


def corrector():
    lines, args, rms, holdout = parse(RUN)
    small_lines, small_args, _, small_holdout = parse(SMALL)
    assert (args["ntrain"], args["emb"], args["hidden"], args["depth"],
            args["act"], args["batch"], args["seq_epochs"], args["seq_lr"],
            args["seq_beta"], args["epochs"], args["e2e_epochs"], args["seed"],
            args["fm"]) == (3048, 32, 128, 3, "gelu", 4, 30, 0.002,
                            0.0, 0, 0, 0, False)
    assert small_args["ntrain"] == 1000 and all(args[key] == small_args[key] for key in CONFIG)
    fin = next(line for line in lines if line.startswith("fin ")).split()
    assert fin[:4] == next(line for line in small_lines if line.startswith("fin ")).split()[:4]
    assert len(rms) == 16
    assert len(holdout) == 30 and next(line for line in lines if line.startswith("best ")).startswith("best seq29")
    wall = float(next(line for line in lines if line.startswith("saved ")).rsplit("t=", 1)[1].rstrip("s"))
    result = json.loads((RUN / "result.json").read_text())
    small = json.loads((SMALL / "result.json").read_text())
    ratio = np.array(result["mini_corr"]) / np.array(result["mini_raw"])
    assert len(ratio) == 100 and abs(sum(result["mini_corr"]) / sum(result["mini_raw"]) - 0.8729) < 1e-4
    assert abs(sum(small["mini_corr"]) / sum(small["mini_raw"]) - 0.8820) < 1e-4
    versus_small = np.array(result["mini_corr"]) / np.array(small["mini_corr"])
    parity = [float(line.split("= ")[1].split()[0]) for line in (RUN / "parity.log").read_text().splitlines()]

    with np.load(RUN / "lt2_weights.npz") as weights, np.load(SMALL / "lt2_weights.npz") as small_weights:
        blob = blob_of(ROOT / "estimator.py")
        types = {"f": "<f4", "d": "<f8", "q": "<i8", "l": "<i8"}
        assert sorted(blob) == sorted(weights.files)
        for key, (shape, dtype, encoded) in blob.items():
            embedded = np.frombuffer(base64.b64decode(encoded), dtype=types[dtype]).reshape(shape)
            assert np.array_equal(embedded, weights[key]), key
        shipped = sum(weights[key].size for key in weights.files)
        scale = weights["iz"]
        layer, column = np.unravel_index(scale.argmax(), scale.shape)
        assert (layer, column) == (0, 3) and small_weights["iz"][layer, column] == 0
        assert np.argwhere((scale == 0) != (small_weights["iz"] == 0)).tolist() == [[0, 3]]
    # the archived exports rebuild both submitted files byte for byte: #336537 from E32, #335351 from Eg
    source = (ROOT / "estimator.py").read_text()
    assert hashlib.md5(embed(source, RUN / "lt2_weights.npz").encode()).hexdigest() == "23b96f59dd1d6b9e9eb1f0b07a0cd6a4"
    assert hashlib.md5(embed(source, SMALL / "lt2_weights.npz").encode()).hexdigest() == "59d9df12826abeeb86fc89c99ade42a7"

    grader = [float(row["corrected_over_base_raw"]) for row in
              csv.DictReader((ROOT / "data/lt2_grader_ratios.csv").open())]
    assert len(grader) == 50
    return {
        "lt2_inputs_per_neuron": int(fin[1]),
        "lt2_trained_parameters": int(fin[3]),
        "lt2_epochs": args["seq_epochs"],
        "lt2_peak_lr": args["seq_lr"],
        "lt2_batch_networks": args["batch"],
        "lt2_train_networks": args["ntrain"],
        "lt2_corpus_networks": args["ntrain"] - small_args["ntrain"],
        "lt2_steps": args["seq_epochs"] * -(-args["ntrain"] // args["batch"]),
        "lt2_target_rms_layer1": rms[1],
        "lt2_target_rms_layer15": rms[15],
        "lt2_holdout_ratio_epoch1": holdout[0],
        "lt2_train_wall_s": wall,
        "lt2_holdout_better": int((ratio < 1).sum()),
        "lt2_holdout_ratio_min": float(ratio.min()),
        "lt2_holdout_ratio_max": float(ratio.max()),
        "lt2_left_layer1": result["local_nmse"][1],
        "lt2_left_layer15": result["local_nmse"][15],
        "lt2_small_holdout_ratio": sum(small["mini_corr"]) / sum(small["mini_raw"]),
        "lt2_small_holdout_ratio_epoch1": small_holdout[0],
        "lt2_corpus_mini_ratio": sum(result["mini_corr"]) / sum(small["mini_corr"]),
        "lt2_corpus_mini_better": int((versus_small < 1).sum()),
        "lt2_parity_max_rel": max(parity),
        "lt2_const_column_scale": float(scale[layer, column]),
        "lt2_shipped_numbers": shipped,
        "lt2_grader_better": sum(x < 1 for x in grader),
        "lt2_grader_ratio_min": min(grader),
        "lt2_grader_ratio_max": max(grader),
    }


def network_six():
    arms = ("active_off", "young_off", "old384", "res16", "fb16", "repr_off", "all_off")
    with np.load(ROOT / "data/mini_truth.npz") as truth_data:
        truth = truth_data["all_layer_means"].astype(np.float64)
    base_prediction = np.load(ABLATION / "b-plain/mini6.npy")
    base = ((base_prediction - truth[6]) ** 2).mean(axis=1)
    relaxed = {arm: ((np.load(ABLATION / f"{arm}-plain/mini6.npy") - truth[6]) ** 2).mean(axis=1)
               for arm in arms}
    others = np.array([((np.load(ABLATION / f"b-plain/mini{i}.npy") - truth[i]) ** 2).mean(axis=1)
                       for i in range(8) if i != 6])
    excess = (base_prediction[14] - truth[6, 14]) ** 2 - (
        np.load(ABLATION / "active_off-plain/mini6.npy")[14] - truth[6, 14]) ** 2
    values = {"abl_net6_b_layers1_14_over_others_max": max(base[i] / others[:, i].max() for i in range(14)),
              "abl_net6_b_layer15_over_others_max": base[14] / others[:, 14].max(),
              "abl_net6_layer15_top20_share": np.sort(excess)[-20:].sum() / excess.sum()}
    for layer in (14, 15):
        values[f"abl_net6_b_layer{layer}"] = base[layer - 1]
        values[f"abl_net6_relaxed_layer{layer}_min"] = min(value[layer - 1] for value in relaxed.values())
        values[f"abl_net6_relaxed_layer{layer}_max"] = max(value[layer - 1] for value in relaxed.values())
    return values


def check_report():
    values = {**corrector(), **network_six()}
    with (ROOT / "data/numbers.csv").open(newline="") as handle:
        report = {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}
    assert len(values) == 36
    for key, actual in values.items():
        assert np.isclose(actual, report[key], rtol=5e-6, atol=1e-12), (key, actual, report[key])
    assert values["lt2_grader_better"] == 50
    assert 5.70e-8 < values["abl_net6_b_layer15"] < 5.72e-8
    return {key: float(value) for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v13_evidence.json")
    args = parser.parse_args()
    values = check_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    print("36 measurements match; 50/50 corrected networks improve; network 6 layer 15 MSE %.5g" %
          values["abl_net6_b_layer15"])


if __name__ == "__main__":
    main()
