"""Check the submitted estimator (#336537) and final report measurements."""
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from submission_pages import ENTRIES, readings
from metered_readings import check_p32

ROOT = Path(__file__).resolve().parent
if sys.argv[1:] == ["--laptop"]:
    run = subprocess.run([sys.executable, str(ROOT / "laptop_exact.py")],
                         capture_output=True, text=True, timeout=90, check=True)
    pattern = re.compile(r"^seed ([345]): K4 raw=([0-9.e+-]+) units=([0-9.]+); "
                         r"K3a raw=([0-9.e+-]+) units=([0-9.]+); "
                         r"K3a/K4=([0-9.]+)x; CPU=([0-9.]+)s RSS=([0-9.]+)MiB$",
                         re.MULTILINE)
    found = {int(seed): (float(k4), float(u4), float(k3), float(u3), float(ratio))
             for seed, k4, u4, k3, u3, ratio, _, _ in pattern.findall(run.stdout)}
    expected = {
        3: (1.223533273e-5, 5.100159135e-5),
        4: (3.129622230e-5, 6.265472083e-5),
        5: (1.604079576e-5, 5.099867608e-5),
    }
    assert set(found) == set(expected), run.stdout
    for seed, (want4, want3) in expected.items():
        got4, u4, got3, u3, ratio = found[seed]
        assert abs(got4 - want4) <= 2e-6 * want4, (seed, got4)
        assert abs(got3 - want3) <= 2e-6 * want3, (seed, got3)
        assert abs(u4 - 366.679227) < 0.0001, (seed, u4)
        assert abs(u3 - 26.763125) < 0.0001, (seed, u3)
        assert ratio > 1
    print(run.stdout, end="")
    print("Three frozen networks, live exact K=4 and augmented K=3, truth hashes, "
          "raw MSEs and flopscope charges match.")
    raise SystemExit(0)
if not sys.argv[1:]:
    from PIL import Image
    source = (ROOT / "estimator.py").read_bytes()
    assert hashlib.md5(source).hexdigest() == "23b96f59dd1d6b9e9eb1f0b07a0cd6a4"
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert manifest["files"][0]["sha256"] == hashlib.sha256(source).hexdigest()
    saved = (ROOT / "data/numbers.csv").read_bytes()
    assert hashlib.sha256(saved).hexdigest() == "a9087913520ab60d8da36c7f0bf81913cb02360dd04102c27d94f71cae4d121a"
    with (ROOT / "data/numbers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1935 and len({row["key"] for row in rows}) == len(rows)
    numbers = {row["key"]: float(row["value"]) for row in rows}
    assert all(math.isfinite(value) for value in numbers.values())
    readings(numbers)
    check_p32(ROOT, numbers)
    with (ROOT / "tables/table4b.csv").open(newline="") as handle:
        entries = list(csv.DictReader(handle))
    assert len(entries) == 10
    for row in entries:
        sid = row["entry"].lstrip("#")
        tag = "hyojun_base" if sid == "334316" else next(tag for tag, number, _ in ENTRIES if number == sid)
        assert float(row["adjusted_error"]) == numbers[f"nb_{tag}_adj"]
        assert float(row["raw_error"]) == numbers[f"nb_{tag}_raw"]
        assert float(row["C_over_B"]) == numbers[f"nb_{tag}_cb"]
    figure_hashes = {
        "fig0_hero_D": {
            ((4086, 2424), "cb4bdeeb2214a9b7e76cf80fac7408395eaf4ea5e70f6d9c28afc16f6284b7d1"),
            ((4085, 2424), "e47e7e5760df9b8798c08a43d1c26fcef1b86c81089ae4f174bef91ca76e53fb"),
        },
        "fig5_where_602_is_bound": {
            ((4086, 2166), "7427511f04c863072e3d11d6a46b7a6836f7d5e2c387f9e5c62872af63d0871a"),
            ((4085, 2166), "00ad446d9cfe02cb9023cda093c714b5dfe193b2e4a8b745e44649fcee586023"),
        },
        "fig_mech": {
            ((4086, 1764), "d8a9f4dd127031d11b43a41a63df7c69759c452667db90a3b017a45bcb97dbe7"),
            ((4085, 1764), "eea224167dfbbcff120e894c6c674f33feddbe3053cd07777c891b15c7c74e62"),
        },
    }
    for name, expected in figure_hashes.items():
        with Image.open(ROOT / "figures" / f"{name}.png") as figure:
            rgba = figure.convert("RGBA")
            actual = (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest())
            assert actual in expected, (name, actual)
    print("Estimator hash, 1,935 frozen numeric readings, ten saved official pages, derived write-up comparisons, leaderboard cross-checks, Table 4b, and Figures 0, 5 and M match.")
    raise SystemExit(0)
if sys.argv[1:] != ["--full"]:
    raise SystemExit("usage: verify.py [--full|--laptop]")

import numpy as np
from PIL import Image
from research.v13_evidence import check_report
from research.v15_evidence import check_report as check_v15
from research.frozen_s16 import check as check_frozen
from research.v19_evidence import check as check_v19
from research.v21_evidence import check as check_v21
from report.reduce_fig5 import inputs as reduce_fig5_inputs
from report.hero_figure import board as hero_board, rows as hero_rows

ROOT = Path(__file__).resolve().parent
source = (ROOT / "estimator.py").read_bytes()
assert hashlib.md5(source).hexdigest() == "23b96f59dd1d6b9e9eb1f0b07a0cd6a4"
manifest = json.loads((ROOT / "manifest.json").read_text())
assert manifest["files"][0]["sha256"] == hashlib.sha256(source).hexdigest()
with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
    numbers = {row["key"]: float(row["value"]) for row in csv.DictReader(handle)}
assert len(numbers) == 1935
readings(numbers)
check_p32(ROOT, numbers)
hero_entries = hero_board()
assert len(hero_entries) == 51
for key, value, *_ in hero_rows(numbers, hero_entries):
    assert abs(value - numbers[key]) <= 5e-6 * max(abs(value), abs(numbers[key]), 1e-12), key
reduction = reduce_fig5_inputs(ROOT, numbers)
assert len(reduction["pts"]) == 83
assert len(reduction["pts2"]) == 8
assert {p["label"] for p in reduction["pts2"] if p["frontier"] and not p["control"]} == {"w9_n1", "w9_n4", "w9_n16"}
assert abs(reduction["e2e"]["m1_net0"]["units"] - numbers["e2e_m1_net0_units"]) < 1
assert abs(reduction["e2e"]["p8_net0"]["units"] - numbers["e2e_p8_net0_units"]) < 1
assert abs(numbers["red2_w9_n4_units"] - 3258.09) < 0.01
assert numbers["e2e_m1_family_rows_metered"] == 538045
grader_public = json.loads((ROOT / "report/data/res_336537.json").read_text())["aggregates"]["public"]
assert grader_public["adjusted_final_layer_score"] == numbers["nb_us_adj"]
assert grader_public["final_layer_mse"] == numbers["nb_us_raw"]
with (ROOT / "data/leaderboard_1017_2105.csv").open(newline="") as handle:
    board = {row["submission"]: row for row in csv.DictReader(handle)}
assert set(board) == {"333534", "336225", "334821", "335813"}
for submission, tag in (("333534", "artyom"), ("336225", "nurlan"), ("334821", "kai"), ("335813", "ddteam")):
    assert abs(float(board[submission]["adjusted_error"]) / numbers[f"nb_{tag}_adj"] - 1) < 0.02
    assert abs(float(board[submission]["raw_error"]) / numbers[f"nb_{tag}_raw"] - 1) < 0.02
forum_excerpt = (ROOT / "data/forum_18252_excerpt.txt").read_text()
posted = re.search(r"#336225: adjusted ([0-9.e-]+), final-layer MSE ([0-9.e-]+), q ([0-9.e-]+)", forum_excerpt)
assert posted
for key, value in zip(("adj", "raw", "cb"), posted.groups()):
    assert abs(float(value.rstrip(".")) / numbers[f"nb_nurlan_{key}"] - 1) < 0.01
assert abs(numbers["red_k31_ratio_sum"] - 0.872532) < 1e-6
assert numbers["red_k31_wins"] == 7
assert abs(numbers["red_units_ck3"] + numbers["red_units_nontree"] - 58095) < 1
assert abs(reduction["rw"]["v1_n0"]["nontree"] - numbers["red_rw_v1_n0_nontree"]) < 1e-6
frozen_checks = check_frozen()
assert frozen_checks["values"] == 122
assert (frozen_checks["C1"], frozen_checks["C2"], frozen_checks["C3"]) == (0, 0, 1)
v13 = check_report()
v15 = check_v15()
assert len(v15) == 102
v19 = check_v19()
assert len(v19) == 53 and v19["fz1024_order_holds"] == 3 and v19["fz1024_networks_run"] == 3
v21 = check_v21()
assert len(v21) == 36 and v21["c45_small_wstar_above1"] == 12 and v21["c45_small_loo_wins"] == 12
assert (v21["c45_1024_pred1_holds"], v21["c45_1024_pred2_holds"], v21["c45_1024_pred3_holds"]) == (3, 3, 0)
assert all(v21[f"lt2_seed{s}_holdout_better"] == 100 for s in (0, 1, 2)) and v21["lt2_seed_spread"] < 1e-3
assert abs(numbers["colcut50_adj_ratio_of_means"] - 1.03461) < 1e-5 and numbers["colcut50_worse"] == 50
assert v13["lt2_grader_better"] == 50
assert abs(v13["abl_net6_b_layer15"] - 5.71e-8) < 1e-10
assert abs(numbers["k4_full_1024_nets0_4"] - 1.7168e-9) < 1e-15
assert abs(numbers["ratio_k3aug_over_k4_1024"] - 11.8009) < 1e-4
assert abs(numbers["lb_s24_adj"] - 3.145e-9) < 1e-15
assert abs(numbers["lb_s21_adj"] - 3.181e-9) < 1e-15
assert abs(numbers["lad50_step8_adj_ratio_of_means"] - 0.900185) < 1e-5 and numbers["lad50_step8_better"] == 49
assert abs(numbers["lad50_corpus_adj_ratio_of_means"] - 0.988764) < 1e-5 and numbers["lad50_corpus_ci_hi"] < 1
assert v13["lt2_train_networks"] == 3048 and v13["lt2_corpus_networks"] == 2048
assert abs(numbers["abl_plain_all_off_adj_mean"] - 1.1854352360452094) < 1e-5
assert abs(numbers["abl_lt2_all_off_adj_mean"] - 4.155512140635324) < 1e-5
assert abs(numbers["abl_plain_all_off_adj_median"] - 1.29367) < 1e-5
assert abs(numbers["abl_plain_all_off_adj_median_ci_lo"] - 1.17898) < 1e-5
assert abs(numbers["abl_plain_all_off_adj_median_ci_hi"] - 1.40113) < 1e-5
assert abs(numbers["abl_lt2_all_off_adj_median"] - 3.94076) < 1e-5
assert abs(numbers["abl_lt2_all_off_adj_median_ci_lo"] - 3.50804) < 1e-5
assert abs(numbers["abl_lt2_all_off_adj_median_ci_hi"] - 5.29282) < 1e-5
for arm, expected, lo, hi in (("all_off", 1.2782, 1.1555, 1.3549),
                              ("active_off", 1.1600, 1.0527, 1.2267)):
    stem = f"abl50_plain_{arm}_adj_ratio_of_means"
    assert abs(numbers[stem] - expected) < 6e-5
    assert abs(numbers[stem + "_ci_lo"] - lo) < 5e-5
    assert abs(numbers[stem + "_ci_hi"] - hi) < 5e-5
    assert numbers[f"abl50_plain_{arm}_worse"] == 49
assert abs(numbers["n_over_L_exponent_depth"] - 0.754) < 1e-3
assert abs(numbers["nl64_512x8_equiv_bound"] - 1.09412) < 1e-5
assert abs(numbers["nl64_256x4_equiv_bound"] - 1.32992) < 1e-5
assert v15["nl64_512x8_inside_080_125"] == 1
assert v15["nl64_256x4_inside_080_125"] == 0
for arm, ratio, worse in (("young_off", 1.044034, 49), ("repr_off", 1.049365, 48)):
    stem = f"abl50_plain_{arm}"
    assert abs(numbers[stem + "_adj_ratio_of_means"] - ratio) < 1e-5
    assert numbers[stem + "_worse"] == worse
assert abs(numbers["abl_net6_new_b_layer15"] - 1.853712458e-8) < 1e-12
assert numbers["abl_net6_same_machine_trace_max_abs"] == 0
assert abs(numbers["lad50_own_steps_adj_ratio_of_means"] - 0.829632) < 1e-5
with (ROOT / "data" / "figure1_layers.csv").open(newline="") as handle:
    layers = list(csv.DictReader(handle))
assert len(layers) == 240
assert len({(row["variant"], row["network"], row["layer"]) for row in layers}) == 240
with (ROOT / "data/ref_depth.csv").open(newline="") as handle:
    ref_depth = list(csv.DictReader(handle))
assert len(ref_depth) == 80
for row in ref_depth:
    match = next(value for value in layers if value["variant"] == "k3_reference"
                 and value["network"] == row["net"] and value["layer"] == row["layer"])
    assert float(match["mse"]) == float(row["mse_ref_k3a"])
assert numbers["abl_plain_all_off_worse"] == 7
assert numbers["abl_lt2_all_off_worse"] == 8
ceiling = [numbers[f"kdir_ceiling_k{k}_net{net}_raw_ratio"] for k in (1, 2, 4, 8) for net in (0, 1)]
assert min(ceiling) < 0.982 and max(ceiling) > 1.030
assert abs(numbers["kdir_ceiling_best_gain"] - 0.0181567) < 1e-7
assert abs(numbers["kdir_joint_r4_birth_units"] - 9.08842) < 1e-5
freeze = json.loads((ROOT / "data/frozen_r3/freeze.json").read_text())
for name, expected_sha in freeze["source_sha256"].items():
    assert hashlib.sha256((ROOT / "lab/cand_k4x/r3/frozen" / name).read_bytes()).hexdigest() == expected_sha
frozen_source = (ROOT / "lab/cand_k4x/r3/frozen/estimator.py").read_text()
frozen_ast_sha = hashlib.sha256(ast.dump(ast.parse(frozen_source), include_attributes=False).encode()).hexdigest()
assert frozen_ast_sha == "4b44bd9881da92d8d5dfd40f769f49c52584bdd898c664bf81e46b20498d4fb9"
target = next(ast.literal_eval(node.value) for node in ast.parse(source).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "SSC_K" for t in node.targets))
fitted = json.loads((ROOT / "lab" / "cand_ssc" / "fitc" / "r1_deltas.json").read_text())
reproduced = json.loads((ROOT / "outputs" / "fit_deltas.json").read_text())
fitted_config = json.loads((ROOT / "lab" / "cand_ssc" / "fitc" / "r1_s1.0.json").read_text())
for key, delta in fitted.items():
    assert abs(reproduced[key] - delta) < 1e-12
    baseline = 1.0 if key.startswith("lam_") else 0.0
    assert target[key][0] == baseline
    assert all(abs(value - (baseline + delta)) < 5e-7 for value in target[key][1:])
    assert all(abs(value - source_value) < 5e-7 for value, source_value in zip(target[key], fitted_config[key]))
with (ROOT / "outputs" / "ablation_summary.csv").open(newline="") as handle:
    summary = {(row["arm"], row["learner"]): row for row in csv.DictReader(handle)}
assert abs(float(summary["all_off", "plain"]["paired_adj_ratio"]) - 1.1854352360452094) < 1e-8
assert abs(float(summary["all_off", "lt2"]["paired_adj_ratio"]) - 4.155512140635324) < 1e-8
saved = {(row["cohort"], row["arm"]): row for row in
         json.loads((ROOT / "data/ablate_s2/summary.json").read_text())}
recomputed = {(row["cohort"], row["arm"]): row for row in
              json.loads((ROOT / "outputs/ablation50_summary.json").read_text())}
assert saved.keys() <= recomputed.keys()
for key in saved:
    for scope in ("all", "without_raw_max_base"):
        a, b = saved[key][scope], recomputed[key][scope]
        for field in ("ratio_of_means", "mean_network_ratio", "median_network_ratio"):
            assert abs(a[field] - b[field]) < 1e-12
            assert all(abs(x-y) < 1e-12 for x, y in zip(a[field + "_ci95"], b[field + "_ci95"]))
        assert a["worse_count"] == b["worse_count"]
assert abs(recomputed["pooled_0_49", "all_off"]["all"]["ratio_of_means"] - 1.2782) < 6e-5
assert abs(recomputed["pooled_0_49", "active_off"]["all"]["ratio_of_means"] - 1.1600) < 5e-5
saved_s3 = {row["arm"]: row for row in json.loads((ROOT / "data/ablate_s3/summary.json").read_text())
            if row["learner"] == "plain"}
for arm in ("young_off", "repr_off"):
    actual = recomputed["pooled_0_49", arm]["all"]
    expected = saved_s3[arm]
    assert abs(actual["ratio_of_means"] - expected["ratio_of_means"]) < 1e-12
    assert all(abs(a - b) < 1e-12 for a, b in zip(actual["ratio_of_means_ci95"], expected["ratio_of_means_ci95"]))
    assert actual["worse_count"] == expected["worse_count"]
with (ROOT / "data/ablate_s2/all_rows.csv").open(newline="") as handle:
    ablation_rows = list(csv.DictReader(handle))
assert len(ablation_rows) == 150
assert set(ablation_rows[0]) == {"batch", "arm", "index", "raw", "cb", "adj", "peak_rss_gib", "wall_s"}
with np.load(ROOT / "data/mini_truth.npz") as truth_file:
    truth = truth_file["all_layer_means"]
    for arm in ("b", "active_off", "all_off"):
        for index in range(8, 50):
            stem = ROOT / "data/ablate_s2" / f"{arm}-plain" / f"mini{index}"
            pred = np.load(stem.with_suffix(".npy"))
            record = json.loads(stem.with_suffix(".json").read_text())
            assert pred.shape == (16, 1024) and record["index"] == index and record["arm"] == arm
            mse = float(np.mean((pred[-1].astype(np.float64) - truth[index, -1]) ** 2))
            assert abs(mse / record["raw"] - 1) < 1e-6
for figure in range(0, 6):
    assert list((ROOT / "figures").glob(f"fig{figure}_*.pdf"))
assert (ROOT / "figures/fig_mech.pdf").is_file()
with Image.open(ROOT / "figures/fig0_hero_D.png") as figure_zero:
    rgba = figure_zero.convert("RGBA")
    assert (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest()) in {
        ((4086, 2424), "cb4bdeeb2214a9b7e76cf80fac7408395eaf4ea5e70f6d9c28afc16f6284b7d1"),
        ((4085, 2424), "e47e7e5760df9b8798c08a43d1c26fcef1b86c81089ae4f174bef91ca76e53fb"),
    }
with Image.open(ROOT / "figures/fig5_where_602_is_bound.png") as figure_five:
    rgba = figure_five.convert("RGBA")
    assert (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest()) in {
        ((4086, 2166), "7427511f04c863072e3d11d6a46b7a6836f7d5e2c387f9e5c62872af63d0871a"),
        ((4085, 2166), "00ad446d9cfe02cb9023cda093c714b5dfe193b2e4a8b745e44649fcee586023"),
    }
with Image.open(ROOT / "figures/fig_mech.png") as figure_mech:
    rgba = figure_mech.convert("RGBA")
    assert (rgba.size, hashlib.sha256(rgba.tobytes()).hexdigest()) in {
        ((4086, 1764), "d8a9f4dd127031d11b43a41a63df7c69759c452667db90a3b017a45bcb97dbe7"),
        ((4085, 1764), "eea224167dfbbcff120e894c6c674f33feddbe3053cd07777c891b15c7c74e62"),
    }
for table in range(1, 11):
    assert (ROOT / "tables" / f"table{table}.csv").is_file()
for table in ("4b", "11", "12"):
    assert (ROOT / "tables" / f"table{table}.csv").is_file()
for table in ("A1", "A2", "A3", "A4"):
    assert (ROOT / "tables" / f"table{table}.csv").is_file()
with (ROOT / "tables" / "tableA2.csv").open(newline="") as handle:
    assert len(list(csv.DictReader(handle))) == 165
with (ROOT / "tables" / "tableA3.csv").open(newline="") as handle:
    assert len(list(csv.DictReader(handle))) == 12
with (ROOT / "tables" / "tableA4.csv").open(newline="") as handle:
    assert len(list(csv.DictReader(handle))) == 18
with (ROOT / "tables" / "table4b.csv").open(newline="") as handle:
    assert len(list(csv.DictReader(handle))) == 10
print("Estimator hash (#336537), SSC coefficients, 1935 measurements, 122 frozen checks, the frozen 1024 birth order on networks 5-7, the star-birth weights and their three 1024 predictions, three corrector seeds, the column-cut panel, 13 main tables, four appendix tables, seven figures, four 50-network paired ablations, grader steps, corrector protocol, and independent network-six traces match.")
