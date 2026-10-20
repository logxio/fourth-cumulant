"""Extract the report's main and appendix tables from measurements."""
import argparse
import csv
from pathlib import Path
from submission_pages import readings

ROOT = Path(__file__).resolve().parent
PREFIXES = {
    1: ("k3simple_1024", "k3aug_1024", "k4_cheap_1024", "k4_full_1024", "ratio_k3aug"),
    2: ("width", "nl64_", "depth_", "n_over_L_", "fz_e1_"),
    3: ("abl01_", "abl512_", "fz_e2_", "fz1024_"),
    4: ("lb_s", "grader_", "lt2_", "lad50_", "young_keep_mass_"),
    5: ("abl_plain_", "abl_lt2_", "abl50_", "abl42_", "abl_net6_", "colcut"),
    6: ("frozen8_", "birth_", "k4star_", "s8_", "star_"),
    7: ("late_", "ledger_", "exec_"),
    8: ("pair_", "kdir_", "lean_", "nontree_"),
}
CLAIMS = (
    ("full K=4 at 1024", "1.72e-9", "lab/k4full.py", "k4_full_1024_nets0_4"),
    ("K=3 augmented at 1024", "2.03e-8", "lab/ref1024.py", "k3aug_1024_nets0_4"),
    ("width, depth, n/L", "Figure 1", "lab/kscan.py; lab/depth_s1/ref_depth.py", "nl64_512x8_equiv_bound"),
    ("K=4 ablations", "Table 3", "lab/k4full.py", "abl01_full"),
    ("frozen checks", "Tables 2b, 3", "report/data/frozen_s16/run.sh; research/frozen_s16.py", "fz_"),
    ("frozen at 1024", "Table 3", "lab/k4full.py; research/v19_evidence.py", "fz1024_"),
    ("star weights", "Section 3", "report/data/c45/arms.py; report/data/c45/build.py; research/v21_evidence.py", "c45_"),
    ("cubic form", "1e-18", "lab/k4n3.py", ""),
    ("young feedback", "4.2e-15", "lab/k4n3_young_direct_chain.py", ""),
    ("oracle patches", "Figure 3a", "lab/var_k4_channel.py", "oracle_C_k4"),
    ("late start", "Figure 3b", "lab/cand_k4x/b_suffix.py", "late_start4_net0"),
    ("frozen retest", "Table 6", "lab/cand_k4x/r3/run_one.py", "frozen8_"),
    ("ledger and dense run", "Table 7", "lab/cand_k4x/meter_b_birth.py", "exec_s4_traced_calls"),
    ("non-tree births", "Table 8", "lab/cand_k4x/annealed; lab/cand_k4x/s11; lab/cand_k4x/s12", "kdir_ceiling_best_gain"),
    ("bottleneck reduction", "Tables 9-10; Figure 5", "lab/reduce_s1/select_run.py; lab/reduce_s1/ledger.py; lab/reduce_s1/make_readout.py", "red_points_n"),
    ("scored entries", "Table 4", "data/grader_step_scores.csv; research/v15_evidence.py", "lad50_own_steps_adj_ratio_of_means"),
    ("own pieces switched off", "Table 5", "lab/ablate_s1/eval.py; lab/ablate_s2/eval.py; lab/ablate_s3/eval.py; ablation/run.py --arm col_off; research/v19_evidence.py", "abl50_plain_young_off_adj_ratio_of_means"),
    ("learned corrector", "Table A3", "lab/lt2_train.py; research/v13_evidence.py; research/v21_evidence.py", "lt2_grader_better"),
)

APPENDIX_A1 = (
    ("full K=4", "two dense fourth-order products per source and dense third-order transport", "one cubic kernel per source", "Tables 1-3; Figures 1-2, 3a"),
    ("K=4 subsets", "the same chain with selected groups and crossings removed", "same dense source state", "Tables 1, 3"),
    ("cubic components", "matrix products on source legs and shared bases", "matrix legs and a shared basis", "Sections 2.4-2.6; Table 7"),
    ("compact complete chain", "cubic components plus dense first reads", "legs, bases and a third-order core", "Table 7; Figure 4b"),
    ("late start", "chain B followed by a full K=4 suffix", "dense fourth-order suffix state", "Figure 3b; Table 7"),
    ("chain B", "factorized third-order chain", "5.3 GiB per network", "Tables 4-6"),
    ("non-tree pairs", "rank-weighted cubic matrix reads", "rank-weighted quadratic state per birth", "Table 8"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", choices=["all", *[str(i) for i in range(1, 13)], "4b", "A1", "A2", "A3", "A4"])
    args = parser.parse_args()
    with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = {row["key"]: float(row["value"]) for row in rows}
    readings(values)
    tables = PREFIXES if args.table == "all" else ({int(args.table): PREFIXES[int(args.table)]} if args.table.isdigit() and int(args.table) in PREFIXES else {})
    target = ROOT / "tables"
    target.mkdir(exist_ok=True)
    for number, prefixes in tables.items():
        selected = [row for row in rows if row["key"].startswith(prefixes)]
        if not selected:
            raise RuntimeError(f"table {number} has no data")
        with (target / f"table{number}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        print(f"table {number}: {len(selected)} rows")
    if args.table in ("all", "9"):
        input_effect = {
            "ck3": values["abl01_no_k3c"] / values["abl01_full"],
            "nontree": values["drop_C_kappa3_nontree_net0"] / values["portable_net0"],
            "tree": values["abl01_no_slicex"] / values["abl01_full"],
            "star": f"{values['c45_1024_nostar_min']}-{values['c45_1024_nostar_max']}",
        }
        families = (
            ("C.k3 births", "ck3", "red_gain_ck3", "without / with from layer 4", "without C.k3 / full K4"),
            ("non-tree births", "nontree", "red_gain_nontree", "without / with from layer 4", "one non-tree class removed / portable K4"),
            ("tree slices and feedback", "tree", "red_gain_tree_alone_over_B", "tree alone / chain B", "slice crossings removed / full K4"),
            ("star births, sources and young legs", "star", "red_gain_star", "without / with, C.k3 off", "no star / full K4, frozen mini 5-7 range"),
            ("chain B, metered", "chainb", "", "", ""),
        )
        with (target / "table9.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("state_family", "units", "share", "read_weight_v1_net0", "layer4_ratio", "layer4_definition", "from_input_ratio", "from_input_definition"))
            for label, key, gain_key, note, input_note in families:
                writer.writerow((label, values[f"red_units_{key}"], values[f"red_share_{key}"],
                                 values.get(f"red_rw_v1_n0_{key}", ""), values[gain_key] if gain_key else "", note,
                                 input_effect.get(key, ""), input_note))
            writer.writerow(("total", values["ledger_s4_total"], 1, "", "", "", values["abl01_full"], "full K4 raw MSE, nets 0-1"))
        print("table 9: 5 state families and total")
    if args.table in ("all", "10"):
        with (target / "table10.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("network", "chain_B", "with_exact_3_1", "ratio", "full_K4_over_B", "share_of_K4_log_gain"))
            for net in (0, 1, 3, 4, 5, 6, 7, 8):
                b = values[f"red_k31_B_net{net}"]
                exact = values[f"red_k31_Bk31_net{net}"]
                writer.writerow((net, b, exact, exact / b, values.get(f"red_k31_k4_over_B_net{net}", ""),
                                 values.get(f"red_k31_share_log_net{net}", "")))
            writer.writerow(("eight networks", "", "", values["red_k31_ratio_sum"], "", ""))
        print("table 10: 8 networks and aggregate")
    if args.table in ("all", "4b"):
        recipes = (
            ("#336537", "chain B with carried corrector", "us"),
            ("#335301", "chain B with corrector off", "chainb"),
            ("#335413", "hyojun_kwon's bound entry with learned corrector", "hyojun"),
            ("#334316", "hyojun_kwon's corrector-off base", "hyojun_base"),
            ("#333534", "K=3 rewrite, scale-mixture K=4 closure and corrector", "artyom"),
            ("#336225", "V29 with an all-distinct K=4 edge term and fitted (3,1) law", "nurlan"),
            ("#336317", "carried K=4 core and scale-mode extraction", "miscusi"),
            ("#334821", "V29 with two-term (3,1) closure", "kai"),
            ("#330093", "V29 bound to the 504aldo write-up", "aldo"),
            ("#335813", "V29-route closure with exact cost cuts", "ddteam"),
        )
        with (target / "table4b.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("entry", "recipe", "adjusted_error", "raw_error", "C_over_B",
                             "ours_over_entry_adjusted", "ours_over_entry_raw", "ours_over_entry_C_over_B"))
            for entry, recipe, tag in recipes:
                prefix = f"nb_{tag}_"
                writer.writerow((entry, recipe, values[prefix + "adj"], values[prefix + "raw"],
                                 values[prefix + "cb"],
                                 *("" if tag == "us" else values.get(f"nb_us_over_{tag}_{field}", values[f"nb_us_{field}"] / values[prefix + field])
                                   for field in ("adj", "raw", "cb"))))
        print("table 4b: 10 graded entries")
    if args.table in ("all", "11"):
        points = (
            ("dense rule", "baseline_w9"),
            ("non-tree rank 16", "w9_n16"),
            ("non-tree rank 4", "w9_n4"),
            ("non-tree rank 1", "w9_n1"),
            ("births 4-6, both rank 8", "w6_n8_c8"),
            ("births 4-6, both rank 4", "w6_n4_c4"),
            ("births 4-13, both rank 8", "full_n8_c8"),
            ("births 4-13, both rank 4", "full_n4_c4"),
        )
        with (target / "table11.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("births_held_as_pairs", "units", "without_kappa3_legs", "raw_over_chain_B",
                             "units_over_tie", "deletion_grid_units"))
            for label, key in points:
                stem = f"red2_{key}_"
                writer.writerow((label, values[stem + "units"],
                                 "" if key in ("baseline_w9", "w9_n16", "w9_n4", "w9_n1") else values[stem + "kunits"],
                                 values[stem + "ratio"], values[stem + "gap"],
                                 values.get(stem + "s1_units", "")))
        print("table 11: 8 compression configurations")
    if args.table in ("all", "12"):
        labels = (
            ("chain B to layer 4", "prefix"),
            ("C.k3 and star births, sources, young legs, crossing", "twoleg"),
            ("feedback, old core and young tree", "feedback"),
            ("old slice reads, 26-matmul form", "oldslice"),
            ("non-tree first reads, dense", "nontree"),
            ("outside the count", "outside"),
        )
        with (target / "table12.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("table7a_row_group", "counted", "dense_metered", "p32_metered"))
            for label, key in labels:
                stem = f"e2e_rec_{key}_"
                writer.writerow((label, "" if key == "outside" else values[stem + "counted"],
                                 values[stem + "metered"], values[f"e2e_p32_rec_{key}_metered"]))
            writer.writerow(("total", values["ledger_s4_total"], values["e2e_m1_net0_units"],
                             values["e2e_p32_net0_units"]))
        print("table 12: 6 row groups and total, dense and 32-pair runs")
    if args.table in ("all", "A4"):
        keys = {row["key"] for row in rows}
        for _, _, _, key in CLAIMS:
            if key and not any(k.startswith(key) for k in keys):
                raise RuntimeError(f"table A4 source is missing: {key}")
        with (target / "tableA4.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("claim", "number", "script", "data_key"))
            writer.writerows(CLAIMS)
        print(f"table A4: {len(CLAIMS)} claims")
    if args.table in ("all", "A1"):
        with (target / "tableA1.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("computation", "arithmetic_per_layer", "state", "reported_in"))
            writer.writerows(APPENDIX_A1)
        print(f"table A1: {len(APPENDIX_A1)} rows")
    if args.table in ("all", "A2"):
        selected = [row for row in rows if row["key"].startswith(("knob_", "portable_net", "cross_three2_", "cross_star_", "feedback_", "pair_n64_", "kdir_joint_"))]
        assert len(selected) >= 100
        with (target / "tableA2.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(selected)
        print(f"table A2: {len(selected)} rows")
    if args.table in ("all", "A3"):
        settings = (
            ("chain", "Scored chain B with its learned corrector disabled during trajectory generation", "lt2_grader_ratio"),
            ("networks", f"Public full networks 0-999 and Keenan Pepper corpus networks 0-2047 for training ({int(values['lt2_train_networks'])}); public mini 0-99 chose #335351's configuration, kept unchanged, and are read after each epoch", "lt2_corpus_networks"),
            ("target", f"Per-layer added mean error divided by training RMS: layer 1 {values['lt2_target_rms_layer1']:.4g}, layer 15 {values['lt2_target_rms_layer15']:.4g}", "lt2_target_rms_layer15"),
            ("inputs per neuron", f"35 current-layer, 16 previous-layer, 64 state, 3 correction, 16 layer indicators; total {int(values['lt2_inputs_per_neuron'])}", "lt2_inputs_per_neuron"),
            ("normalization", "Signed log1p transform then per-layer standardization of 51 chain columns; tanh state; correction scaled by layer RMS", "lt2_target_rms_layer1"),
            ("network", f"134 to 128 to 128 to 128 to 33, GELU; {int(values['lt2_trained_parameters'])} trained parameters", "lt2_trained_parameters"),
            ("loss", "Mean squared added-error loss over 16 layers with full recurrent backpropagation", "lt2_epochs"),
            ("optimizer", f"Adam one-cycle peak {values['lt2_peak_lr']:.4g}; {int(values['lt2_batch_networks'])} networks per batch, {int(values['lt2_epochs'])} epochs, {int(values['lt2_steps'])} steps, seed 0", "lt2_steps"),
            ("selection", f"Lowest mini raw ratio at epoch 30, the last; first epoch {values['lt2_holdout_ratio_epoch1']:.4f}, final {values['lt2_holdout_ratio']:.4f}; #335351's corrector {values['lt2_small_holdout_ratio']:.4f}", "lt2_holdout_ratio_epoch1"),
            ("shipped", f"{int(values['lt2_shipped_numbers'])} embedded numbers, bitwise equal to the archived weight export; NumPy against PyTorch {values['lt2_parity_max_rel']:.3g} of the largest correction", "lt2_shipped_numbers"),
            ("cost", f"Training {int(values['lt2_train_wall_s'])} seconds on one A40; {values['lt2_run_hours']:.1f} h from launch to the embedded file; archived run log and weights supplied", "lt2_train_wall_s"),
            ("result", f"Mini improvement on {int(values['lt2_holdout_better'])}/100; grader improvement on {int(values['lt2_grader_better'])}/50; against #335351 adj {values['lad50_corpus_adj_ratio_of_means']:.4f}, lower on {int(values['lad50_corpus_better'])}/50; #335351's corrector retrained with seeds 0, 1, 2: mini {values['lt2_seed0_holdout_ratio']:.4f}, {values['lt2_seed1_holdout_ratio']:.4f}, {values['lt2_seed2_holdout_ratio']:.4f}, lower on {int(values['lt2_seed0_holdout_better'])}, {int(values['lt2_seed1_holdout_better'])}, {int(values['lt2_seed2_holdout_better'])} of 100", "lt2_grader_better"),
        )
        with (target / "tableA3.csv").open("w", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(("item", "setting", "evidence_key"))
            writer.writerows(settings)
        print(f"table A3: {len(settings)} rows")


if __name__ == "__main__":
    main()
