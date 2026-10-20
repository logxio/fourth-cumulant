"""Rebuild the report's seven figures with the report's own drawing code (report/hero_figure.py, visual_figures.py, mech_figure.py, cel.py, palette.py,
figcheck.py) from released measurements: data/numbers.csv, data/figure1_layers.csv (per-layer full K=4),
data/ref_depth.csv (per-layer augmented K=3), data/figure1_nl64.csv (per-network errors of the n/L = 64 shapes),
report/data/layer_noise.json (truth noise), runs/kscan_128_16.log (the 128 x 16 order scan) and
data/grader_step_scores.csv (the 50 public grader networks of Table 4's eight steps)."""
import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1792195200")  # fixed PDF date (17 Oct 2026): a rebuild is byte for byte
import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "report"))
import visual_figures  # noqa: E402
import hero_figure  # noqa: E402
import mech_figure  # noqa: E402
from reduce_fig5 import inputs as reduce_fig5_inputs  # noqa: E402
from research.v15_evidence import check_report  # noqa: E402
from submission_pages import readings  # noqa: E402
from metered_readings import check_p32  # noqa: E402

OUT = ROOT / "figures"
LEDGER_ROWS = {  # row name in the report's ledger -> its numbers.csv key
    "chain B (metered)": "chain_B_metered",
    "K=4 sources, kappa3 legs n wide": "K_4_sources_kappa3_legs_n_wide",
    "young-leg transport": "young_leg_transport",
    "feedback": "feedback",
    "old slice reads": "old_slice_reads",
    "C.k3 births at real kappa3 width, beyond n-wide legs": "C_k3_births_at_real_kappa3_width_beyond_n_wide_legs",
    "C.k3 crossing at rank 16": "C_k3_crossing_at_rank_16",
    "non-tree first reads, dense, classes merged": "non_tree_first_reads_dense_classes_merged",
}


def namespace():
    with (ROOT / "data" / "numbers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    N = {row["key"]: float(row["value"]) for row in rows}
    if len(N) != 1935:
        raise RuntimeError(f"final report measurements are required, found {len(N)}")
    readings(N)
    check_p32(ROOT, N)
    D = {"ROWS": [{"key": k, "value": v} for k, v in N.items()]}
    widths = [256, 384, 512, 1024]
    seeds = {256: (11, 12, 13), 384: (21, 22, 23), 512: (41, 42, 43), 1024: (0, 1, 2, 3, 4)}
    D["WSET"] = widths
    D["wstat"] = {(w, k): (N[f"width{w}_{k}_mean"], N[f"width{w}_{k}_ci_lo"], N[f"width{w}_{k}_ci_hi"])
                  for w in widths for k in ("k3a", "k4")}
    D["wnet"] = {w: [(s, {k: (N[f"width{w}_seed{s}_{k}"],) for k in ("k3a", "k4")}) for s in seeds[w]] for w in widths}
    D["alpha"] = {k: (N[f"width_exponent_{k}"], N[f"width_exponent_{k}_ci_lo"], N[f"width_exponent_{k}_ci_hi"])
                  for k in ("k3a", "k4")}
    with (ROOT / "data" / "figure1_nl64.csv").open(newline="") as handle:
        nl = list(csv.DictReader(handle))
    D["NLPER"] = {key: np.array([[float(r["k3_direct"]), float(r["k4_direct"])] for r in nl if r["shape"] == key])
                  for key in ("256x4", "512x8")}
    ks = []
    for line in (ROOT / "runs" / "kscan_128_16.log").read_text().splitlines():
        m = re.match(r"(k3a|k4s): final MSE ([0-9.e+-]+)", line)
        if m and m.group(1) == "k3a":
            ks.append({"k3a": float(m.group(2))})
        elif m:
            ks[-1]["k4s"] = float(m.group(2))
    D["_ks128"] = ks
    with (ROOT / "data" / "figure1_layers.csv").open(newline="") as handle:
        layers = {(r["variant"], int(r["network"]), int(r["layer"])): float(r["mse"]) for r in csv.DictReader(handle)}
    D["k4w"] = {("full", i): (N[f"width1024_seed{i}_k4"], np.array([layers["k4_full", i, l] for l in range(1, 17)]))
                for i in range(5)}
    with (ROOT / "data" / "ref_depth.csv").open(newline="") as handle:
        ref = {(int(r["net"]), int(r["layer"])): float(r["mse_ref_k3a"]) for r in csv.DictReader(handle)}
    D["ref3_layers"] = {i: np.array([ref[i, l] for l in range(1, 17)]) for i in range(5)}
    noise = json.loads((ROOT / "report" / "data" / "layer_noise.json").read_text())
    D["noise"] = {int(k): np.array(v["noise_1e9"]) for k, v in noise.items()}
    D["FZ2"] = {(n, s, arm): N[f"fz_e2_{n}_s{s}_{arm}"] for n, ss in ((128, (3, 4, 5)), (192, (601, 602, 603)))
                for s in ss for arm in ("full", "no_k3c", "no_path", "no_slicex", "k3a")}
    D["abl01"] = {k.replace("abl01_", ""): v for k, v in N.items() if k.startswith("abl01_")}
    D["RV"] = {(i, arm): N[f"fz1024_mini{i}_{arm}"] for i in (5, 6, 7) for arm in ("full", "no_k3c", "no_path", "no_slicex")}
    D["late"] = {"B": N["late_B_alone_net0"], 4: N["late_start4_net0"], 6: N["late_start6_net0"],
                 8: N["late_start8_net0"], "empty8": N["late_start8_empty_net0"]}
    with (ROOT / "data" / "grader_step_scores.csv").open(newline="") as handle:
        grader = list(csv.DictReader(handle))
    D["_lad"] = {}
    for sid in (332957, 334990, 335127, 335162, 335185, 335300, 335301, 336537):
        rows_s = sorted((int(r["network"]), float(r["adjusted_final_layer_score"])) for r in grader
                        if int(r["submission"]) == sid)
        assert len(rows_s) == 50
        D["_lad"][sid] = np.array([v for _, v in rows_s])
    D["LAD"] = {"own": {"rom": N["lad50_own_steps_adj_ratio_of_means"],
                        "ci": (N["lad50_own_steps_ci_lo"], N["lad50_own_steps_ci_hi"]),
                        "better": int(N["lad50_own_steps_better"])}}
    D["LEDGER_FULL"] = {}
    for s0 in (0, 4, 6, 8):
        Lg = {"rows": {name: N[f"ledger_s{s0}_{key}"] for name, key in LEDGER_ROWS.items()},
              "total": N[f"ledger_s{s0}_total"], "gain_share": N[f"ledger_s{s0}_gain_share"],
              "raw": N["portable_net0"] if s0 == 0 else N[f"late_start{s0}_net0"], "tie": N[f"ledger_s{s0}_tie_units"]}
        if s0 in (0, 4):
            Lg["raw_c"] = N["young_m0f0_k128_drop_triangle"] if s0 == 0 else N["exec_s4_raw"]
            Lg["tie_c"] = N[f"ledger_s{s0}_tie_units_compressed"]
        D["LEDGER_FULL"][s0] = Lg
    # chain B's metered units on network 0, from the tie of an exact-raw column: tie = raw_B * units_B / raw
    D["B0_UNITS"] = N["ledger_s6_tie_units"] * N["late_start6_net0"] / N["late_B_alone_net0"]
    D["EXEC"] = {"total": N["exec_s4_total_units"]}
    D["RED_FIG"] = reduce_fig5_inputs(ROOT, N)
    def sci(value):
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    D["RED_FIG"]["caps"] = {
        "m1_net0": ("dense kernels", f"{sci(N['e2e_m1_net0_raw'])} at {N['e2e_m1_net0_units']:,.0f}"),
        "p32_net0": ("32 pairs", f"{sci(N['e2e_p32_net0_raw'])} at {N['e2e_p32_net0_units']:,.0f} "
                       f"(−{1 - N['e2e_p32_over_m1_units']:.0%})"),
        "p8_net0": ("8 pairs", f"{sci(N['e2e_p8_net0_raw'])} at {N['e2e_p8_net0_units']:,.0f}"),
    }
    assert D["RED_FIG"]["caps"] == {
        "m1_net0": ("dense kernels", "6.37e-9 at 756,620"),
        "p32_net0": ("32 pairs", "6.95e-9 at 525,045 (−31%)"),
        "p8_net0": ("8 pairs", "8.34e-9 at 512,138"),
    }
    D["RED_FIG"]["count_ratio"] = N["e2e_m1_over_counted"]
    assert f"{D['RED_FIG']['count_ratio']:.1f}" == "12.5"
    return D, N


def main():
    check_report()
    D, N = namespace()
    OUT.mkdir(exist_ok=True)
    visual_figures.render_all(D, OUT)
    hero_figure.render(hero_figure.KEEP, OUT, N)
    mech_figure.render(OUT)
    print("fig_mech: two propagation tracks, figures rebuilt from 1,935 released readings")
    print(f"fig0_hero_D: {int(N['lb1018_listed_n'])} listed entries, "
          f"{int(N['lb1018_top50_n'])} at ranks 1-50; rank-1 raw / K=4 {N['lb1018_rank1_raw_over_k4']:.5g}x")
    print(f"fig1_width_and_depth: width 1024 ratio {N['width1024_ratio']:.6g}x; n/L exponents "
          f"{N['n_over_L_exponent_width']:.2f} and {N['n_over_L_exponent_depth']:.3g}; shape-equivalence bounds "
          f"{N['nl64_512x8_equiv_bound']:.6g} and {N['nl64_256x4_equiv_bound']:.6g}")
    print(f"fig2_what_must_be_kept: cheap subset {N['abl01_cheap'] / N['abl01_full']:.5g}x full K=4 at width 1024")
    print(f"fig3_whole_chain_late_start: layer 4 raw {N['late_start4_net0']:.6g}")
    own = D["LAD"]["own"]
    print(f"fig4_scored_and_cost: our steps {own['rom']:.6g}x; layer-4 compact K=4 {N['ledger_s4_total']:,.0f} units, "
          f"ties chain B at {N['ledger_s4_tie_units_compressed']:,.0f}")
    print(f"fig5_where_602_is_bound: {sum(p['net'] == 0 for p in D['RED_FIG']['pts'])} network-0 points; "
          f"C.k3 and non-tree {N['red_share_ck3'] + N['red_share_nontree']:.1%} of counted units")


if __name__ == "__main__":
    main()
