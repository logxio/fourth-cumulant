"""Dual-track mechanism diagram before section 2 of the report:
  Track 1 (Regime 3, full-scale factored K=4, sections 2 and 5, role 'k4' blue)
  Track 2 (Regime 2, in-budget K=3 chain B, section 4, role 'chainb' iris)
sharing the common input "ARC's K=3 / K=4 cumulant recursion (Wu et al. 2026)".

Three variants with distinct main visual forms:
  A  Left-right parallel dual-track pipeline (two side-by-side cel columns fed by a top recursion header)
  B  Top-bottom shared layer-axis diagram (two horizontal propagation rails across layers b -> l -> L=16)
  C  Central-spine diverging blueprint (vertical 16-layer recursion spine in the center radiating left to Regime 3
     and right to Regime 2, with large tensor/pruning specimens and comparative rulers)

Run from the repository root with `python3 make_figures.py`.
"""
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cel  # noqa: E402
import figcheck  # noqa: E402
from palette import HUES, INK, MUTED  # noqa: E402
from visual_figures import FS, FSS, W, canvas, setup  # noqa: E402

NUMBERS = HERE.parent / "data" / "numbers.csv"
X = r"$\times$"


def load_numbers(path=NUMBERS):
    with open(path) as f:
        return {r["key"]: float(r["value"]) for r in csv.DictReader(f)}


def verified_strings(N=None):
    """Read and assert every number shown on the mechanism figure against numbers.csv."""
    N = N or load_numbers()

    # Regime 3 (Track 1: full-scale factored K=4, sections 2 and 5)
    k4_mse = N["k4_full_1024_nets0_4"]
    k3a_mse = N["k3aug_1024_nets0_4"]
    ratio_1024 = N["ratio_k3aug_over_k4_1024"]
    exp_k4 = N["width_exponent_k4"]
    exp_k3a = N["width_exponent_k3a"]
    p32_mse = N["e2e_p32_net0_raw"]
    p32_units = N["e2e_p32_net0_units"]
    m1_units = N["e2e_m1_net0_units"]
    p32_ratio = N["e2e_p32_over_m1_units"]
    p32_cut_pct = (1.0 - p32_ratio) * 100.0

    # Regime 2 (Track 2: in-budget K=3 chain B, section 4)
    v29_adj = N["nb_aldo_adj"]
    young_keep = N["young_keep_mass_335301"] * 100.0
    young_keep_final = N["young_keep_mass_336537"] * 100.0
    s335301_adj = N["nb_chainb_adj"]
    s335301_rank = int(round(N["nb_chainb_rank_no_corrector_no_table"]))
    writeups_n = int(round(N["nb_writeups_n"]))
    lt2_params = int(round(N["lt2_trained_parameters"]))
    s336537_adj = N["nb_us_adj"]
    s336537_rank = int(round(N["nb_us_rank_of_writeups"]))
    below_next_pct = N["nb_us_below_next_adj"] * 100.0

    s = {
        "k4_mse": f"{k4_mse:.2e}".replace("e-0", "e-"),
        "k3a_mse": f"{k3a_mse:.2e}".replace("e-0", "e-"),
        "ratio_1024": f"{ratio_1024:.1f}",
        "exp_k4": f"{exp_k4:.2f}",
        "exp_k3a": f"{exp_k3a:.2f}",
        "p32_mse": f"{p32_mse:.2e}".replace("e-0", "e-"),
        "p32_units": f"{p32_units:,.0f}",
        "m1_units": f"{m1_units:,.0f}",
        "p32_cut_pct": f"{p32_cut_pct:.0f}%",
        "v29_adj": f"{v29_adj:.3e}".replace("e-0", "e-"),
        "young_keep": f"{young_keep:.1f}%",
        "s335301_adj": f"{s335301_adj:.3e}".replace("e-0", "e-"),
        "s335301_rank": f"{s335301_rank}",
        "writeups_n": f"{writeups_n}",
        "lt2_params": f"{lt2_params:,d}",
        "s336537_adj": f"{s336537_adj:.3e}".replace("e-0", "e-"),
        "s336537_rank": f"{s336537_rank}",
        "below_next_pct": f"{below_next_pct:.1f}%",
    }

    assert s["k4_mse"] == "1.72e-9", s["k4_mse"]
    assert s["k3a_mse"] == "2.03e-8", s["k3a_mse"]
    assert s["ratio_1024"] == "11.8", s["ratio_1024"]
    assert s["exp_k4"] == "3.95", s["exp_k4"]
    assert s["exp_k3a"] == "3.08", s["exp_k3a"]
    assert s["p32_mse"] == "6.95e-9", s["p32_mse"]
    assert s["p32_units"] == "525,045", s["p32_units"]
    assert s["m1_units"] == "756,620", s["m1_units"]
    assert s["p32_cut_pct"] == "31%", s["p32_cut_pct"]
    assert s["v29_adj"] == "5.418e-9", s["v29_adj"]
    assert s["young_keep"] == "99.5%" and f"{young_keep_final:.1f}%" == "99.5%", s["young_keep"]
    assert s["s335301_adj"] == "3.494e-9", s["s335301_adj"]
    assert s["s335301_rank"] == "1", s["s335301_rank"]
    assert s["writeups_n"] == "8", s["writeups_n"]
    assert s["lt2_params"] == "54,561", s["lt2_params"]
    assert s["s336537_adj"] == "3.145e-9", s["s336537_adj"]
    assert s["s336537_rank"] == "1", s["s336537_rank"]
    assert s["below_next_pct"] == "5.9%", s["below_next_pct"]
    return s


# ================================================================ cel & specimen primitives
def _panel_box(fx, x, y, w, h, role="gray", level=0, tone_lvl=1, z=1.5, r=0.022):
    """A cel stage plate with hard cast shadow, screentone ground, inner rim highlight, and corner tick marks."""
    f, s, l = cel.ramp(role, level)
    face = cel.wash(role) if level == 0 else f
    sh = FancyBboxPatch((x + 1.2 / 72, y - 1.2 / 72), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc=cel.SHADE, ec="none", zorder=z - 0.1, gid="texture")
    fx.add_patch(sh)
    bg = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc=face, ec=l, lw=0.6, zorder=z, gid="texture")
    fx.add_patch(bg)
    if tone_lvl:
        tp = Rectangle((x + 0.014, y + 0.014), max(0.01, w - 0.028), max(0.01, h - 0.028),
                       fc="none", ec="none", lw=0, zorder=z + 0.02, gid="texture")
        fx.add_patch(cel.tier(tp, cel.light(role), tone_lvl))
    # top highlight gleam + bottom-right inner shadow band
    fx.plot([x + 0.028, x + w - 0.028], [y + h - 0.011, y + h - 0.011],
            color=cel.PAPER, lw=0.65, zorder=z + 0.05, gid="texture")
    fx.plot([x + 0.028, x + w - 0.028], [y + 0.012, y + 0.012],
            color=cel.light(role), lw=0.5, zorder=z + 0.04, gid="texture")


def _specimen_well(fx, x, y, w, h, role="gray", z=2.0):
    """Recessed paper specimen well inside a stage plate: crisp double hairline + fine ruler ticks."""
    ln = cel.line(role)
    fx.add_patch(Rectangle((x + 0.010, y - 0.010), w, h, fc=cel.light(role), ec="none", zorder=z - 0.05, gid="texture"))
    fx.add_patch(Rectangle((x, y), w, h, fc=cel.PAPER, ec=ln, lw=0.45, zorder=z, gid="texture"))
    # subtle fine grid inside the specimen well
    for frac in (0.25, 0.50, 0.75):
        fx.plot([x + w * frac, x + w * frac], [y, y + h], color=HUES["gray"][3], lw=0.28, zorder=z + 0.01, gid="texture")
        fx.plot([x, x + w], [y + h * frac, y + h * frac], color=HUES["gray"][3], lw=0.28, zorder=z + 0.01, gid="texture")
    # top ruler ticks
    for k in range(9):
        tx = x + (k + 0.5) / 9 * w
        th = 0.018 if k % 2 == 0 else 0.010
        fx.plot([tx, tx], [y + h, y + h - th], color=cel.light(role), lw=0.32, zorder=z + 0.02, gid="texture")


def _flow_arrow(fx, x0, y0, x1, y1, role="gray", lw=0.85, z=4.0):
    """Cel connector arrow with pale rim and mid-tone shaft, tagged texture."""
    c = cel.deeper(cel.line(role), 0.10)
    fx.plot([x0, x1], [y0, y1], color=cel.PAPER, lw=lw + 1.6, solid_capstyle="round", zorder=z - 0.05, gid="texture")
    fx.plot([x0, x1], [y0, y1], color=c, lw=lw, solid_capstyle="round", zorder=z, gid="texture")
    dx, dy = x1 - x0, y1 - y0
    L = float(np.hypot(dx, dy))
    if L > 1e-6:
        ux, uy = dx / L, dy / L
        px, py = -uy, ux
        hl, hw = 0.052, 0.026
        pts = [(x1, y1), (x1 - hl * ux + hw * px, y1 - hl * uy + hw * py),
               (x1 - hl * ux - hw * px, y1 - hl * uy - hw * py)]
        fx.add_patch(Polygon(pts, closed=True, fc=c, ec=c, lw=0.3, zorder=z + 0.05, gid="texture"))


def _iso_cube(fx, x, y, w, h, dx, dy, role="k4", slices=3, z=2.5):
    """True 3D oblique tensor block (n x n x R) with front grid, top face, right face, and slice seams."""
    cs = cel.shades(role)
    ln = cel.deeper(cel.line(role), 0.14)
    # cast shadow
    fx.add_patch(Polygon([(x + 0.012, y - 0.012), (x + w + 0.012, y - 0.012),
                          (x + w + dx + 0.012, y + dy - 0.012), (x + w + dx + 0.012, y + h + dy - 0.012),
                          (x + dx + 0.012, y + h + dy - 0.012), (x + 0.012, y + h - 0.012)],
                         closed=True, fc=cel.SHADE, ec="none", zorder=z - 0.05, gid="texture"))
    # front face (n x n)
    fx.add_patch(Rectangle((x, y), w, h, fc=cs[1], ec=ln, lw=0.5, zorder=z + 0.1, gid="texture"))
    for i in range(3):
        for j in range(3):
            c_cell = cs[3] if i == j else (cs[2] if (i + j) % 2 == 0 else cs[0])
            fx.add_patch(Rectangle((x + j * w / 3, y + (2 - i) * h / 3), w / 3, h / 3,
                                   fc=c_cell, ec=cel.PAPER, lw=0.32, zorder=z + 0.12, gid="texture"))
    # top face (n x R)
    top_pts = [(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)]
    fx.add_patch(Polygon(top_pts, closed=True, fc=cs[0], ec=ln, lw=0.45, zorder=z + 0.15, gid="texture"))
    # right face (n x R)
    right_pts = [(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)]
    fx.add_patch(Polygon(right_pts, closed=True, fc=cs[3], ec=ln, lw=0.45, zorder=z + 0.15, gid="texture"))
    # R-slice seams along depth (dx, dy)
    for k in range(1, slices):
        t = k / slices
        sx, sy = dx * t, dy * t
        fx.plot([x + sx, x + w + sx], [y + h + sy, y + h + sy], color=ln, lw=0.32, zorder=z + 0.18, gid="texture")
        fx.plot([x + w + sx, x + w + sx], [y + sy, y + h + sy], color=cel.PAPER, lw=0.32, zorder=z + 0.18, gid="texture")
    # top-left white gleam
    fx.plot([x + 0.008, x + w - 0.008], [y + h - 0.006, y + h - 0.006], color=cel.PAPER, lw=0.55,
            zorder=z + 0.2, gid="texture")


def draw_tensor_factored(fx, x, y, w, h, z=2.5):
    """True specimen for Track 1 Stage 1:
    - 3D isometric cube B_r (n x n x R) in static host RAM (120 GB)
    - 2D transport matrix Q (n x n, 60.4 GB GPU peak, 1 n^3/layer)
    - 2 thin legs x, y (n x R)
    """
    _specimen_well(fx, x, y, w, h, role="k4", z=z - 0.2)
    cs = cel.shades("k4")
    ln = cel.deeper(cel.line("k4"), 0.14)
    # 1. 3D isometric birth kernel B_r on the left
    cw = w * 0.31
    ch = h * 0.56
    dx, dy = w * 0.11, h * 0.18
    _iso_cube(fx, x + w * 0.05, y + h * 0.14, cw, ch, dx, dy, role="k4", slices=4, z=z + 0.1)
    # 2. 2D n x n transport matrix Q in the middle
    qx0, qy0 = x + w * 0.53, y + h * 0.14
    qw, qh = w * 0.27, h * 0.68
    fx.add_patch(Rectangle((qx0 + 0.012, qy0 - 0.012), qw, qh, fc=cel.SHADE, ec="none", zorder=z, gid="texture"))
    fx.add_patch(Rectangle((qx0, qy0), qw, qh, fc=cs[1], ec=ln, lw=0.5, zorder=z + 0.05, gid="texture"))
    pat = ((3, 1, 2, 0), (1, 3, 0, 2), (2, 0, 3, 1), (0, 2, 1, 3))
    for i in range(4):
        for j in range(4):
            fx.add_patch(Rectangle((qx0 + j * qw / 4, qy0 + (3 - i) * qh / 4), qw / 4, qh / 4,
                                   fc=cs[pat[i][j]], ec=cel.PAPER, lw=0.28, zorder=z + 0.1, gid="texture"))
    # 3. Thin legs x, y on the right
    lx0 = x + w * 0.85
    lw_ = w * 0.09
    for k, (yy, ht, step) in enumerate(((qy0 + qh * 0.54, qh * 0.46, 3), (qy0, qh * 0.46, 2))):
        fx.add_patch(Rectangle((lx0 + 0.008, yy - 0.008), lw_, ht, fc=cel.SHADE, ec="none", zorder=z, gid="texture"))
        fx.add_patch(Rectangle((lx0, yy), lw_, ht, fc=cs[step], ec=ln, lw=0.45, zorder=z + 0.1, gid="texture"))
        fx.plot([lx0, lx0 + lw_], [yy + ht / 2, yy + ht / 2], color=cel.PAPER, lw=0.3, zorder=z + 0.12, gid="texture")


def draw_outer_readout(fx, x, y, w, h, z=2.5):
    """True specimen for Track 1 Stage 2:
    3-leg outer product contraction Sym(sum_r (Qx) otimes (Qy) otimes Q B_r Q^T) assembling S_l + 11.8x MSE bars.
    """
    _specimen_well(fx, x, y, w, h, role="k4", z=z - 0.2)
    cs = cel.shades("k4")
    ln = cel.deeper(cel.line("k4"), 0.14)
    # Left half: 3 tensor legs converging into 3D symmetric tensor S_l
    cx, cy = x + w * 0.24, y + h * 0.48
    angles = [142, 218, 18]
    for idx, ang in enumerate(angles):
        rad = np.radians(ang)
        x1 = cx + np.cos(rad) * (w * 0.165)
        y1 = cy + np.sin(rad) * (h * 0.33)
        fx.plot([cx, x1], [cy, y1], color=ln, lw=0.85, zorder=z + 0.05, gid="texture")
        bw, bh = w * 0.075, h * 0.13
        fx.add_patch(Rectangle((x1 - bw / 2, y1 - bh / 2), bw, bh, fc=cs[idx + 1], ec=ln, lw=0.45,
                               zorder=z + 0.12, gid="texture"))
    # Central 3D symmetric tensor cube S_l at (cx, cy)
    sw, sh_ = w * 0.125, h * 0.24
    _iso_cube(fx, cx - sw / 2, cy - sh_ / 2, sw, sh_, w * 0.045, h * 0.085, role="k4", slices=2, z=z + 0.2)
    # Right half: comparative error bars (K=3 aug 2.026e-8 vs K=4 full 1.717e-9, 11.8x)
    bx0 = x + w * 0.54
    bw_max = w * 0.40
    by_top = y + h * 0.58
    by_bot = y + h * 0.20
    bh_bar = h * 0.22
    # K=3 aug bar (gray with cel shadow)
    fx.add_patch(Rectangle((bx0 + 0.008, by_top - 0.008), bw_max, bh_bar, fc=cel.SHADE, ec="none",
                           zorder=z + 0.05, gid="texture"))
    fx.add_patch(Rectangle((bx0, by_top), bw_max, bh_bar, fc=cel.mid("k3"), ec=cel.line("k3"), lw=0.45,
                           zorder=z + 0.1, gid="texture"))
    # K=4 full bar (deep blue, 11.8x lower)
    w_k4 = bw_max * 0.24
    fx.add_patch(Rectangle((bx0 + 0.008, by_bot - 0.008), w_k4, bh_bar, fc=cel.SHADE, ec="none",
                           zorder=z + 0.05, gid="texture"))
    fx.add_patch(Rectangle((bx0, by_bot), w_k4, bh_bar, fc=cs[3], ec=ln, lw=0.5,
                           zorder=z + 0.1, gid="texture"))
    # dashed span indicator between K=4 bar end and K=3 bar end
    fx.plot([bx0 + w_k4, bx0 + bw_max], [by_bot + bh_bar / 2, by_bot + bh_bar / 2],
            color=cel.line("focus"), lw=0.55, ls=(0, (1.4, 1.0)), zorder=z + 0.12, gid="texture")
    fx.plot([bx0 + bw_max, bx0 + bw_max], [by_bot + bh_bar * 0.15, by_top],
            color=cel.line("focus"), lw=0.5, ls=(0, (1.2, 1.0)), zorder=z + 0.12, gid="texture")


def draw_rank_pairs(fx, x, y, w, h, z=2.5):
    """True specimen for Track 1 Stage 3:
    Rank-r matrix-pair factorization sum_{k=1}^{32} u_k v_k^T replacing dense birth kernel, plus unit comparison bar.
    """
    _specimen_well(fx, x, y, w, h, role="k4", z=z - 0.2)
    cs = cel.shades("k4")
    ln = cel.deeper(cel.line("k4"), 0.14)
    px0 = x + w * 0.04
    for k in range(3):
        ox = px0 + k * (w * 0.165)
        oy = y + h * 0.14
        uh = h * 0.70
        uw = w * 0.042
        vw = w * 0.105
        vh = h * 0.11
        # column u_k
        fx.add_patch(Rectangle((ox, oy), uw, uh, fc=cs[3 - k], ec=ln, lw=0.4, zorder=z + 0.12, gid="texture"))
        # row v_k^T
        fx.add_patch(Rectangle((ox + uw + 0.008, oy + uh - vh), vw, vh, fc=cs[2 - k // 2], ec=ln, lw=0.4,
                               zorder=z + 0.12, gid="texture"))
        # rank-1 outer product plane u_k v_k^T
        fx.add_patch(Rectangle((ox + uw + 0.008, oy), vw, uh - vh - 0.008, fc=cs[0], ec=cel.light("k4"), lw=0.32,
                               zorder=z + 0.06, gid="texture"))
    # Unit comparison bars on right half: 756,620 dense vs 525,045 (-31%)
    bx0 = x + w * 0.55
    bw_max = w * 0.39
    bh_bar = h * 0.22
    by_top = y + h * 0.58
    by_bot = y + h * 0.20
    fx.add_patch(Rectangle((bx0, by_top), bw_max, bh_bar, fc=cel.light("k3"), ec=cel.line("k3"), lw=0.45,
                           zorder=z + 0.1, gid="texture"))
    fx.add_patch(Rectangle((bx0, by_bot), bw_max * 0.694, bh_bar, fc=cs[3], ec=ln, lw=0.5,
                           zorder=z + 0.1, gid="texture"))
    p_cut = Rectangle((bx0 + bw_max * 0.694, by_bot), bw_max * 0.306, bh_bar, fc=cel.wash("k3"),
                      ec=ln, lw=0.4, ls=(0, (1.4, 1.0)), zorder=z + 0.08, gid="texture")
    fx.add_patch(cel.hatched(p_cut, cel.light("k4"), lw=0.28, n=5, kind="/"))


def draw_pruned_columns_and_atoms(fx, x, y, w, h, z=2.5):
    """True specimen for Track 2 Stage 1:
    Left half: n x n matrix with columns where w_1 < 0.014 blanked/skipped (active-set trimming).
    Right half: sorted young-atom spectrum bars keeping 99.5% of s_q mass, tail atoms zeroed.
    """
    _specimen_well(fx, x, y, w, h, role="chainb", z=z - 0.2)
    cs = cel.shades("chainb")
    ln = cel.deeper(cel.line("chainb"), 0.14)
    mx0, my0 = x + w * 0.04, y + h * 0.14
    mw, mh = w * 0.43, h * 0.72
    fx.add_patch(Rectangle((mx0 + 0.010, my0 - 0.010), mw, mh, fc=cel.SHADE, ec="none", zorder=z, gid="texture"))
    fx.add_patch(Rectangle((mx0, my0), mw, mh, fc=cel.PAPER, ec=ln, lw=0.5, zorder=z + 0.05, gid="texture"))
    active_cols = (1, 0, 1, 1, 0, 1, 0, 1)
    cw = mw / len(active_cols)
    for idx, keep in enumerate(active_cols):
        cx_ = mx0 + idx * cw
        if keep:
            for r_i in range(4):
                c_step = 3 if (idx + r_i) % 3 == 0 else (2 if (idx + r_i) % 2 == 0 else 1)
                fx.add_patch(Rectangle((cx_ + 0.002, my0 + r_i * mh / 4 + 0.002), cw - 0.004, mh / 4 - 0.004,
                                       fc=cs[c_step], ec="none", zorder=z + 0.1, gid="texture"))
        else:
            p = Rectangle((cx_ + 0.002, my0 + 0.004), cw - 0.004, mh - 0.008,
                          fc=cel.wash("k3"), ec="none", zorder=z + 0.08, gid="texture")
            fx.add_patch(cel.hatched(p, cel.light("k3"), lw=0.28, n=6, kind="/"))
    # Sorted young-atom s_q mass spectrum on the right
    bx0, by0 = x + w * 0.53, y + h * 0.14
    bw, bh = w * 0.42, h * 0.72
    heights = [0.95, 0.78, 0.62, 0.48, 0.34, 0.22, 0.11, 0.05]
    bar_w = bw / len(heights)
    for idx, ht in enumerate(heights):
        kept = idx < 6
        fc = cs[3 if idx < 2 else (2 if idx < 4 else 1)] if kept else cel.wash("k3")
        ec = ln if kept else cel.line("k3")
        fx.add_patch(Rectangle((bx0 + idx * bar_w + 0.003, by0), bar_w - 0.006, bh * ht,
                               fc=fc, ec=ec, lw=0.35, zorder=z + 0.1, gid="texture"))
    cut_x = bx0 + 6 * bar_w
    fx.plot([cut_x, cut_x], [by0, by0 + bh * 0.98], color=cel.line("focus"), lw=0.65, ls=(0, (1.5, 1.0)),
            zorder=z + 0.15, gid="texture")


def draw_ssc_closure(fx, x, y, w, h, z=2.5):
    """True specimen for Track 2 Stage 2:
    Puffi SSC closure on the regeneration channel with 9 refitted coefficients (closed loop + 9-bar spectrum).
    """
    _specimen_well(fx, x, y, w, h, role="chainb", z=z - 0.2)
    cs = cel.shades("chainb")
    ln = cel.deeper(cel.line("chainb"), 0.14)
    bx0, by0 = x + w * 0.05, y + h * 0.14
    bw, bh = w * 0.90, h * 0.72
    # Main K=3 rail on top + regeneration feedback loop into SSC closure node
    y_rail = by0 + bh * 0.80
    fx.plot([bx0, bx0 + bw], [y_rail, y_rail], color=ln, lw=1.0, zorder=z + 0.1, gid="texture")
    for frac in (0.15, 0.50, 0.85):
        px = bx0 + bw * frac
        fx.add_patch(Rectangle((px - 0.018, y_rail - 0.018), 0.036, 0.036, fc=cs[2], ec=ln, lw=0.4,
                               zorder=z + 0.15, gid="texture"))
    # 9 refitted SSC coefficients along lower strip, wired up to the regeneration channel
    coeff_h = [0.54, 0.72, 0.46, 0.64, 0.78, 0.40, 0.60, 0.50, 0.68]
    cw = bw / 9.0
    for i, ch in enumerate(coeff_h):
        cx_ = bx0 + i * cw + 0.003
        ht = bh * 0.54 * ch
        fx.add_patch(Rectangle((cx_, by0), cw - 0.006, ht,
                               fc=cs[1 + (i % 3)], ec=ln, lw=0.35, zorder=z + 0.12, gid="texture"))
        fx.plot([cx_ + (cw - 0.006) / 2, cx_ + (cw - 0.006) / 2], [by0 + ht, y_rail - 0.018],
                color=cel.light("chainb"), lw=0.3, zorder=z + 0.08, gid="texture")


def draw_sidecar_corrector(fx, x, y, w, h, z=2.5):
    """True specimen for Track 2 Stage 3:
    Propagated state rail passes straight through untouched; a read-only vertical tap feeds the 54,561-parameter
    learned corrector network (134 -> 128^3 -> 33) which adds only to the final readout.
    """
    _specimen_well(fx, x, y, w, h, role="chainb", z=z - 0.2)
    cs = cel.shades("chainb")
    ln = cel.deeper(cel.line("chainb"), 0.14)
    x0, x1 = x + w * 0.05, x + w * 0.95
    y_rail = y + h * 0.76
    y_side = y + h * 0.30
    # Unchanged propagated state rail on top
    fx.plot([x0, x1], [y_rail, y_rail], color=ln, lw=1.05, zorder=z + 0.1, gid="texture")
    for frac in (0.16, 0.48, 0.84):
        px = x0 + (x1 - x0) * frac
        fx.add_patch(Rectangle((px - 0.018, y_rail - 0.018), 0.036, 0.036, fc=cs[1], ec=ln, lw=0.45,
                               zorder=z + 0.15, gid="texture"))
    # Read-only tap down to sidecar MLP layers (134 -> 128 -> 128 -> 128 -> 33) and back up to final readout
    tap_x = x0 + (x1 - x0) * 0.16
    out_x = x0 + (x1 - x0) * 0.84
    fx.plot([tap_x, tap_x], [y_rail - 0.018, y_side], color=ln, lw=0.55, ls=(0, (1.4, 1.0)),
            zorder=z + 0.1, gid="texture")
    fracs = (0.30, 0.42, 0.54, 0.66, 0.76)
    heights = (0.30, 0.42, 0.42, 0.42, 0.22)
    for k, (frac, h_rel) in enumerate(zip(fracs, heights)):
        lx = x0 + (x1 - x0) * frac
        lh = h * h_rel
        fx.add_patch(Rectangle((lx - 0.014, y_side - lh / 2), 0.028, lh,
                               fc=cs[3 if 1 <= k <= 3 else 2], ec=ln, lw=0.38, zorder=z + 0.15, gid="texture"))
        if k > 0:
            prev_x = x0 + (x1 - x0) * fracs[k - 1]
            fx.plot([prev_x + 0.014, lx - 0.014], [y_side, y_side], color=ln, lw=0.45,
                    zorder=z + 0.12, gid="texture")
    fx.plot([tap_x, x0 + (x1 - x0) * fracs[0] - 0.014], [y_side, y_side], color=ln, lw=0.45,
            zorder=z + 0.12, gid="texture")
    fx.plot([x0 + (x1 - x0) * fracs[-1] + 0.014, out_x, out_x], [y_side, y_side, y_rail - 0.020],
            color=ln, lw=0.55, zorder=z + 0.12, gid="texture")


# ================================================================ Variant A: Left-Right Parallel Dual-Track Pipeline
def build_variant_a(S):
    H = 2.94
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)

    # Top shared input sheet: ARC's K=3 / K=4 cumulant recursion (Wu et al. 2026)
    in_x, in_y, in_w, in_h = 1.10, 2.63, 4.61, 0.25
    _panel_box(fx, in_x, in_y, in_w, in_h, role="k3", level=0, tone_lvl=1, z=2.0)
    cel.brackets(fig, in_x, in_y, in_w, in_h, arm=4.0, lw=0.55, color=cel.LINE, z=3.0)
    fx.text(in_x + in_w / 2, in_y + in_h / 2,
            "ARC's K=3 / K=4 cumulant recursion (Wu et al. 2026)",
            ha="center", va="center", fontsize=FS, color=INK, **cel.BOLD, zorder=5)

    lx, ly, lw, lh = 0.08, 0.08, 3.24, 2.39
    # Bifurcation arrows from shared input to Track 1 (left) and Track 2 (right)
    _flow_arrow(fx, 2.25, in_y, 1.70, ly + lh + 0.02, role="k4", lw=0.95)
    _flow_arrow(fx, 4.56, in_y, 5.11, ly + lh + 0.02, role="chainb", lw=0.95)

    # Left sheet (a): Regime 3, full-scale factored K=4 (sections 2, 5)
    cel.sheet(fig, lx, ly, lw, lh, drop_pt=2.2, stack=2.0)
    cel.corners(fig, lx, ly, lw, lh, which="1101")
    cel.pegs(fig, lx + lw * 0.84, ly + lh - 0.055, s=0.82)
    cel.badge(fig, lx - 0.04, ly + lh - 0.075, "a", "gray", 0, gid="tab")

    fx.text(lx + 0.12, ly + lh - 0.12,
            r"Regime 3: full-scale factored K=4 (§2, §5)",
            ha="left", va="center", fontsize=FS, color=cel.deeper(cel.line("k4"), 0.18), **cel.BOLD)

    # Stage A1: Factored storage (x, y, B, Q) & memory wall
    s1_y, s1_h = 1.62, 0.66
    s2_y, s2_h = 0.82, 0.68
    s3_y, s3_h = 0.16, 0.54
    _panel_box(fx, lx + 0.07, s1_y, lw - 0.14, s1_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_tensor_factored(fx, lx + 0.11, s1_y + 0.06, 0.80, s1_h - 0.12, z=2.4)
    tx1 = lx + 0.96
    fx.text(tx1, s1_y + 0.53,
            r"Store $(x,y,B,Q)$, $Q=T_l\cdots T_{b+1}$ ($n\times n$, 1 $n^3$/layer)",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx1, s1_y + 0.33,
            r"Static host kernel $B_r$ ($n\times n\times R$), written once",
            ha="left", va="center", fontsize=FSS, color=MUTED)
    fx.text(tx1, s1_y + 0.13,
            r"~256 GB $\rightarrow$ 120 GB host + 60.4 GB GPU peak",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18), **cel.BOLD)

    _flow_arrow(fx, lx + lw / 2, s1_y, lx + lw / 2, s2_y + s2_h + 0.01, role="k4", lw=0.85)

    # Stage A2: Readout S_l (Proposition 1) & 11.8x lower MSE at 1024 x 16
    _panel_box(fx, lx + 0.07, s2_y, lw - 0.14, s2_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_outer_readout(fx, lx + 0.11, s2_y + 0.06, 0.80, s2_h - 0.12, z=2.4)
    fx.text(tx1, s2_y + 0.545,
            r"$S_l=\mathrm{Sym}(\sum_r(Qx)\otimes(Qy)\otimes QB_rQ^\top)$, 0 approx.",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx1, s2_y + 0.34,
            f"Prop. 1; 1024{X}16 MSE {S['k4_mse']} vs {S['k3a_mse']}",
            ha="left", va="center", fontsize=FSS, color=MUTED)
    fx.text(tx1, s2_y + 0.14,
            rf"Width law $n^{{-{S['exp_k4']}}}$ vs $n^{{-{S['exp_k3a']}}}$",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18))
    # Accent #1: 11.8x
    cel.chip(fx, lx + lw - 0.14, s2_y + 0.15, f"{S['ratio_1024']}{X}", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)

    _flow_arrow(fx, lx + lw / 2, s2_y, lx + lw / 2, s3_y + s3_h + 0.01, role="k4", lw=0.85)

    # Stage A3: Rank-r matrix-pair compression
    _panel_box(fx, lx + 0.07, s3_y, lw - 0.14, s3_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_rank_pairs(fx, lx + 0.11, s3_y + 0.06, 0.80, s3_h - 0.12, z=2.4)
    fx.text(tx1, s3_y + 0.39,
            f"Rank-$r$ matrix pairs: 32 pairs MSE {S['p32_mse']}",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx1, s3_y + 0.15,
            f"{S['p32_units']} vs dense {S['m1_units']} units (−{S['p32_cut_pct']})",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18))

    # Right sheet (b): Regime 2, in-budget K=3 chain B (section 4)
    rx, ry, rw, rh = 3.49, 0.08, 3.24, 2.39
    cel.sheet(fig, rx, ry, rw, rh, drop_pt=2.2, stack=2.0)
    cel.corners(fig, rx, ry, rw, rh, which="1101")
    cel.pegs(fig, rx + rw * 0.84, ry + rh - 0.055, s=0.82)
    cel.badge(fig, rx - 0.04, ry + rh - 0.075, "b", "gray", 0, gid="tab")

    fx.text(rx + 0.12, ry + rh - 0.12,
            r"Regime 2: in-budget K=3 chain B (§4)",
            ha="left", va="center", fontsize=FS, color=cel.deeper(cel.line("chainb"), 0.18), **cel.BOLD)

    # Stage B1: 504aldo V29 + cost work + Puffi SSC closure (Table 4, steps 1-5)
    tx2 = rx + 0.96
    _panel_box(fx, rx + 0.07, s1_y, rw - 0.14, s1_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_ssc_closure(fx, rx + 0.11, s1_y + 0.06, 0.80, s1_h - 0.12, z=2.4)
    fx.text(tx2, s1_y + 0.53,
            f"504aldo V29 (adj {S['v29_adj']}) + Puffi SSC closure",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx2, s1_y + 0.33,
            "9 closure coefficients refit end to end",
            ha="left", va="center", fontsize=FSS, color=MUTED)
    fx.text(tx2, s1_y + 0.13,
            "Plus Strassen joins, SVD ranks, wall-clock blocks",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18))

    _flow_arrow(fx, rx + rw / 2, s1_y, rx + rw / 2, s2_y + s2_h + 0.01, role="chainb", lw=0.85)

    # Stage B2: readout-guided pruning (Table 4, steps 6-7) -> #335301
    _panel_box(fx, rx + 0.07, s2_y, rw - 0.14, s2_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_pruned_columns_and_atoms(fx, rx + 0.11, s2_y + 0.06, 0.80, s2_h - 0.12, z=2.4)
    fx.text(tx2, s2_y + 0.545,
            f"Trim columns $w_1<0.014$, young atoms {S['young_keep']} $s_q$",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx2, s2_y + 0.34,
            f"#{S['s335301_rank']} of {S['writeups_n']} write-ups' chains w/o corrector/table",
            ha="left", va="center", fontsize=FSS, color=MUTED)
    fx.text(tx2, s2_y + 0.14,
            f"adj {S['s335301_adj']}",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18), **cel.BOLD)
    # Accent #2: #335301
    cel.chip(fx, rx + rw - 0.14, s2_y + 0.15, "#335301", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)

    _flow_arrow(fx, rx + rw / 2, s2_y, rx + rw / 2, s3_y + s3_h + 0.01, role="chainb", lw=0.85)

    # Stage B3: 54,561-param learned corrector -> #336537
    _panel_box(fx, rx + 0.07, s3_y, rw - 0.14, s3_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_sidecar_corrector(fx, rx + 0.11, s3_y + 0.06, 0.80, s3_h - 0.12, z=2.4)
    fx.text(tx2, s3_y + 0.39,
            f"+ {S['lt2_params']}-param read-only corrector",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(tx2, s3_y + 0.15,
            f"adj {S['s336537_adj']}, #{S['s336537_rank']} of {S['writeups_n']}, −{S['below_next_pct']} vs #335413",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18))
    # Accent #3: #336537
    cel.chip(fx, rx + rw - 0.14, s3_y + 0.16, "#336537", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)

    return fig


# ================================================================ Variant B: Top-Bottom Shared Layer-Axis Architecture
def build_variant_b(S):
    H = 3.24
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)

    # Left vertical shared input spine
    sx, sy, sw, sh = 0.08, 0.34, 1.04, 2.80
    cel.sheet(fig, sx, sy, sw, sh, drop_pt=1.8, stack=1.4)
    _panel_box(fx, sx + 0.06, sy + 0.08, sw - 0.12, sh - 0.16, role="k3", level=0, tone_lvl=1, z=1.8)
    fx.text(sx + sw / 2, sy + 2.38, "Shared Input", ha="center", va="center",
            fontsize=FS, color=INK, **cel.BOLD)
    fx.text(sx + sw / 2, sy + 1.92, "ARC's\nK=3 / K=4\ncumulant\nrecursion\n(Wu et al.\n2026)",
            ha="center", va="center", fontsize=FSS, color=INK, linespacing=1.18)
    for k in range(4):
        yy = sy + 0.32 + k * 0.24
        fx.add_patch(Rectangle((sx + 0.22, yy), sw - 0.44, 0.13, fc=cel.shades("k3")[k],
                               ec=cel.line("k3"), lw=0.4, zorder=2.5, gid="texture"))
        if k < 3:
            _flow_arrow(fx, sx + sw / 2, yy + 0.13, sx + sw / 2, yy + 0.24, role="k3", lw=0.6)

    # Arrows from shared input into Top Rail (Track 1) and Bottom Rail (Track 2)
    _flow_arrow(fx, sx + sw, 2.46, 1.25, 2.46, role="k4", lw=0.95)
    _flow_arrow(fx, sx + sw, 1.04, 1.25, 1.04, role="chainb", lw=0.95)

    # Top Rail (a): Regime 3, full-scale factored K=4
    rx, tw, th = 1.25, 5.48, 1.36
    ty = 1.78
    cel.sheet(fig, rx, ty, tw, th, drop_pt=2.0, stack=1.6)
    cel.pegs(fig, rx + tw * 0.88, ty + th - 0.055, s=0.80)
    cel.badge(fig, rx - 0.04, ty + th - 0.075, "a", "gray", 0, gid="tab")
    fx.text(rx + 0.12, ty + th - 0.11,
            r"Regime 3: full-scale factored K=4 (§2, §5)  |  ~256 GB $\rightarrow$ 120 GB host + 60.4 GB GPU peak",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18), **cel.BOLD)

    # 3 horizontal stage boxes along Top Rail
    c1_x, c1_w = rx + 0.07, 1.66
    c2_x, c2_w = rx + 1.81, 1.98
    c3_x, c3_w = rx + 3.87, 1.54
    cy_top, ch_top = ty + 0.08, 1.08

    # Box T1: Factored tuple (x, y, B, Q)
    _panel_box(fx, c1_x, cy_top, c1_w, ch_top, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_tensor_factored(fx, c1_x + 0.06, cy_top + 0.44, c1_w - 0.12, 0.56, z=2.4)
    fx.text(c1_x + 0.06, cy_top + 0.31,
            r"$(x,y,B,Q)$, $Q=T_l\cdots T_{b+1}$",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c1_x + 0.06, cy_top + 0.12,
            r"1 $n^3$/layer; $B_r$ ($n\times n\times R$) host",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    _flow_arrow(fx, c1_x + c1_w, cy_top + ch_top / 2, c2_x, cy_top + ch_top / 2, role="k4", lw=0.85)

    # Box T2: Readout S_l + 11.8x
    _panel_box(fx, c2_x, cy_top, c2_w, ch_top, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_outer_readout(fx, c2_x + 0.06, cy_top + 0.48, c2_w - 0.78, 0.52, z=2.4)
    cel.chip(fx, c2_x + c2_w - 0.08, cy_top + 0.78, f"{S['ratio_1024']}{X}", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(c2_x + 0.06, cy_top + 0.32,
            r"$S_l=\mathrm{Sym}(\sum_r(Qx)\otimes(Qy)\otimes QB_rQ^\top)$",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c2_x + 0.06, cy_top + 0.12,
            rf"{S['k4_mse']} vs {S['k3a_mse']} ($n^{{-{S['exp_k4']}}}$ vs $n^{{-{S['exp_k3a']}}}$)",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    _flow_arrow(fx, c2_x + c2_w, cy_top + ch_top / 2, c3_x, cy_top + ch_top / 2, role="k4", lw=0.85)

    # Box T3: Rank-r matrix pairs
    _panel_box(fx, c3_x, cy_top, c3_w, ch_top, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_rank_pairs(fx, c3_x + 0.06, cy_top + 0.46, c3_w - 0.12, 0.54, z=2.4)
    fx.text(c3_x + 0.06, cy_top + 0.31,
            f"32 pairs MSE {S['p32_mse']}",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c3_x + 0.06, cy_top + 0.12,
            f"{S['p32_units']} vs {S['m1_units']} (−{S['p32_cut_pct']})",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18))

    # Bottom Rail (b): Regime 2, in-budget K=3 chain B
    by = 0.34
    cel.sheet(fig, rx, by, tw, th, drop_pt=2.0, stack=1.6)
    cel.pegs(fig, rx + tw * 0.88, by + th - 0.055, s=0.80)
    cel.badge(fig, rx - 0.04, by + th - 0.075, "b", "gray", 0, gid="tab")
    fx.text(rx + 0.12, by + th - 0.11,
            f"Regime 2: in-budget K=3 chain B (§4)  |  starts from 504aldo V29 (adj {S['v29_adj']})",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18), **cel.BOLD)

    cy_bot, ch_bot = by + 0.08, 1.08

    # Box B1: Active-set trimming + young-atom pruning + 12 cost cuts
    _panel_box(fx, c1_x, cy_bot, c1_w, ch_bot, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_pruned_columns_and_atoms(fx, c1_x + 0.06, cy_bot + 0.44, c1_w - 0.12, 0.56, z=2.4)
    fx.text(c1_x + 0.06, cy_bot + 0.31,
            r"Trim $w_1 < 0.014$ + 12 cuts",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c1_x + 0.06, cy_bot + 0.12,
            f"Keep {S['young_keep']} young $s_q$ mass",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    _flow_arrow(fx, c1_x + c1_w, cy_bot + ch_bot / 2, c2_x, cy_bot + ch_bot / 2, role="chainb", lw=0.85)

    # Box B2: Puffi SSC closure + 9 refit coeffs -> #335301
    _panel_box(fx, c2_x, cy_bot, c2_w, ch_bot, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_ssc_closure(fx, c2_x + 0.06, cy_bot + 0.48, c2_w - 0.82, 0.52, z=2.4)
    cel.chip(fx, c2_x + c2_w - 0.08, cy_bot + 0.78, "#335301", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(c2_x + 0.06, cy_bot + 0.31,
            f"Puffi SSC, 9 coeffs (adj {S['s335301_adj']})",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c2_x + 0.06, cy_bot + 0.12,
            f"#{S['s335301_rank']} of {S['writeups_n']} write-ups (no corrector/table)",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    _flow_arrow(fx, c2_x + c2_w, cy_bot + ch_bot / 2, c3_x, cy_bot + ch_bot / 2, role="chainb", lw=0.85)

    # Box B3: 54,561-param learned corrector -> #336537
    _panel_box(fx, c3_x, cy_bot, c3_w, ch_bot, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_sidecar_corrector(fx, c3_x + 0.06, cy_bot + 0.48, c3_w - 0.82, 0.52, z=2.4)
    cel.chip(fx, c3_x + c3_w - 0.08, cy_bot + 0.78, "#336537", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(c3_x + 0.06, cy_bot + 0.31,
            f"+ {S['lt2_params']}-param corrector",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(c3_x + 0.06, cy_bot + 0.12,
            f"adj {S['s336537_adj']}, #{S['s336537_rank']}/{S['writeups_n']} (−{S['below_next_pct']})",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18))

    # Shared bottom layer-progression ruler
    ry_line = 0.21
    fx.plot([rx + 0.08, rx + tw - 0.08], [ry_line, ry_line], color=cel.LINE, lw=0.65, zorder=2.0, gid="texture")
    for x_tick, lab in ((c1_x + c1_w / 2, "Stage 1: Birth & Pruning"),
                        (c2_x + c2_w / 2, "Stage 2: Transport & Closure"),
                        (c3_x + c3_w / 2, r"Stage 3: Readout ($L=16$)")):
        fx.plot([x_tick, x_tick], [ry_line - 0.03, ry_line + 0.03], color=cel.LINE, lw=0.55, zorder=2.0, gid="texture")
        fx.text(x_tick, 0.07, lab, ha="center", va="center", fontsize=FSS, color=MUTED)

    return fig


# ================================================================ Variant C: Central-Spine Diverging Architecture
def build_variant_c(S):
    H = 3.24
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)

    # Central vertical shared recursion spine (x = 2.94..3.87)
    cx, cy, cw, ch = 2.94, 0.10, 0.93, 3.04
    cel.sheet(fig, cx, cy, cw, ch, drop_pt=2.0, stack=1.6)
    _panel_box(fx, cx + 0.06, cy + 0.08, cw - 0.12, ch - 0.16, role="k3", level=0, tone_lvl=1, z=1.8)
    fx.text(cx + cw / 2, cy + 2.72, "Shared\nInput", ha="center", va="center",
            fontsize=FS, color=INK, **cel.BOLD, linespacing=1.10)
    fx.text(cx + cw / 2, cy + 2.15, "ARC's\nK=3 / K=4\ncumulant\nrecursion\n(Wu et al.\n2026)",
            ha="center", va="center", fontsize=FSS, color=INK, linespacing=1.15)
    # 16-layer vertical propagation ladder inside the central spine
    for k in range(6):
        yy = cy + 0.38 + k * 0.20
        fx.add_patch(Rectangle((cx + 0.18, yy), cw - 0.36, 0.11, fc=cel.shades("k3")[min(4, k)],
                               ec=cel.line("k3"), lw=0.38, zorder=2.5, gid="texture"))
    fx.text(cx + cw / 2, cy + 0.20, r"1024 $\times$ 16", ha="center", va="center", fontsize=FSS, color=MUTED)

    # Diverging arrows from central spine to Left wing (Regime 3) and Right wing (Regime 2)
    for yy in (2.32, 1.48, 0.56):
        _flow_arrow(fx, cx, yy, 2.82, yy, role="k4", lw=0.85)
        _flow_arrow(fx, cx + cw, yy, 3.99, yy, role="chainb", lw=0.85)

    # Left Wing (a): Regime 3, full-scale factored K=4 (x = 0.08..2.82)
    lx, ly, lw, lh = 0.08, 0.10, 2.74, 3.04
    cel.sheet(fig, lx, ly, lw, lh, drop_pt=2.0, stack=1.8)
    cel.pegs(fig, lx + lw * 0.82, ly + lh - 0.055, s=0.80)
    cel.badge(fig, lx - 0.04, ly + lh - 0.075, "a", "gray", 0, gid="tab")
    fx.text(lx + 0.14, ly + lh - 0.12,
            r"Regime 3: factored K=4 (§2, §5)",
            ha="left", va="center", fontsize=FS, color=cel.deeper(cel.line("k4"), 0.18), **cel.BOLD)

    # Wing A1 (top): Factored storage & memory wall
    w1_y, w1_h = 1.98, 0.90
    _panel_box(fx, lx + 0.06, w1_y, lw - 0.12, w1_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_tensor_factored(fx, lx + 0.10, w1_y + 0.42, lw - 0.20, 0.42, z=2.4)
    fx.text(lx + 0.10, w1_y + 0.28,
            r"$(x,y,B,Q)$, $Q=T_l\cdots T_{b+1}$ ($n\times n$); $B_r$ ($n\times n\times R$)",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(lx + 0.10, w1_y + 0.11,
            r"~256 GB $\rightarrow$ 120 GB host + 60.4 GB GPU peak",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18), **cel.BOLD)

    # Wing A2 (middle): Readout S_l & 11.8x
    _flow_arrow(fx, lx + lw / 2, w1_y, lx + lw / 2, 1.90, role="k4", lw=0.75)
    w2_y, w2_h = 1.00, 0.90
    _panel_box(fx, lx + 0.06, w2_y, lw - 0.12, w2_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_outer_readout(fx, lx + 0.10, w2_y + 0.42, lw - 0.86, 0.42, z=2.4)
    cel.chip(fx, lx + lw - 0.12, w2_y + 0.64, f"{S['ratio_1024']}{X}", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(lx + 0.10, w2_y + 0.28,
            r"$S_l=\mathrm{Sym}(\sum_r(Qx)\otimes(Qy)\otimes QB_rQ^\top)$ (Prop. 1)",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(lx + 0.10, w2_y + 0.11,
            rf"MSE {S['k4_mse']} vs {S['k3a_mse']} ($n^{{-{S['exp_k4']}}}$ vs $n^{{-{S['exp_k3a']}}}$)",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    # Wing A3 (bottom): Rank-r matrix-pair compression
    _flow_arrow(fx, lx + lw / 2, w2_y, lx + lw / 2, 0.92, role="k4", lw=0.75)
    w3_y, w3_h = 0.16, 0.76
    _panel_box(fx, lx + 0.06, w3_y, lw - 0.12, w3_h, role="k4", level=0, tone_lvl=1, z=1.8)
    draw_rank_pairs(fx, lx + 0.10, w3_y + 0.36, lw - 0.20, 0.34, z=2.4)
    fx.text(lx + 0.10, w3_y + 0.24,
            f"Rank-$r$ pairs: 32 pairs MSE {S['p32_mse']}",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(lx + 0.10, w3_y + 0.09,
            f"{S['p32_units']} vs dense {S['m1_units']} units (−{S['p32_cut_pct']})",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("k4"), 0.18))

    # Right Wing (b): Regime 2, in-budget K=3 chain B (x = 3.99..6.73)
    rx, ry, rw, rh = 3.99, 0.10, 2.74, 3.04
    cel.sheet(fig, rx, ry, rw, rh, drop_pt=2.0, stack=1.8)
    cel.pegs(fig, rx + rw * 0.82, ry + rh - 0.055, s=0.80)
    cel.badge(fig, rx - 0.04, ry + rh - 0.075, "b", "gray", 0, gid="tab")
    fx.text(rx + 0.14, ry + rh - 0.12,
            r"Regime 2: in-budget K=3 chain B (§4)",
            ha="left", va="center", fontsize=FS, color=cel.deeper(cel.line("chainb"), 0.18), **cel.BOLD)

    # Wing B1 (top): V29 + active-set trimming + young-atom pruning + 12 cuts
    _panel_box(fx, rx + 0.06, w1_y, rw - 0.12, w1_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_pruned_columns_and_atoms(fx, rx + 0.10, w1_y + 0.42, rw - 0.20, 0.42, z=2.4)
    fx.text(rx + 0.10, w1_y + 0.28,
            f"V29 ({S['v29_adj']}) + skip cols $w_1<0.014$ + 12 cuts",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(rx + 0.10, w1_y + 0.11,
            f"Young-atom pruning keeps {S['young_keep']} of $s_q$ mass",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    # Wing B2 (middle): Puffi SSC closure + 9 refit coeffs -> #335301
    _flow_arrow(fx, rx + rw / 2, w1_y, rx + rw / 2, 1.90, role="chainb", lw=0.75)
    _panel_box(fx, rx + 0.06, w2_y, rw - 0.12, w2_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_ssc_closure(fx, rx + 0.10, w2_y + 0.42, rw - 0.90, 0.42, z=2.4)
    cel.chip(fx, rx + rw - 0.12, w2_y + 0.64, "#335301", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(rx + 0.10, w2_y + 0.28,
            f"Puffi SSC closure, 9 coeffs (adj {S['s335301_adj']})",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(rx + 0.10, w2_y + 0.11,
            f"#{S['s335301_rank']} of {S['writeups_n']} write-ups (no corrector/table)",
            ha="left", va="center", fontsize=FSS, color=MUTED)

    # Wing B3 (bottom): 54,561-param learned corrector -> #336537
    _flow_arrow(fx, rx + rw / 2, w2_y, rx + rw / 2, 0.92, role="chainb", lw=0.75)
    _panel_box(fx, rx + 0.06, w3_y, rw - 0.12, w3_h, role="chainb", level=0, tone_lvl=1, z=1.8)
    draw_sidecar_corrector(fx, rx + 0.10, w3_y + 0.36, rw - 0.90, 0.34, z=2.4)
    cel.chip(fx, rx + rw - 0.12, w3_y + 0.52, "#336537", "focus",
             size=FS, weight="bold", level=1, ha="right", va="center", pad=0.20)
    fx.text(rx + 0.10, w3_y + 0.24,
            f"+ {S['lt2_params']}-param read-only corrector",
            ha="left", va="center", fontsize=FSS, color=INK, **cel.BOLD)
    fx.text(rx + 0.10, w3_y + 0.09,
            f"adj {S['s336537_adj']}, #{S['s336537_rank']}/{S['writeups_n']} (−{S['below_next_pct']} vs #335413)",
            ha="left", va="center", fontsize=FSS, color=cel.deeper(cel.line("chainb"), 0.18))

    return fig


VARIANTS = {
    "A": build_variant_a,
    "B": build_variant_b,
    "C": build_variant_c,
}
KEEP = "A"


def build(v="A"):
    """Build and return the matplotlib Figure for variant v in ('A', 'B', 'C')."""
    setup()
    S = verified_strings()
    if v not in VARIANTS:
        raise ValueError(f"Unknown variant {v!r}; expected one of {tuple(VARIANTS)}")
    return VARIANTS[v](S)


def render(out=HERE / "figures"):
    """Figure M of the report (variant KEEP), called by make_figures.py after numbers.csv is written."""
    fig = build(KEEP)
    hits = figcheck.report(fig, "fig_mech")
    assert not hits, hits
    fig.savefig(Path(out) / "fig_mech.pdf")
    fig.savefig(Path(out) / "fig_mech.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    out_figs = HERE.parent / "figures"
    out_figs.mkdir(parents=True, exist_ok=True)
    render(out_figs)
