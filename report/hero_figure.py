"""Figure 0, the first-page figure: full K=4 against ARC's augmented K=3 and the leaderboard on one raw-error ruler, and
where the complete chain's counted units sit. Three variants for the first-page samples, one kept in the report:
  A  two panels: the width law with the leaderboard band at 1024 | the units ladder from 60,549 to the tie
  B  the width law | the 1024 column magnified: every official network and every leaderboard entry ranked 1-50
  C  three number cards, each over the real data it reads (11.8x, 6.4x, 75x -> 15x)
  D  Figure 3's 16 glass plates, larger, with both factors and the leaderboard on one raw ruler | the width law and the
     units ladder below, unframed (the report's Figure 0 since v22g)

Run from the repo root: python3 report/hero_figure.py D --out figures
Measurements are in data/numbers.csv. The leaderboard inputs are the entries ranked 1-50
and two aggregate fields excerpted from the official 2026-10-18 16:06 page.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cel  # noqa: E402
import figcheck  # noqa: E402
from palette import HUES, INK, MUTED  # noqa: E402
from visual_figures import FS, FSS, W, log_ruler, nudge, rule, say, setup, tube  # noqa: E402

ROOT = HERE.parent
LB = ROOT / "data" / "leaderboard_1018_1606.json"
NUMBERS = ROOT / "data" / "numbers.csv"
SEEDS = {256: (11, 12, 13), 384: (21, 22, 23), 512: (41, 42, 43), 1024: (0, 1, 2, 3, 4)}
WIDTHS = (256, 384, 512, 1024)
X = r"$\times$"


# ---------------------------------------------------------------- data
def numbers(path=NUMBERS):
    with open(path) as f:
        return {r["key"]: float(r["value"]) for r in csv.DictReader(f)}


def board(path=LB):
    """Entries ranked 1-50, with aggregate counts from the same official page."""
    return json.loads(Path(path).read_text())["entries"]


def rows(N, entries):
    """The leaderboard numbers the figure and the first page use, as numbers.csv rows (key, value, unit, what, source)."""
    src = str(LB.relative_to(ROOT))
    meta = json.loads(LB.read_text())
    top = [e for e in entries if e["rank"] <= 50]
    raw = [e["raw"] for e in top]
    r1 = min(entries, key=lambda e: (e["rank"], e["raw"]))
    ours = next(e for e in entries if e["sid"] == "336537")
    k4 = N["width1024_k4_mean"]
    note = "(the board's 50 leaderboard networks against official networks 0-4: same metric and shape, other networks)"
    return [
        ("lb1018_listed_n", meta["listed_n"], "count", "entries on the Phase 2 public leaderboard, 2026-10-18 16:06 JST "
         "(each participant's or team's listed entry)", src),
        ("lb1018_top50_n", len(top), "count", "entries ranked 1-50, ties included", src),
        ("lb1018_top50_raw_min", min(raw), "MSE", "lowest final-layer MSE among entries ranked 1-50", src),
        ("lb1018_top50_raw_max", max(raw), "MSE", "highest final-layer MSE among entries ranked 1-50", src),
        ("lb1018_top50_raw_median", statistics.median(raw), "MSE", "median final-layer MSE, entries ranked 1-50", src),
        ("lb1018_rank1_adj", r1["adj"], "score", f"rank 1 ({r1['who']}, #{r1['sid']}) adjusted score", src),
        ("lb1018_rank1_raw", r1["raw"], "MSE", f"rank 1 ({r1['who']}, #{r1['sid']}) final-layer MSE", src),
        ("lb1018_lowest_raw", meta["lowest_raw"], "MSE", "lowest final-layer MSE of every listed entry", src),
        ("lb1018_ours_rank", ours["rank"], "rank", "#336537 rank on the same page", src),
        ("lb1018_rank1_raw_over_k4", r1["raw"] / k4, "x", "rank 1's final-layer MSE over full K=4's mean at 1024 x 16 "
         + note, "derived"),
        ("lb1018_top50_raw_max_over_k4", max(raw) / k4, "x", "highest final-layer MSE ranked 1-50 over full K=4's mean "
         + note, "derived"),
        ("lb1018_k3aug_in_top50_band", float(min(raw) <= N["width1024_k3a_mean"] <= max(raw)), "flag",
         "1 = ARC's augmented K=3 mean at 1024 x 16 lies inside the raw range of entries ranked 1-50", "derived"),
    ]


def width_data(N):
    """Per width: (networks, mean, interval) for K=3 aug. and K=4 full, and the least-squares fit of log mean on log n."""
    d = {}
    for key in ("k3a", "k4"):
        for w in WIDTHS:
            nets = [N[f"width{w}_seed{s}_{key}"] for s in SEEDS[w]]
            d[(w, key)] = (nets, N[f"width{w}_{key}_mean"], N[f"width{w}_{key}_ci_lo"], N[f"width{w}_{key}_ci_hi"])
        slope, icpt = np.polyfit(np.log(WIDTHS), np.log([d[(w, key)][1] for w in WIDTHS]), 1)
        p = N["width_exponent_k3a" if key == "k3a" else "width_exponent_k4"]
        assert abs(-slope - p) < 2e-4, (key, slope, p)     # the same fit the report quotes
        d[key] = (slope, icpt)
    return d


def budget(N):
    """The units ladder of the complete chain from layer 4 on network 0: (label, counted units, tie units, ratio)."""
    return [
        ("complete chain", N["red_full_s4_units"], N["red_tie_c60549"], N["red_tie_gap_c60549"]),
        ("best deletion", N["red_n0_window_s4_b9_treeall_units"], N["red_tie_w20586"], N["red_tie_gap_w20586"]),
        ("non-tree as rank-4 pairs", N["red2_w9_n4_units"], N["red2_w9_n4_tie"], N["red2_w9_n4_gap"]),
    ]


# ---------------------------------------------------------------- shared pieces
def fmt(v):
    return f"{v:,.0f}"


def xspan(ax, t, pad=1.4):
    """Data-x extent of a placed text with its whole box, widened by pad pt."""
    r = ax.figure.canvas.get_renderer()
    if t.get_bbox_patch() is not None:
        t.update_bbox_position_size(r)
        b = t.get_bbox_patch().get_window_extent(r)
    else:
        b = t.get_window_extent(r)
    p = pad * ax.figure.dpi / 72
    inv = ax.transData.inverted()
    return inv.transform((b.x0 - p, b.y0))[0], inv.transform((b.x1 + p, b.y0))[0]


def ratio_strip(ax, N, gold_last=False, widths=WIDTHS, lift=0.135, shift=None, pad=0.2):
    """K=3 aug. / K=4 at each width as a row of chips under the data, lift in above the axis, each chip's face as deep
    as its ratio; shift moves a chip off its width by a factor (where neighbours would touch)."""
    from matplotlib.transforms import blended_transform_factory
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    y = lift / cel.box_of(ax)[3]
    blue = cel.shades("k4")
    lo, hi = np.log(3.0), np.log(12.5)
    out = []
    for w in widths:
        r = N[f"width{w}_ratio"]
        last = w == widths[-1]
        x = w * (shift or {}).get(w, 1.05 if last else 1.0)
        if last and gold_last:
            t = cel.chip(ax, x, y, f"{r:.1f}" + X, "focus", size=8.4, weight="bold", level=1, ha="center",
                         pad=pad, transform=tr)
        else:
            j = cel.depth_of(np.log(r), lo, hi)
            face = blue[j]
            t = cel.chip(ax, x, y, f"{r:.1f}" + X, "k4", size=8.4 if last else FSS,
                         weight="bold" if last else "normal", ha="center", pad=pad, face=face, transform=tr,
                         color=cel.PAPER if j >= 3 else cel.deeper(cel.line("k4"), 0.18))
        out.append(t)
    return out


def draw_width(ax, N, entries, band=False, xmax=2900, ylo=3.2e-10, strip=True, chips=True, gold_last=False,
               widths=WIDTHS, xmin=212, shift=None):
    """Raw MSE against width at depth 16: every network a small disc, means as large marks, 95% t tubes; a row of
    ratio chips under the data; band=True lays every leaderboard entry ranked 1-50 as a barcode behind 1024."""
    D = width_data(N)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylo, 4.5e-6)
    jit = {0: 0.975, 1: 1.0, 2: 1.025, 3: 0.99, 4: 1.012}
    for key, role in (("k3a", "k3"), ("k4", "k4")):
        sh = 1.05 if key == "k4" else 0.95
        for w in WIDTHS:
            nets, m, lo, hi = D[(w, key)]
            tube(ax, w * sh, lo, hi, role, width=2.6, z=2.4)
            cel.discs(ax, [w * sh * jit[i] for i in range(len(nets))], nets, role, 0, d=2.8, lw=0.4, z=4.2,
                      spark=False)
        means = [D[(w, key)][1] for w in WIDTHS]
        cel.cline(ax, [w * sh for w in WIDTHS], means, role, 1, lw=1.0, rim=2.2, z=3, inner=cel.light(role))
        if key == "k4":
            cel.discs(ax, [w * sh for w in WIDTHS], means, role, 1, d=6.0, lw=0.55, z=5, cast=True)
        else:
            cel.diamonds(ax, [w * sh for w in WIDTHS], means, role, 1, d=6.8, lw=0.55, z=5)
    if chips:
        a3, a4 = N["width_exponent_k3a"], N["width_exponent_k4"]
        cel.chip(ax, 300, 3.0e-6, rf"ARC's K=3 aug.  $n^{{-{a3:.2f}}}$", "k3", size=FSS)
        cel.chip(ax, 236, 3.4e-9, rf"full K=4  $n^{{-{a4:.2f}}}$", "k4", size=FSS)
    strip_texts = ratio_strip(ax, N, gold_last=gold_last, widths=widths, shift=shift) if strip else []
    if band:
        top = sorted(e["raw"] for e in entries if e["rank"] <= 50)
        x0, x1 = 1024 * 0.80, 1024 * 1.80
        r = Rectangle((x0, min(top)), x1 - x0, max(top) - min(top), fc=HUES["gray"][3], ec=cel.line("k3"), lw=0.5,
                      zorder=1.5)
        ax.add_patch(cel.drop(r, 1.0, -1.0))
        band_ticks(ax, x0, x1, top, z=1.6)
        say(ax, x1, max(top), "leaderboard\nranks 1–50", ha="right", va="bottom", size=FSS, color=MUTED, linespacing=1.0,
            transform=nudge(ax, 0, 2.5))
        k4 = N["width1024_k4_mean"]
        xb = 1024 * 1.60
        ax.plot([1024 * 1.05 * 1.08, xb], [k4, k4], color=cel.line("focus"), lw=0.45, ls=(0, (1.0, 1.2)), zorder=3.1)
        ax.annotate("", xy=(xb, k4 * 1.05), xytext=(xb, min(top) / 1.02), zorder=3.3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color=cel.line("focus"),
                                    lw=0.8, shrinkA=0, shrinkB=0))
        cel.chip(ax, xb, np.sqrt(k4 * min(top)), f"{N['lb1018_rank1_raw_over_k4']:.1f}" + X,
                 "focus", size=9.5, weight="bold", level=1, ha="left", transform=nudge(ax, 4.0, 0), pad=0.22)
    ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cel.axes_style(ax)
    log_ruler(ax, "y", -10, -5)
    ax.set_yticks([1e-9, 1e-8, 1e-7, 1e-6])
    return strip_texts


def band_ticks(ax, x0, x1, raws, z=3.2, color=None):
    """Every leaderboard entry as one tick across a column (a barcode on the raw axis)."""
    c = color or cel.deeper(cel.line("k3"), 0.04)
    for r in raws:
        ax.plot([x0, x1], [r, r], color=c, lw=0.42, zorder=z, solid_capstyle="butt", gid="texture")


def draw_loupe(ax, N, entries, ylim=(1.12e-9, 3.4e-8), names=("ARC's K=3 aug.", "leaderboard\nranks 1–50",
                                                             "full K=4"), rank1=True, yticks=True, lead="focus"):
    """The 1024 x 16 column magnified: ARC's augmented K=3 and full K=4 on official networks 0-4 and every leaderboard
    entry ranked 1-50 between them, on one raw axis; both factors land on the K=4 mean, the one from rank 1 in the
    role `lead` at the emphasized level."""
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.80, 2.45)
    k3 = [N[f"width1024_seed{s}_k3a"] for s in range(5)]
    k4 = [N[f"width1024_seed{s}_k4"] for s in range(5)]
    top = sorted((e for e in entries if e["rank"] <= 50), key=lambda e: (e["rank"], e["raw"]))
    for x in (0.0, 1.0, 2.0):
        ax.add_patch(Rectangle((x - 0.36, ylim[0]), 0.72, ylim[1] - ylim[0], fc=HUES["gray"][3], ec="none",
                               zorder=0.3, gid="texture"))
    xs = 1.0 + (np.arange(len(top)) - (len(top) - 1) / 2) / len(top) * 0.60
    cel.discs(ax, xs, [e["raw"] for e in top], "k3", 0, d=3.4, lw=0.4, z=4.0, spark=False,
              colors=(cel.wash("k3"), cel.light("k3"), cel.line("k3")))
    r1 = top[0]
    cel.discs(ax, [xs[0]], [r1["raw"]], "k3", 1, d=4.8, lw=0.5, z=4.3, spark=False)
    if rank1 == "below":
        say(ax, xs[0], r1["raw"], "rank 1", ha="left", va="top", size=FSS, color=MUTED, transform=nudge(ax, -2, -4))
    elif rank1:
        say(ax, xs[0], r1["raw"], "rank 1", ha="right", va="center", size=FSS, color=MUTED, transform=nudge(ax, -4, 0))
    for vals, x, role in ((k3, 0.0, "k3"), (k4, 2.0, "k4")):
        px = x + (np.arange(5) - 2) * 0.13
        m = float(np.mean(vals))
        ax.plot([x - 0.33, x + 0.33], [m, m], color=cel.line(role), lw=0.9, zorder=3.6, solid_capstyle="butt")
        if role == "k4":
            cel.discs(ax, px, vals, role, 1, d=5.4, lw=0.5, z=5, cast=True)
        else:
            cel.diamonds(ax, px, vals, role, 1, d=6.2, lw=0.5, z=5)
    k4m, k3m = float(np.mean(k4)), float(np.mean(k3))
    xl, xr = -0.62, 1.50
    ax.plot([xl, 2.36], [k4m] * 2, color=cel.line("k4"), lw=0.45, ls=(0, (1.0, 1.2)), zorder=3.1)
    ax.plot([xl, -0.34], [k3m] * 2, color=cel.line("k4"), lw=0.45, ls=(0, (1.0, 1.2)), zorder=3.1)
    lc = cel.line(lead) if lead == "focus" else cel.deeper(cel.line(lead), 0.12)
    ax.plot([xs[0] + 0.03, xr], [r1["raw"]] * 2, color=lc, lw=0.45, ls=(0, (1.0, 1.2)), zorder=3.1)
    for x, a, c, lw in ((xl, k3m, cel.line("k4"), 0.75), (xr, r1["raw"], lc, 0.85)):
        ax.annotate("", xy=(x, k4m * 1.05), xytext=(x, a / 1.03), zorder=3.3,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.32,head_width=0.16", color=c, lw=lw,
                                    shrinkA=0, shrinkB=0))
    cel.chip(ax, xl, np.sqrt(k4m * k3m), f"{N['width1024_ratio']:.1f}" + X, "k4", size=8.4, weight="bold", ha="left",
             transform=nudge(ax, 4.0, 0), pad=0.2)
    cel.chip(ax, xr, np.sqrt(k4m * r1["raw"]), f"{N['lb1018_rank1_raw_over_k4']:.1f}" + X, lead, size=9.5,
             weight="bold", level=1, ha="left", transform=nudge(ax, 4.0, 0), pad=0.22)
    ax.set_xticks([0, 1, 2], list(names))
    ax.tick_params(axis="x", length=0, pad=3.0)
    for lab, c in zip(ax.get_xticklabels(), (INK, MUTED, cel.deeper(cel.line("k4"), 0.16))):
        lab.set_color(c)
        lab.set_linespacing(1.0)
    cel.axes_style(ax)
    log_ruler(ax, "y", -10, -5)
    if yticks:
        ax.set_yticks([2e-9, 5e-9, 1e-8, 2e-8], [r"2$\times$10$^{-9}$", r"5$\times$10$^{-9}$", r"10$^{-8}$",
                                                r"2$\times$10$^{-8}$"])
    else:
        ax.set_yticks([])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return k4m, r1


def draw_ladder(ax, N, labels=("complete chain from layer 4", "best deletion", "non-tree births as rank-4 pairs"),
                left=0.30, right=0.42, budget_label=True):
    """Counted units against the units at which each configuration ties scored chain B, log units on a linear axis:
    a ring where it would tie, a disc at what it counts, the factor between them cut into the bar."""
    L = budget(N)
    lo_d, hi_d = np.log10(min(t for _, _, t, _ in L)), np.log10(max(u for _, u, _, _ in L))
    aw = cel.box_of(ax)[2]
    per = (aw - left - right) / (hi_d - lo_d)
    ax.set_xlim(lo_d - left / per, hi_d + right / per)
    ax.set_ylim(-0.72, 2.72)
    roles = ("k4", "k4", "focus")
    avoid = []
    for (lab, units, tie, gap), y, role, name in zip(L, (2.0, 1.0, 0.0), roles, labels):
        lo, hi = np.log10(tie), np.log10(units)
        lvl = 1 if role == "focus" else 0
        c = cel.chip(ax, (lo + hi) / 2, y, f"{gap:.0f}" + X, role, size=8.4, weight="bold", level=lvl, ha="center",
                     va="center", pad=0.18)
        a, b = xspan(ax, c, pad=0.8)
        for u, v in ((lo, a), (b, hi)):
            if v > u:
                cel.hbar(ax, y, u, v, role, lvl, h=3.0, lw=0.5, z=3, shadow=0.42, cast=0.9)
        cel.discs(ax, [lo], [y], "k3", 0, d=6.0, lw=0.6, z=4.4, spark=False,
                  colors=(cel.PAPER, cel.PAPER, cel.deeper(cel.line(role), 0.06)))
        cel.discs(ax, [hi], [y], role, 1, d=6.4, lw=0.55, z=5, cast=True)
        t1 = say(ax, hi, y, fmt(units), ha="left", va="center", size=FSS, color=cel.deeper(cel.line(role), 0.16),
                 transform=nudge(ax, 5.0, 0))
        t2 = say(ax, lo, y, fmt(tie), ha="right", va="center", size=FSS, color=MUTED, transform=nudge(ax, -5.0, 0))
        avoid += [c, t1, t2]
        if name:
            avoid.append(say(ax, lo, y + 0.33, name, ha="left", va="bottom", size=FSS, color=INK,
                             transform=nudge(ax, -3.5, 0)))
    b = np.log10(1024)
    if budget_label:
        avoid.append(say(ax, b, -0.52, "budget", ha="left", va="center", size=FSS, color=MUTED,
                         transform=nudge(ax, 2.5, 0)))
    rule(ax, b, -0.72, 2.72, avoid=avoid, pad=1.2, vertical=True, color=cel.LINE, lw=0.7, zorder=1.2)
    x0, x1 = ax.get_xlim()
    ticks = [e for e in range(1, 7) if x0 <= e <= x1]
    ax.set_xticks(ticks, [f"10$^{e}$" for e in ticks])
    ax.set_yticks([])
    pos, lv = [], []
    for e in range(1, 7):
        for m in range(1, 10):
            v = np.log10(m * 10.0 ** e)
            if x0 <= v <= x1:
                pos.append(v)
                lv.append(0 if m == 1 else (1 if m in (2, 5) else 2))
    cel.axes_style(ax, left=False)
    ax.tick_params(axis="x", length=0, pad=4.0)
    cel.ruler(ax, pos, lv)
    return L


def share_card(fig, x, y, w, h, N, letter="b"):
    """The 60,549 counted units of the complete chain from layer 4 as one strip cut by birth family (linear shares)."""
    cel.sheet(fig, x, y, w, h, drop_pt=1.8, stack=2.0)
    cel.badge(fig, x - 0.05, y + h - 0.075, letter, "gray", 0, gid="tab")
    tr = fig.dpi_scale_trans
    fw, fh = fig.get_size_inches()
    sx, sy, sw, sh = x + 0.13, y + 0.10, w - 0.26, 0.105
    parts = [("C·κ3 births", N["red_share_ck3"], cel.mid("k4")),
             ("non-tree births", N["red_share_nontree"], cel.line("k4")),
             ("rest", 1 - N["red_share_ck3"] - N["red_share_nontree"], cel.light("k3"))]
    fig.add_artist(cel.drop(Rectangle((sx, sy), sw, sh, transform=tr, fc=cel.PAPER, ec="none", zorder=7.9), 1.1, -1.1))
    xx = sx
    for k, (lab, s, fc) in enumerate(parts):
        fig.add_artist(Rectangle((xx, sy), sw * s, sh, transform=tr, fc=fc, ec="none", zorder=8))
        if k < 2:
            fig.text((xx + 0.02) / fw, (sy + sh + 0.035) / fh, f"{lab} {s * 100:.0f}%", fontsize=FSS, ha="left",
                     va="bottom", color=cel.deeper(cel.line("k4"), 0.16))
            fig.add_artist(Rectangle((xx + sw * s - 0.004, sy), 0.008, sh, transform=tr, fc=cel.PAPER, ec="none",
                                     zorder=8.05))
        xx += sw * s
    fig.add_artist(Rectangle((sx, sy), sw, sh, transform=tr, fc="none", ec=cel.deeper(cel.line("k4"), 0.10), lw=0.5,
                             zorder=8.1))
    fig.text(sx / fw, (y + h - 0.075) / fh, f"{fmt(N['red_full_s4_units'])} units from layer 4",
             fontsize=FSS, ha="left", va="top", color=INK)
    fig.text((sx + sw) / fw, (y + h - 0.075) / fh, f"{(N['red_share_ck3'] + N['red_share_nontree']) * 100:.0f}% "
             "in two births", fontsize=FSS, ha="right", va="top", color=cel.deeper(cel.line("k4"), 0.16), **cel.BOLD)


# ---------------------------------------------------------------- variants
def variant_a(N, entries):
    H = 3.02
    fig = plt.figure(figsize=(W, H))
    ax = cel.axes(fig, 0.52, 0.40, 3.98, 2.70)
    draw_width(ax, N, entries, band=True)
    ax.set_xlabel(r"width $n$, depth 16", labelpad=1.5)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    cel.ylab(ax, "final-layer raw MSE", dx=-0.02).set_y(1.0 + 12.0 / 72 / cel.box_of(ax)[3])
    share_card(fig, 4.36, 2.20, 2.26, 0.58, N, letter="b")
    bx = cel.axes(fig, 4.36, 0.40, 6.62, 1.98)
    draw_ladder(bx, N)
    bx.set_xlabel("units per network, network 0", labelpad=1.5)
    cel.sheet_panel(bx, letter="c", header=0.0, stack=2.0, which="0101", peg_at=0.80)
    return fig, "fig0_hero_A"


def variant_b(N, entries):
    H = 2.92
    fig = plt.figure(figsize=(W, H))
    ax = cel.axes(fig, 0.52, 0.40, 3.80, 2.60)
    draw_width(ax, N, entries, band=False, xmax=1450)
    ax.set_xlabel(r"width $n$, depth 16", labelpad=1.5)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    cel.ylab(ax, "final-layer raw MSE", dx=-0.02).set_y(1.0 + 12.0 / 72 / cel.box_of(ax)[3])
    ylim = (1.12e-9, 3.4e-8)
    cel.view(ax, 1024 * 0.84, 1024 * 1.24, ylim[0], ylim[1], role="gray")
    bx = cel.axes(fig, 4.62, 0.40, 6.62, 2.60)
    draw_loupe(bx, N, entries, ylim)
    cel.sheet_panel(bx, letter="b", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    cel.ylab(bx, r"at 1024 $\times$ 16, magnified", dx=-0.02).set_y(1.0 + 12.0 / 72 / cel.box_of(bx)[3])
    return fig, "fig0_hero_B"


def number_card(fig, x, y, w, h, big, lines, role, index=None):
    """A cel card: the reading as a large numeral (big: text, or (text, role) pieces set one after another), two or
    three lines saying what it is, and the data it reads drawn below on its own sheet; index rings the number of the
    first-page finding the card belongs to in the upper right corner."""
    cel.sheet(fig, x, y, w, h, drop_pt=2.0, stack=2.0)
    fw, fh = fig.get_size_inches()
    r = fig.canvas.get_renderer()
    cx = x + 0.12
    for text, rl in ([(big, role)] if isinstance(big, str) else big):
        t = fig.text(cx / fw, (y + h - 0.10) / fh, text, fontsize=23, ha="left", va="top",
                     color=cel.deeper(cel.ramp(rl, 1)[2], 0.10), **cel.BOLD)
        cx = t.get_window_extent(r).x1 / fig.dpi
    fig.text((x + 0.12) / fw, (y + h - 0.47) / fh, lines, fontsize=FS, ha="left", va="top", color=INK,
             linespacing=1.15)
    if index is not None:
        cel.numeral(fig, x + w - 0.17, y + h - 0.20, index, "gray", size=FSS, r=5.0)


def variant_c(N, entries):
    H = 3.05
    fig = plt.figure(figsize=(W, H))
    cw, gap, x0, y0, ch = 2.08, 0.17, 0.10, 0.10, 2.86
    xs = [x0 + k * (cw + gap) for k in range(3)]
    top = y0 + 1.86
    number_card(fig, xs[0], y0, cw, ch, f"{N['width1024_ratio']:.1f}" + X,
                "lower raw error than ARC's augmented\nK=3 at 1024 × 16; the gap opens with\n"
                "width at the 3 and 4 of $c_K(L/n)^K$", "k4", index=1)
    number_card(fig, xs[1], y0, cw, ch, f"{N['lb1018_rank1_raw_over_k4']:.1f}" + X,
                "below rank 1's leaderboard raw error\n(its 50 networks, same shape); full\nK=4 runs beyond the budget",
                "k4", index=2)
    number_card(fig, xs[2], y0, cw, ch, [(f"{N['red_tie_gap_w20586']:.0f}" + X + r"$\,\rightarrow\,$", "k4"),
                                         (f"{N['red2_w9_n4_gap']:.0f}" + X, "focus")],
                "counted units over the units at which\nit ties our scored chain: best deletion,\n"
                "then non-tree births as rank-4 pairs", "k4", index=3)
    ax = cel.axes(fig, xs[0] + 0.32, y0 + 0.36, xs[0] + cw - 0.10, top)
    draw_width(ax, N, entries, band=False, xmin=232, xmax=1450, ylo=2.3e-10, chips=False,
               shift={256: 1.045, 384: 0.98, 512: 1.0, 1024: 1.0})
    ax.set_xticks(WIDTHS, [str(w) for w in WIDTHS])
    ax.set_yticks([1e-9, 1e-7], ["10$^{-9}$", "10$^{-7}$"])
    ax.set_xlabel("width, depth 16", labelpad=1.0)
    cel.chip(ax, 760, 4.6e-7, rf"K=3 aug.  $n^{{-{N['width_exponent_k3a']:.2f}}}$", "k3", size=FSS, ha="center")
    cel.chip(ax, 246, 4.4e-9, rf"full K=4  $n^{{-{N['width_exponent_k4']:.2f}}}$", "k4", size=FSS)
    bx = cel.axes(fig, xs[1] + 0.52, y0 + 0.36, xs[1] + cw - 0.12, top)
    draw_loupe(bx, N, entries, (1.12e-9, 3.4e-8), names=("K=3", "ranks\n1–50", "K=4"), rank1="below", lead="k4")
    cx = cel.axes(fig, xs[2] + 0.12, y0 + 0.36, xs[2] + cw - 0.12, top)
    draw_ladder(cx, N, labels=("complete", "deletion", "rank-4 pairs"), budget_label=True)
    cx.set_xlabel("units, network 0", labelpad=1.0)
    for a_ in (ax, bx, cx):
        cel.sheet_panel(a_, header=0.0, stack=0.0, which="0000", drop_pt=1.2)
    return fig, "fig0_hero_C"


def glass_rows(fig, N, entries, ytop, nets_k=(0, 1, 2, 3, 4)):
    """Figure 3's glass plates, larger: ARC's augmented K=3 and full K=4 run through the 16 layers of the 1024 x 16
    official networks, each beam as long as its final-layer raw MSE on one log ruler, every leaderboard entry ranked
    1-50 a tick on the same ruler between them; both factors are arrows that land on the K=4 mean. ytop: screen inches
    of the top beam. Returns the canvas and the ruler's lowest text edge (inches)."""
    import visual_figures as vf
    fx = vf.canvas(fig)
    x1, dx, sx, sy = 1.40, 0.122, 0.46, 0.30
    xe = x1 + 15 * dx + sx + 0.07
    xr1, gap = 6.64, 0.30
    with vf.glass_geometry(X1=x1, DX=dx, SX=sx, SY=sy, XE=xe, XR1=xr1, KW=0.0084):
        y3 = ytop - sy / 2                      # rail heights (mid-depth); a beam's screen height is y + sy/2
        y4 = y3 - 2 * gap
        vf.glass_stack(fx, y4 - 0.21, y3 + 0.16)
        lo, hi = 6.0e-10, 3.4e-8
        span = np.log10(hi) - np.log10(lo)
        xr = lambda v: xe + (np.log10(v) - np.log10(lo)) / span * (xr1 - xe)
        xg = lambda a, b: xr(np.sqrt(a * b))
        k3 = [N[f"width1024_seed{s}_k3a"] for s in nets_k]
        k4 = [N[f"width1024_seed{s}_k4"] for s in nets_k]
        m3, m4 = N["width1024_k3a_mean"], N["width1024_k4_mean"]
        s3, s4 = vf._p(16, y3)[1], vf._p(16, y4)[1]
        sm = (s3 + s4) / 2
        texts = []
        for y, lab in ((y3, "ARC's augmented K=3"), (y4, "full K=4")):
            texts.append(fx.text(x1 - 0.06, vf._p(1, y)[1], lab, ha="right", va="center", fontsize=FS, color=INK))
        vf._rail(fx, y3, 1, 16, "k3", lw=1.0)
        fx.plot([vf._p(16.5, y3)[0], xe], [s3, s3], color=cel.line("k3"), lw=1.0, zorder=vf._zp(16) + 0.025)
        vf._body(fx, y4, 1, 99)
        for s, m, nets, role in ((s3, m3, k3, "k3"), (s4, m4, k4, "k4")):
            for w, a, c in ((7.4, 0.40, cel.wash(role)), (4.4, 0.75, cel.light(role)), (1.2, 1.0, cel.mid(role))):
                fx.plot([xe, xr(m)], [s, s], color=c, lw=w, alpha=a, solid_capstyle="butt", zorder=2.95,
                        gid=None if w == 1.2 else "texture")
            # the five networks ride 2.5 pt clear of the mean disc's rim
            cel.discs(fx, [xr(v) for v in nets], [s + 0.085 + 2.5 / 72] * len(nets), role, 0, d=2.9, lw=0.4, z=5.6,
                      spark=False)
            cel.discs(fx, [xr(m)], [s], role, 1, d=7.0, lw=0.6, z=6, cast=True)
            texts.append(fx.text(xr(m) + 0.075, s, vf.sci(m, 3), ha="left", va="center", fontsize=FSS, zorder=7,
                                 color=MUTED if role == "k3" else cel.deeper(cel.line("k4"), 0.16)))
        # every leaderboard entry ranked 1-50, one tick each, rank 1 a disc
        top = sorted((e for e in entries if e["rank"] <= 50), key=lambda e: (e["rank"], e["raw"]))
        raws = [e["raw"] for e in top]
        fx.add_patch(cel.drop(Rectangle((xr(min(raws)), sm - 0.05), xr(max(raws)) - xr(min(raws)), 0.10,
                                        fc=HUES["gray"][3], ec=cel.light("k3"), lw=0.4, zorder=2.6), 0.9, -0.9))
        for r in raws:
            fx.plot([xr(r)] * 2, [sm - 0.04, sm + 0.04], color=cel.deeper(cel.line("k3"), 0.04), lw=0.45,
                    zorder=2.7, solid_capstyle="butt", gid="texture")
        r1 = top[0]["raw"]
        cel.discs(fx, [xr(r1)], [sm], "k3", 1, d=5.0, lw=0.5, z=5.5, spark=False)
        texts.append(fx.text(xr(r1), sm + 0.05, "rank 1", ha="center", va="bottom", fontsize=FSS, color=MUTED))
        texts.append(fx.text(xr(r1) - 0.075, sm, "leaderboard, ranks 1–50", ha="right", va="center", fontsize=FSS,
                             color=MUTED))
        # the two factors, arrows on the log ruler that land on the K=4 mean, each broken around its chip
        lc = cel.deeper(cel.line("k4"), 0.10)
        ya, yb = s4 - 0.15, s4 - (0.40 if matplotlib.__version__.startswith("3.10.") else 0.36)
        chips = [cel.chip(fx, xg(r1, m4), ya, f"{N['lb1018_rank1_raw_over_k4']:.1f}" + X, "k4", size=10, weight="bold",
                          level=1, ha="center", pad=0.22),
                 cel.chip(fx, xg(m3, m4), yb, f"{N['width1024_ratio']:.1f}" + X, "k4", size=10, weight="bold", level=1,
                          ha="center", pad=0.22)]
        fig.canvas.draw()
        for x0, y, c in ((xr(r1), ya, chips[0]), (xr(m3), yb, chips[1])):
            a, b = xspan(fx, c, 1.0)
            fx.plot([b, x0], [y, y], color=lc, lw=0.85, solid_capstyle="butt", zorder=6.5)
            fx.annotate("", xy=(xr(m4) + 0.02, y), xytext=(a, y), zorder=6.5,
                        arrowprops=dict(arrowstyle="-|>,head_length=0.40,head_width=0.19", color=lc, lw=0.85,
                                        shrinkA=0, shrinkB=0))
        dot = dict(color=lc, lw=0.45, ls=(0, (1.0, 1.2)), zorder=6.4)
        vf.rule(fx, xr(r1), ya, sm - 0.035, avoid=texts, **dot)
        vf.rule(fx, xr(m4), yb, s4 - 0.05, avoid=texts, **dot)
        vf.rule(fx, xr(m3), yb, s3 - 0.05, avoid=texts, **dot)
        yr = yb - 0.13
        tk = [(m * 10.0 ** e, 0 if m == 1 else (1 if m in (2, 5) else 2)) for e in range(-10, -6) for m in range(1, 10)
              if lo <= m * 10.0 ** e <= hi]
        lab = [(10.0 ** -9, "10$^{-9}$"), (2e-9, r"2$\times$10$^{-9}$"), (5e-9, r"5$\times$10$^{-9}$"),
               (1e-8, "10$^{-8}$"), (2e-8, r"2$\times$10$^{-8}$")]
        vf._ruler(fx, yr, tk, lab, xr, r"final-layer raw MSE at 1024 $\times$ 16")
        for l in (1, 4, 8, 12, 16):
            x, yy = vf._p(l, y3 + 0.16, 1.0)
            fx.text(x, yy + 0.03, str(l), ha="center", va="bottom", fontsize=FSS, color=MUTED)
        x, yy = vf._p(1, y3 + 0.16, 1.0)
        fx.text(x - 0.09, yy + 0.03, "layer", ha="right", va="bottom", fontsize=FSS, color=MUTED)
    return fx, yr - 0.205 - 0.10


def variant_d(N, entries):
    """The first page's figure since v22g: (a) the glass plates of Figure 3, larger, over (b) the width law and (c) the
    units ladder, side by side and unframed."""
    H = 4.04
    fig = plt.figure(figsize=(W, H))
    fx, ylow = glass_rows(fig, N, entries, ytop=3.44)
    cel.badge(fig, 0.06, 3.80, "a", "gray", 0, gid="tab")
    y1 = ylow - 0.20
    ax = cel.axes(fig, 0.58, 0.36, 3.18, y1)
    draw_width(ax, N, entries, band=False, xmin=232, xmax=1450, ylo=2.6e-10, chips=False,
               shift={256: 1.045, 384: 0.98, 512: 1.0, 1024: 1.0})
    ax.set_xlabel(r"width $n$, depth 16", labelpad=1.5)
    cel.ylab(ax, "final-layer raw MSE", dx=-0.02).set_y(1.0 + 4.0 / 72 / cel.box_of(ax)[3])
    cel.chip(ax, 640, 6.5e-7, rf"ARC's K=3 aug.  $n^{{-{N['width_exponent_k3a']:.2f}}}$", "k3", size=FSS, ha="left")
    cel.chip(ax, 250, 4.6e-9, rf"full K=4  $n^{{-{N['width_exponent_k4']:.2f}}}$", "k4", size=FSS)
    cel.badge(fig, 0.06, y1 + 0.10, "b", "gray", 0, gid="tab")
    cx = cel.axes(fig, 3.78, 0.36, 6.62, y1)
    draw_ladder(cx, N, labels=("complete chain from layer 4", "best deletion", "non-tree births as rank-4 pairs"))
    cx.set_xlabel("units per network, official network 0", labelpad=1.5)
    cel.badge(fig, 3.62, y1 + 0.10, "c", "gray", 0, gid="tab")
    return fig, "fig0_hero_D"


VARIANTS = {"A": variant_a, "B": variant_b, "C": variant_c, "D": variant_d}
KEEP = ("D",)      # the variant the report includes; make_figures.py renders only these


def render(names=KEEP, out=HERE / "figures", N=None):
    setup()
    N = dict(N or numbers())
    entries = board()
    for key, value, *_ in rows(N, entries):
        N[key] = value
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig, stem = VARIANTS[name](N, entries)
        figcheck.report(fig, stem)
        fig.savefig(out / f"{stem}.pdf")
        fig.savefig(out / f"{stem}.png", dpi=600)
        plt.close(fig)


if __name__ == "__main__":
    args = sys.argv[1:]
    out = HERE / "figures"
    if "--out" in args:
        i = args.index("--out")
        out = Path(args[i + 1])
        del args[i:i + 2]
    render(tuple(args) or KEEP, out)
