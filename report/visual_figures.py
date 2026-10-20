"""The four report figures, drawn from the numbers make_figures.py computes, each in its own main form:
Figure 1 a field (the width x depth plane, n/L as its contours), Figure 2 a flow (births to the layers that read them),
Figure 3 a spatial chain (16 glass layer plates in oblique view) and Figure 4 a frontier (cost against accuracy) with
paired traces. One role palette (palette.py) and one cel layer (cel.py) run through all four.

Figures go into the PDF at the text width (6.81 in) and keep their size: every readable label is 6.8 pt or more.
Blue is ARC's full K=4, gray ARC's augmented K=3 and baselines, iris our scored chain B, gold the two or three
readings a figure points at.
"""
from contextlib import contextmanager
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from matplotlib.transforms import offset_copy

import cel
import figcheck
from palette import HUES, INK, MUTED, mix

W = 6.81
FS, FSS = 7.2, 6.8


def setup():
    matplotlib.rcParams.update({
        "font.family": "STIX Two Text", "mathtext.fontset": "stix", "font.size": FS, "axes.labelsize": FS,
        "xtick.labelsize": FSS, "ytick.labelsize": FSS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": cel.LINE, "ytick.color": cel.LINE, "pdf.fonttype": 42, "savefig.dpi": 600,
        "axes.linewidth": 0.7, "axes.unicode_minus": True, "figure.facecolor": "white",
    })


def say(ax, x, y, s, size=FSS, color=INK, z=9, **kw):
    return ax.text(x, y, s, fontsize=size, color=color, zorder=z, **kw)


def save(fig, output, name):
    figcheck.report(fig, name)
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=600)
    plt.close(fig)


def inch(ax, x, y):
    return ax.figure.dpi_scale_trans.inverted().transform(ax.transData.transform((x, y)))


def canvas(fig, z=0):
    """An axes over the whole figure in inches, for objects drawn in figure space."""
    fx = fig.add_axes([0, 0, 1, 1], zorder=z)
    w, h = fig.get_size_inches()
    fx.set_xlim(0, w)
    fx.set_ylim(0, h)
    fx.axis("off")
    fx.patch.set_alpha(0)
    return fx


def tube(ax, x, lo, hi, role, width=4.0, z=3, level=0):
    """A vertical interval as a cel tube on any axis scale: hard shadow, flat body, edge lines, a highlight on the lit
    side and end caps. width in pt."""
    f, s, l = cel.ramp(role, level)
    sh = offset_copy(ax.transData, ax.figure, x=0.9, y=-0.9, units="points")
    ax.plot([x, x], [lo, hi], color=cel.SHADE, lw=width, solid_capstyle="butt", transform=sh, zorder=z - 0.05)
    ax.plot([x, x], [lo, hi], color=l, lw=width + 1.0, solid_capstyle="butt", zorder=z)
    ax.plot([x, x], [lo, hi], color=f, lw=width, solid_capstyle="butt", zorder=z + 0.01)
    ax.plot([x, x], [lo, hi], color=s, lw=width * 0.35, solid_capstyle="butt", zorder=z + 0.02,
            transform=offset_copy(ax.transData, ax.figure, x=0.32 * width, y=0, units="points"))
    ax.plot([x, x], [lo, hi], color=cel.PAPER, lw=0.45, solid_capstyle="butt", zorder=z + 0.03,
            transform=offset_copy(ax.transData, ax.figure, x=-0.3 * width, y=0, units="points"))
    ax.plot([x, x], [lo, hi], ls="none", marker="_", ms=width + 3.2, mew=0.6, color=l, zorder=z + 0.04)


def log_ruler(ax, axis="y", lo=-10, hi=-5):
    """Ruler marks on a log axis: decades major, 2 and 5 half, the rest fine."""
    pos, lv = [], []
    for e in range(lo, hi + 1):
        for m in range(1, 10):
            pos.append(m * 10.0 ** e)
            lv.append(0 if m == 1 else (1 if m in (2, 5) else 2))
    a, b = sorted(ax.get_ylim() if axis == "y" else ax.get_xlim())
    keep = [(p, v) for p, v in zip(pos, lv) if a <= p <= b]
    pos, lv = [p for p, _ in keep], [v for _, v in keep]
    ax.tick_params(axis=axis, length=0, pad=4.2)
    cel.ruler(ax, pos, lv, axis=axis)
    cel.ruler(ax, pos, lv, axis=axis, top=True, color=cel.TICK, length=(2.6, 1.8, 1.1), lw=0.35, z=1.65)


def sci(v, nd=2):
    m, e = f"{v:.{nd - 1}e}".split("e")
    return rf"{m}$\times$10$^{{{int(e)}}}$"


def nudge(ax, dx=0.0, dy=0.0):
    """Data transform shifted by dx, dy pt: a label set a fixed distance from its mark on any axis scale."""
    return offset_copy(ax.transData, ax.figure, x=dx, y=dy, units="points")


def _boxes(ax, texts, pad):
    """Display boxes (x0, x1, y0, y1) of placed texts (a chip counts with its whole box), widened by pad pt."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    p = pad * fig.dpi / 72
    out = []
    for t in texts:
        patch = t.get_bbox_patch()
        if patch is not None:
            t.update_bbox_position_size(r)
        b = (patch if patch is not None else t).get_window_extent(r)
        out.append((b.x0 - p, b.x1 + p, b.y0 - p, b.y1 + p))
    return out


def _pieces(lo, hi, holes, shortest):
    """What is left of lo..hi after removing the holes; pieces shorter than `shortest` are dropped."""
    keep = [(lo, hi)]
    for a, b in holes:
        keep = [q for u, v in keep for q in (((u, v),) if b <= u or a >= v else ((u, min(a, v)), (max(b, u), v)))
                if q[1] > q[0]]
    return [(u, v) for u, v in keep if v - u >= shortest]


def rule(ax, a, b0, b1, avoid=(), pad=1.6, vertical=True, **kw):
    """A straight line at data a (x when vertical, else y) from b0 to b1 that breaks pad pt clear of every text in
    avoid it would run through: a label never sits on a line."""
    tr, inv = ax.transData, ax.transData.inverted()
    pick = (lambda p: p[1]) if vertical else (lambda p: p[0])
    P0, P1 = tr.transform((a, b0) if vertical else (b0, a)), tr.transform((a, b1) if vertical else (b1, a))
    A = P0[0] if vertical else P0[1]
    holes = [(y0, y1) if vertical else (x0, x1) for x0, x1, y0, y1 in _boxes(ax, avoid, pad)
             if (x0 <= A <= x1 if vertical else y0 <= A <= y1)]
    for u, v in _pieces(min(pick(P0), pick(P1)), max(pick(P0), pick(P1)), holes, 2.0 * ax.figure.dpi / 72):
        ends = [inv.transform((A, w) if vertical else (w, A)) for w in (u, v)]
        if vertical:
            ax.plot([a, a], [ends[0][1], ends[1][1]], **kw)
        else:
            ax.plot([ends[0][0], ends[1][0]], [a, a], **kw)


def text_right(ax, t, pad=0.0):
    """Data x of a placed text's right edge (plus pad pt)."""
    r = ax.figure.canvas.get_renderer()
    b = t.get_window_extent(r)
    return ax.transData.inverted().transform((b.x1 + pad * ax.figure.dpi / 72, b.y0))[0]


# ================================================================ Figure 1: the width x depth field
def figure_one(D, N, output):
    H = 3.50
    fig = plt.figure(figsize=(W, H))
    widths, stats = D["WSET"], D["wstat"]
    blue = cel.shades("k4")

    def shade(r):
        return blue[cel.depth_of(np.log(r), np.log(1.8), np.log(13.0))]

    # ---- a: every measured shape on the width x depth plane; contours of n/L
    ax = cel.axes(fig, 0.50, 0.40, 3.20, 3.16)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    x0, x1, y0, y1 = 44, 2900, 3.15, 23.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    xs = np.geomspace(x0, x1, 240)
    cuts = [0.5, 2, 4, 8, 16, 32, 64, 128, 256, 4096]
    for k in range(len(cuts) - 1):                       # n/L bands, deeper with n/L
        top, bot = np.clip(xs / cuts[k], y0 * 0.8, y1 * 1.25), np.clip(xs / cuts[k + 1], y0 * 0.8, y1 * 1.25)
        t = 0.02 + 0.30 * k / (len(cuts) - 2)
        ax.add_patch(Polygon(np.r_[np.c_[xs, top], np.c_[xs[::-1], bot[::-1]]], closed=True,
                             fc=mix(cel.PAPER, HUES["gray"][1], t), ec="none", zorder=0.2, gid="texture"))
    labels = []
    for c in (4, 8, 16, 32, 64, 128, 256):
        hot = c == 64
        if c >= 128:
            ax.plot(xs, xs / c, color=cel.TICK, lw=0.4, zorder=0.6, gid="texture")
            continue
        ax.plot(xs, xs / c, color=cel.line("focus") if hot else (cel.LINE if c == 8 else cel.TICK),
                lw=0.85 if hot else (0.55 if c == 8 else 0.4), zorder=0.6, gid="texture" if not hot else None)
        text = ("$n/L$ = 64" if hot else ("$n/L$ = 8" if c == 8 else str(c)))
        ly = y1 / 1.16
        if c * ly * 1.05 < x1 / 1.25:
            pos = dict(x=c * ly * 1.13, y=ly, ha="left", va="center")
        else:
            lx = x1 / 1.05
            pos = dict(x=lx, y=lx / c / 1.10, ha="right", va="top")
        labels.append(say(ax, pos.pop("x"), pos.pop("y"), text, color=cel.deeper(cel.line("focus"), 0.15) if hot
                          else MUTED, **pos, **(cel.BOLD if hot else {})))
    k3_1024 = [e["k3a"][0] / e["k4"][0] for _, e in D["wnet"][1024]]
    stations = [(w, 16, N[f"width{w}_ratio"], [e["k3a"][0] / e["k4"][0] for _, e in D["wnet"][w]]) for w in widths[:3]]
    stations += [(1024, 16, N["ratio_k3aug_over_k4_1024"], k3_1024),
                 (256, 4, N["nl64_256x4_ratio_point"], [a / b for a, b in D["NLPER"]["256x4"]]),
                 (512, 8, N["nl64_512x8_ratio_point"], [a / b for a, b in D["NLPER"]["512x8"]])]
    for n, L in ((64, 8), (96, 12)):
        stations.append((n, L, None, [N[f"fz_e1_{n}x{L}_s{s}_k3a_over_k4full"] for s in (301, 302, 303)]))
    stations.append((128, 16, None, [b["k3a"] / b["k4s"] for b in D["_ks128"]]))
    side = {(64, 8): "right", (96, 12): "right", (128, 16): "left", (256, 16): "below", (384, 16): "above",
            (512, 16): "below", (256, 4): "right", (512, 8): "right"}
    for n, L, r, nets in stations:
        g = float(np.exp(np.mean(np.log(nets)))) if r is None else r
        cel.discs(ax, [n], [L], "k4", colors=(shade(g), cel.deeper(shade(g), 0.07), cel.line("k4")), d=8.4,
                  lw=0.6, z=6, cast=True)
        k = len(nets)
        for j, q in enumerate(sorted(nets)):
            cel.discs(ax, [n], [L], "k4", colors=(shade(q), shade(q), cel.line("k4")), d=3.0, lw=0.35, z=6.2,
                      spark=False, transform=nudge(ax, (j - (k - 1) / 2) * 3.6, -8.6))
        if (n, L) == (1024, 16):
            continue
        txt = f"{r:.1f}" + r"$\times$" if r is not None else f"{min(nets):.1f}–{max(nets):.1f}" + r"$\times$"
        where = side[(n, L)]
        off = {"right": (6.4, 0, "left", "center"), "left": (-6.4, 0, "right", "center"),
               "above": (0, 7.0, "center", "bottom"), "below": (0, -13.4, "center", "top")}[where]
        say(ax, n, L, txt, ha=off[2], va=off[3], transform=nudge(ax, off[0], off[1]),
            color=cel.deeper(cel.line("k4"), 0.16))
    cel.chip(ax, 1024, 16, f"{N['ratio_k3aug_over_k4_1024']:.1f}" + r"$\times$", "focus", size=11.0, weight="bold",
             level=1, ha="left", va="center", transform=nudge(ax, 11.0, 2.5), pad=0.26)
    ax.set_xticks([64, 128, 256, 512, 1024], ["64", "128", "256", "512", "1024"])
    ax.set_yticks([4, 8, 12, 16], ["4", "8", "12", "16"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cel.axes_style(ax)
    ax.tick_params(length=0, pad=4.0)
    ax.set_xlabel(r"width $n$", labelpad=2)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    cel.ylab(ax, r"depth $L$", dx=-0.05).set_y(1.0 + 12.0 / 72 / cel.box_of(ax)[3])

    # ---- b: width at depth 16, every network a disc
    bx = cel.axes(fig, 4.00, 2.04, 6.66, 3.16)
    bx.set_xscale("log", base=2)
    bx.set_yscale("log")
    bx.set_xlim(215, 2600)
    bx.set_ylim(9e-10, 7e-6)
    jit = {0: 0.975, 1: 1.0, 2: 1.025, 3: 0.99, 4: 1.012}
    for key, role, mark in (("k3a", "k3", "diamond"), ("k4", "k4", "disc")):
        means = [stats[(w, key)][0] for w in widths]
        sh = 1.06 if key == "k4" else 0.94
        for w in widths:
            tube(bx, w * sh, stats[(w, key)][1], stats[(w, key)][2], role, width=2.8, z=2.4)
            vals = [entry[key][0] for _, entry in D["wnet"][w]]
            cel.discs(bx, [w * sh * jit[i % 5] for i in range(len(vals))], vals, role, 0, d=2.9, lw=0.4, z=4.2,
                      spark=False)
        cel.cline(bx, widths, means, role, 1, lw=1.0, rim=2.2, z=3, inner=cel.light(role))
        if mark == "disc":
            cel.discs(bx, widths, means, role, 1, d=5.8, lw=0.55, z=5, cast=True)
        else:
            cel.diamonds(bx, widths, means, role, 1, d=6.6, lw=0.55, z=5)
    a3, a4 = D["alpha"]["k3a"], D["alpha"]["k4"]
    say(bx, 1024 * 1.16, stats[(1024, "k3a")][0], "K=3 aug.", ha="left", va="center", color=MUTED)
    say(bx, 1024 * 1.16, stats[(1024, "k4")][0], "K=4 full", ha="left", va="center",
        color=cel.deeper(cel.line("k4"), 0.16))
    cel.chip(bx, 560, 2.6e-6, rf"$n^{{-{a3[0]:.2f}}}$ [{a3[1]:.2f}, {a3[2]:.2f}]", "k3", size=FSS)
    cel.chip(bx, 262, 2.7e-9, rf"$n^{{-{a4[0]:.2f}}}$ [{a4[1]:.2f}, {a4[2]:.2f}]", "k4", size=FSS)
    bx.set_xticks(widths, [str(w) for w in widths])
    bx.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cel.axes_style(bx)
    bx.set_xlabel(r"width $n$, depth 16", labelpad=1.5)
    cel.sheet_panel(bx, letter="b", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    log_ruler(bx, "y", -10, -5)
    bx.set_yticks([1e-9, 1e-8, 1e-7, 1e-6])
    cel.ylab(bx, "final-layer raw MSE", dx=-0.02).set_y(1.0 + 12.0 / 72 / cel.box_of(bx)[3])

    # ---- c: per layer at width 1024, two real error matrices (5 networks x 16 layers)
    cx = cel.axes(fig, 4.00, 0.40, 6.66, 1.56)
    k4m = np.array([D["k4w"][("full", i)][1] for i in range(5)])
    k3m = np.array([D["ref3_layers"][i] for i in range(5)])
    noise = np.mean([D["noise"][i] for i in range(5)], axis=0)
    lo_, hi_ = -10.0, -7.3
    cx.set_xlim(0.5, 16.5)
    y3 = 7.4
    cx.set_ylim(-1.75, y3 + 5.4)
    for mat, role, yb in ((k3m, "k3", y3), (k4m, "k4", 0.0)):
        cs = cel.shades(role)
        for i in range(5):
            for l in range(16):
                j = cel.depth_of(np.log10(mat[i, l]), lo_, hi_)
                cx.add_patch(Rectangle((l + 0.5, yb + 4 - i), 1, 1, fc=cs[j], ec=cel.PAPER, lw=0.35, zorder=2))
        cx.add_patch(Rectangle((0.5, yb), 16, 5, fc=cel.SHADE, ec="none", zorder=1.9,
                               transform=offset_copy(cx.transData, fig, x=1.2, y=-1.2, units="points")))
        cx.add_patch(Rectangle((0.5, yb), 16, 5, fc="none", ec=cel.line(role), lw=0.6, zorder=2.2))
        cx.add_patch(Rectangle((0.5, yb + 4), 16, 1, fc="none", ec=cel.deeper(cel.line(role), 0.18), lw=0.75,
                               zorder=2.3))
    cs = cel.shades("k3")
    for l in range(16):
        j = cel.depth_of(np.log10(noise[l]), lo_, hi_)
        cx.add_patch(Rectangle((l + 0.5, -1.45), 1, 0.9, fc=cs[j], ec=cel.PAPER, lw=0.35, zorder=2))
        cx.add_patch(cel.tier(Rectangle((l + 0.5, -1.45), 1, 0.9, fc="none", ec="none", lw=0, zorder=2.05),
                              cel.line("k3"), 1))
    cx.add_patch(Rectangle((0.5, -1.45), 16, 0.9, fc="none", ec=cel.line("k3"), lw=0.45, ls=(0, (1.5, 1.0)),
                           zorder=2.2))
    for l, key, ha, x in ((8, "depth_ratio_layer8", "center", 8), (11, "depth_ratio_layer11", "center", 11),
                          (16, "depth_ratio_layer16", "right", 16.15)):
        t = say(cx, x, (5.0 + y3) / 2, f"{N[key]:.1f}" + r"$\times$", ha=ha, va="center", size=FSS,
                color=cel.deeper(cel.line("k4"), 0.18),
                bbox=dict(boxstyle="round,pad=0.14,rounding_size=0.15", fc=cel.PAPER, ec=cel.line("k4"), lw=0.45))
        rule(cx, l, 5.0, y3, avoid=[t], pad=0.0, color=cel.line("k4"), lw=0.55, zorder=3)
    cx.set_yticks([])
    cx.set_xticks([1, 4, 8, 12, 16])
    cel.axes_style(cx, left=False)
    cx.set_xlabel("layer, width 1024 (official networks 0–4)", labelpad=1.5)
    cel.sheet_panel(cx, letter="c", header=7.0, stack=2.0, which="0101", peg_at=0.80)
    for text, y in (("K=3 aug.", y3 + 2.5), ("K=4 full", 2.5), ("truth noise", -1.0)):
        yy = inch(cx, 0.5, y)[1]
        fig.text(3.95 / W, yy / H, text, fontsize=FSS, ha="right", va="center", color=INK if y > 0 else MUTED)
    cel.depth_key(fig, 3.34, 0.105, "k3", "", "", w=0.24, h=0.05, size=FSS)
    cel.depth_key(fig, 3.62, 0.105, "k4", "", "", w=0.24, h=0.05, size=FSS)
    fig.text(3.30 / W, 0.13 / H, r"$10^{-10}$", fontsize=FSS, ha="right", va="center", color=MUTED)
    fig.text(3.90 / W, 0.13 / H, r"$10^{-7.3}$", fontsize=FSS, ha="left", va="center", color=MUTED)
    save(fig, output, "fig1_width_and_depth")


# ================================================================ Figure 2: births flowing to the layers that read them
def figure_two(D, N, output):
    H = 3.20
    fig = plt.figure(figsize=(W, H))
    FZ2, ab = D["FZ2"], D["abl01"]
    a5 = {k.replace("abl512_", ""): v for k, v in N.items() if k.startswith("abl512_")}

    # ---- a: each birth type's removal cost drifting with width, 128 -> 192 -> 512 -> 1024
    ax = cel.axes(fig, 0.30, 0.42, 2.74, 2.96)
    ax.set_xscale("log")
    ax.set_xlim(0.115, 24)
    rows = [  # label, role, E2 arm, frozen-1024 arm, 512 value, 1024 value
        (r"all of $\kappa_4$", "k3", "k3a", None, N["width512_seed41_k3a"] / a5["full"], ab["k3aug"] / ab["full"]),
        (r"C·$\kappa_3$ births", "focus", "no_k3c", "no_k3c", a5["no_three2"] / a5["full"], ab["no_k3c"] / ab["full"]),
        ("path births", "k3", "no_path", "no_path", a5["no_path"] / a5["full"], ab["no_path"] / ab["full"]),
        ("slice crossings", "k3", "no_slicex", "no_slicex", a5["no_crossed"] / a5["full"],
         ab["no_slicex"] / ab["full"])]
    ax.set_ylim(-0.80, len(rows) - 0.35)
    wkeys = [128, 192, 512, 1024]
    steps = {w: k + 1 for k, w in enumerate(wkeys)}
    for i, (label, role, arm, farm, v512, v1024) in enumerate(rows):
        y = len(rows) - 1 - i
        cs = cel.shades(role)
        nets = {n: [FZ2[(n, s, arm)] / FZ2[(n, s, "full")] for s in seeds]
                for n, seeds in ((128, (3, 4, 5)), (192, (601, 602, 603)))}
        means = [np.mean(nets[128]), np.mean(nets[192]), v512, v1024]
        hot = role == "focus"
        # the drift: arcs from each width's bead to the next, rising over the row
        for a, b in zip(means[:-1], means[1:]):
            if abs(np.log(b / a)) < 0.02:
                continue
            t = np.linspace(0, 1, 40)
            xx = np.exp(np.log(a) + (np.log(b) - np.log(a)) * t)
            yy = y + 0.30 * np.sin(np.pi * t) * min(1.0, abs(np.log(b / a)) / 0.5 + 0.35)
            ax.plot(xx, yy, color=cel.line(role), lw=0.9 if hot else 0.6, zorder=3, solid_capstyle="round")
        tail = [(m, k) for k, m in enumerate(means)]
        a, b = means[-2], means[-1]
        if abs(np.log(b / a)) >= 0.02:
            ax.annotate("", xy=(b, y), xytext=(np.exp(np.log(a) + 0.88 * (np.log(b) - np.log(a))), y + 0.10),
                        arrowprops=dict(arrowstyle="-|>,head_length=0.36,head_width=0.18", lw=0.6,
                                        color=cel.line(role), shrinkA=0, shrinkB=4.2), zorder=3.1)
        ax.plot([0.55, 21], [y, y], color=cel.light("k3"), lw=0.45, zorder=1.0)
        for m, k in tail:
            w = wkeys[k]
            f = cs[steps[w]]
            cel.discs(ax, [m], [y], role, colors=(f, cel.deeper(f, 0.07), cel.line(role)), d=4.6 + 0.9 * k, lw=0.5,
                      z=5 + k * 0.01, cast=True)
        for n in (128, 192):
            f = cs[steps[n]]
            cel.discs(ax, nets[n], [y - 0.24] * 3, role, colors=(f, f, cel.line(role)), d=3.0, lw=0.35, z=4.5,
                      spark=False)
        if farm:
            q = [D["RV"][(i5, farm)] / D["RV"][(i5, "full")] for i5 in (5, 6, 7)]
            f = cs[4]
            cel.discs(ax, q, [y - 0.24] * 3, role, colors=(f, f, cel.line(role)), d=3.0, lw=0.35, z=4.5, spark=False)
        say(ax, 0.13, y + 0.02, label, ha="left", va="center", color=cel.deeper(cel.line(role), 0.16) if hot else INK,
            **(cel.BOLD if hot else {}))
        if hot:
            cel.chip(ax, v1024, y, f"{v1024:.1f}" + r"$\times$", "focus", size=8.6, weight="bold", level=1,
                     ha="left", va="bottom", transform=nudge(ax, 5.5, 6.0))
    rule(ax, 1.0, -0.78, len(rows) - 0.45, avoid=ax.texts, color=cel.LINE, lw=0.55, ls=(0, (2.4, 1.6)), zorder=1.2)
    ax.set_xticks([0.6, 1, 2, 5, 10, 20], ["0.6", "1", "2", "5", "10", "20"])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([])
    cel.axes_style(ax, left=False)
    ax.spines["bottom"].set_bounds(0.55, 22)
    ax.set_xlabel(r"final MSE $\div$ full K=4, one birth type removed", labelpad=1.5)
    ax.xaxis.set_label_coords(0.60, -0.085)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.82)
    # width key: bead shade and size by width
    kx = ax.transData.inverted().transform(ax.transAxes.transform((0.02, 0.0)))[0]
    for k, w in enumerate(wkeys):
        xk = 0.19 * 1.62 ** k
        f = cel.shades("k3")[steps[w]]
        cel.discs(ax, [xk], [-0.58], "k3", colors=(f, cel.deeper(f, 0.07), cel.line("k3")), d=4.6 + 0.9 * k, lw=0.5,
                  z=5)
        say(ax, xk, -0.58, str(w), ha="center", va="bottom", transform=nudge(ax, 0, 4.4), color=MUTED)

    # ---- b: the K=4 state of official network 0 as reads: an arc from each birth layer to each later read
    bx = cel.axes(fig, 3.30, 0.30, 6.66, 2.96)
    bx.set_xticks([])
    bx.set_yticks([])
    cel.sheet_panel(bx, letter="b", header=7.0, stack=2.0, which="0101", peg_at=0.86)
    ar = cel.axes(fig, 3.36, 0.34, 5.30, 2.92)
    ar.set_zorder(3)
    ar.set_xlim(-0.3, 16.9)
    ar.set_ylim(-1.75, 1.75)
    ar.axis("off")
    cuts = [2, 3, 4, 6, 16]
    s4 = cel.shades("k4")
    band = [s4[4], s4[3], mix(s4[2], s4[3], 0.5), s4[2], s4[1]]   # age <= 2 deepest line, the long reads pale
    full0 = N["portable_net0"]
    lives = [N[f"lifetime{m}_net0"] for m in (2, 3, 4, 6)] + [full0]
    t = np.linspace(0, np.pi, 48)
    hk = 0.168
    for age in range(15, 0, -1):
        k = next(i for i, m in enumerate(cuts) if age <= m)
        for b in range(1, 17 - age):
            l = b + age
            cxm, hw, h = (b + l) / 2, age / 2, hk * age ** 0.82
            ar.plot(cxm - hw * np.cos(t), 0.06 + h * np.sin(t), color=band[k], lw=0.55 if k < 3 else 0.45,
                    zorder=2 + (5 - k) * 0.01, gid="texture", solid_capstyle="round")
            basis = age > 2
            ar.plot(cxm - hw * np.cos(t), -0.06 - h * np.sin(t), color=cel.TICK if basis else band[k],
                    lw=0.3 if basis else 0.55, zorder=2 + (0 if basis else 0.05), gid="texture")
    ar.plot([0.7, 16.3], [0, 0], color=cel.LINE, lw=0.6, zorder=3)
    cel.discs(ar, list(range(1, 17)), [0] * 16, "k4", 1, d=3.6, lw=0.4, z=4, spark=False)
    say(ar, 0.7, 0, "1", ha="right", va="center", color=MUTED, transform=nudge(ar, -2.0))
    say(ar, 16.3, 0, "16", ha="left", va="center", color=MUTED, transform=nudge(ar, 2.0))
    # ladder 1: final MSE when every source lives m layers (log)
    l1 = cel.axes(fig, 5.66, 1.72, 6.60, 2.80)
    l1.set_zorder(3)
    l1.set_xscale("log")
    l1.set_xlim(0.8, 1200)
    l1.set_ylim(4.6, -0.6)
    l1.axis("off")
    l1.spines[:].set_visible(False)
    for i, (m, v) in enumerate(zip(cuts, lives)):
        q = v / full0
        l1.add_patch(Rectangle((1.0, i - 0.30), q - 1.0 if q > 1.0 else 0.0, 0.60, fc=cel.light("k4"), ec="none",
                               zorder=2))
        l1.plot([1.0, max(q, 1.0)], [i, i], color=band[i], lw=1.6, zorder=2.1, solid_capstyle="butt")
        hot = m == 2
        say(l1, max(q, 1.0), i, (f"{q:.0f}" + r"$\times$") if m < 16 else "1", ha="left", va="center",
            transform=nudge(l1, 3.2), color=cel.deeper(cel.line("focus"), 0.15) if hot else
            cel.deeper(cel.line("k4"), 0.14), **(cel.BOLD if hot else {}))
        say(l1, 1.0, i, str(m), ha="right", va="center", transform=nudge(l1, -3.2), color=INK)
    say(l1, 1.0, -0.70, "layers lived", ha="right", va="bottom", transform=nudge(l1, 6.0), color=MUTED)
    # ladder 2: shared-basis entry, final MSE / exact
    l2 = cel.axes(fig, 5.66, 0.38, 6.60, 1.40)
    l2.set_zorder(3)
    l2.set_xscale("log")
    l2.set_xlim(0.8, 1200)
    l2.set_ylim(4.6, -0.6)
    l2.axis("off")
    l2.spines[:].set_visible(False)
    sets = [("birth, 112", N["shared_m00_k112_net0"]), ("1, 128", N["shared_m01_k128_net0"]),
            ("2, 128", N["shared_m02_k128_net0"]), ("2, 256", N["shared_m02_k256_net0_rescored"]), ("exact", full0)]
    for i, (lab, v) in enumerate(sets):
        q = v / full0
        cut = q > 1.05
        l2.add_patch(Rectangle((1.0, i - 0.30), max(q - 1.0, 0.0), 0.60, fc=cel.light("k3") if cut else cel.light("k4"),
                               ec="none", zorder=2))
        l2.plot([1.0, max(q, 1.0)], [i, i], color=cel.line("k3") if cut else band[0], lw=1.6, zorder=2.1,
                solid_capstyle="butt")
        say(l2, max(q, 1.0), i, f"{q:.2f}" + r"$\times$", ha="left", va="center", transform=nudge(l2, 3.2),
            color=MUTED if cut else cel.deeper(cel.line("k4"), 0.14), **(cel.BOLD if i == 2 else {}))
        say(l2, 1.0, i, lab, ha="right", va="center", transform=nudge(l2, -3.2), color=INK)
    say(l2, 1.0, -0.70, "entry, rank", ha="right", va="bottom", transform=nudge(l2, 6.0), color=MUTED)
    save(fig, output, "fig2_what_must_be_kept")


# ================================================================ Figure 3: the chain as 16 glass layer plates
X1, DX = 1.36, 0.110            # left edge of plate 1, plate pitch (in)
SX, SY = 0.40, 0.25             # a plate's depth into the page, oblique up-right: the reader faces the output side
XE = X1 + 15 * DX + SX + 0.07   # where the rails leave the stack
XR1 = 6.66                      # right end of the readout rulers
GAP = 0.170                     # rail pitch
KW = 0.0050                     # half-width of the K=4 body per live source group (in)


@contextmanager
def glass_geometry(**kw):
    """The constants above replaced inside the block (Figure 0 draws the same stack larger), restored on exit."""
    g = globals()
    old = {k: g[k] for k in kw}
    g.update(kw)
    try:
        yield
    finally:
        g.update(old)


def _zp(l):
    """Plate l's z; whatever lies between plates l and l+1 sits at _zp(l) + 0.025, so the nearer plates tint it."""
    return 2.0 + l * 0.05


def _p(l, y, z=0.5):
    """Screen inches of layer l, rail height y, depth z (0 near edge, 1 far edge; rails at mid-depth)."""
    return X1 + (l - 1) * DX + z * SX, y + z * SY


def glass_stack(fx, yb, yt):
    floor = [_p(0.35, yb - 0.03, -0.06), _p(16.65, yb - 0.03, -0.06), _p(16.65, yb - 0.03, 1.08),
             _p(0.35, yb - 0.03, 1.08)]
    fx.add_patch(Polygon(floor, closed=True, fc=cel.wash("k3"), ec=cel.light("k3"), lw=0.35, zorder=1.0))
    for l in range(1, 17):
        a, b = _p(l, yb - 0.03, -0.06), _p(l, yb - 0.03, 1.08)
        fx.plot([a[0], b[0]], [a[1], b[1]], color=cel.light("k3"), lw=0.3, zorder=1.05)
    sh = [_p(1, yb - 0.03, 0.0), _p(16, yb - 0.03, 0.0), _p(16, yb - 0.03, 1.0), _p(1, yb - 0.03, 1.0)]
    fx.add_patch(Polygon([(x + 0.07, y - 0.014) for x, y in sh], closed=True, fc=cel.SHADE, ec="none", alpha=0.75,
                         zorder=1.1))
    for l in range(1, 17):
        z = _zp(l)
        p = [_p(l, yb, 0), _p(l, yt, 0), _p(l, yt, 1), _p(l, yb, 1)]
        fx.add_patch(Polygon(p, closed=True, fc=cel.mid("k3"), ec="none", alpha=0.12, zorder=z))
        fx.add_patch(Polygon(p, closed=True, fc="none", ec=cel.light("k3"), lw=0.35, zorder=z + 0.004))
        fx.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], color=mix(cel.line("k3"), cel.TICK, 0.5), lw=0.4,
                zorder=z + 0.005)
        fx.plot([p[1][0], p[2][0]], [p[1][1], p[2][1]], color=cel.line("k3"), lw=0.35, zorder=z + 0.005)
        fx.plot([p[1][0] + 0.02, p[2][0] - 0.03], [p[1][1] + 0.005, p[2][1] - 0.016], color=cel.PAPER, lw=0.55,
                zorder=z + 0.006)


def _rail(fx, y, l0, l1, role, lw=0.9):
    """A K=3 chain from before plate l0 to just past plate l1, cut per gap so the nearer plates tint it."""
    for l in range(l0 - 1, l1 + 1):
        a, b = _p(max(l, l0 - 0.9), y), _p(min(l + 1, l1 + 0.5), y)
        fx.plot([a[0], b[0]], [a[1], b[1]], color=cel.line(role), lw=lw, zorder=_zp(max(l, 0)) + 0.025,
                solid_capstyle="butt")


def _body(fx, y, l0, cap, hatch=False, start=1):
    """The K=4 state from plate l0 to the exit: its half-width counts the live source groups (capped at cap)."""
    f, s, ln = cel.ramp("k4", 0)
    for l in range(l0, 17):
        n0, n1 = min(l - l0 + start, cap), min(l - l0 + start + 1, cap) if l < 16 else min(l - l0 + start, cap)
        h0, h1 = KW * max(n0, 0.45), KW * max(n1, 0.45)
        a = _p(l, y)
        b = _p(l + 1, y) if l < 16 else (a[0] + 0.06, a[1])
        z = _zp(l) + 0.025
        fx.add_patch(Polygon([(a[0], a[1] - h0 - 0.012), (a[0], a[1] + h0 + 0.012), (b[0], b[1] + h1 + 0.012),
                              (b[0], b[1] - h1 - 0.012)], closed=True, fc=cel.wash("k4"), ec="none", zorder=z - 0.001,
                             gid="texture"))
        poly = Polygon([(a[0], a[1] - h0), (a[0], a[1] + h0), (b[0], b[1] + h1), (b[0], b[1] - h1)], closed=True,
                       fc=f, ec="none", zorder=z)
        if hatch:
            cel.hatched(poly, ln, lw=0.3, n=7, kind="\\")
        fx.add_patch(poly)
        fx.plot([a[0], b[0]], [a[1] + h0, b[1] + h1], color=ln, lw=0.45, zorder=z + 0.001)
        fx.plot([a[0], b[0]], [a[1] - h0, b[1] - h1], color=ln, lw=0.45, zorder=z + 0.001)
        fx.plot([a[0] + 0.004, b[0] - 0.004], [a[1] + h0 * 0.55, b[1] + h1 * 0.55], color=cel.PAPER, lw=0.35,
                zorder=z + 0.002)
        if n0 >= 1:                       # a rib where this layer's births join the state
            fx.plot([a[0], a[0]], [a[1] - h0 * 0.8, a[1] + h0 * 0.8], color=cel.PAPER, lw=0.5, zorder=z + 0.002)
    # past the last plate the state narrows into the beam that reads it out
    a = (_p(16, y)[0] + 0.06, _p(16, y)[1])
    h = KW * max(min(16 - l0 + start, cap), 0.45)
    z = _zp(16) + 0.025
    fx.add_patch(Polygon([(a[0], a[1] - h), (a[0], a[1] + h), (XE, a[1] + 0.007), (XE, a[1] - 0.007)], closed=True,
                         fc=f, ec="none", zorder=z))
    for sg in (1, -1):
        fx.plot([a[0], XE], [a[1] + sg * h, a[1] + sg * 0.007], color=ln, lw=0.45, zorder=z + 0.001)


def _ignite(fx, y, l, hot=False, hollow=False):
    """Where the K=4 state starts: a ring on the glass between plates l and l+1, with fine rays."""
    x, yy = _p(l + 0.5, y)
    c = cel.line("focus") if hot else cel.deeper(cel.line("k4"), 0.05)
    z = _zp(l) + 0.035
    fx.scatter([x], [yy], s=5.0 ** 2, marker="o", fc="none" if hollow else cel.PAPER, ec=c, lw=0.55, zorder=z)
    fx.scatter([x], [yy], s=8.8 ** 2, marker="o", fc="none", ec=c, lw=0.3, zorder=z)
    for a in np.radians(np.arange(0, 360, 30) + 15):
        r1 = 0.082 if hot else 0.071
        fx.plot([x + np.cos(a) * 0.053, x + np.cos(a) * r1], [yy + np.sin(a) * 0.053, yy + np.sin(a) * r1],
                color=c, lw=0.35, zorder=z)
    if not hollow:
        fx.scatter([x], [yy], s=2.4 ** 2, marker="o", fc=c, ec="none", zorder=z + 0.001)


def _inject(fx, y, layers, kind):
    """Exact covariance (square, above the rail) or kappa4 (diamond, below) handed in at each marked layer."""
    for l in layers:
        x, yy = _p(l, y)
        if kind == "c":
            fx.scatter([x], [yy + 0.026], marker="s", s=4.6, fc=cel.mid("k4"), ec=cel.line("k4"), lw=0.3,
                       zorder=_zp(l) - 0.002)
        else:
            fx.scatter([x], [yy - 0.026], marker="D", s=4.2, fc=cel.line("k4"), ec="none", zorder=_zp(l) - 0.002)


def _beam(fx, y, x1, role):
    _, yy = _p(16, y)
    for w, a, c in ((6.2, 0.40, cel.wash(role)), (3.6, 0.75, cel.light(role)), (1.0, 1.0, cel.mid(role))):
        fx.plot([XE, x1], [yy, yy], color=c, lw=w, alpha=a, solid_capstyle="butt", zorder=2.95,
                gid=None if w == 1.0 else "texture")


def _ruler(fx, yr, ticks, labels, xmap, title):
    fx.plot([XE, XR1], [yr, yr], color=cel.LINE, lw=0.6, zorder=2.9)
    for v, lv in ticks:
        fx.plot([xmap(v)] * 2, [yr, yr - (0.05, 0.032, 0.018)[lv]], color=cel.LINE, lw=0.4, zorder=2.9)
    for v, t in labels:
        fx.text(xmap(v), yr - 0.068, t, ha="center", va="top", fontsize=FSS, color=INK)
    fx.text(XR1, yr - 0.205, title, ha="right", va="top", fontsize=FSS, color=INK)


def _bracket(fx, x, y0, y1, starts, text):
    c = cel.line("focus")
    fx.plot([x, x], [y0, y1], color=c, lw=0.8, zorder=6)
    for yy, x0 in zip((y0, y1), starts):
        fx.plot([x0, x], [yy, yy], color=c, lw=0.5, ls=(0, (1.4, 1.0)), zorder=6)
    cel.chip(fx, x + 0.05, (y0 + y1) / 2, text, "focus", size=8.4, weight="bold", level=1)


def figure_three(D, N, output):
    H = 3.86
    fig = plt.figure(figsize=(W, H))
    fx = canvas(fig)

    def row_label(y, text):
        fx.text(X1 - 0.05, _p(1, y)[1], text, ha="right", va="center", fontsize=FSS, color=INK)

    # ---- a: ARC's augmented K=3 handed the K=4 chain's own exact quantities, against the K=4 chain itself
    rows = [("K=3 augmented", N.get("oracle_unpatched", 1.0), "", ()),
            (r"+ $\kappa_4$, layers 9–15", N["oracle_k4_proj_late"], "k", range(9, 16)),
            (r"+ $\kappa_4$", N["oracle_k4"], "k", range(1, 17)),
            ("+ covariance", N["oracle_C"], "c", range(1, 17)),
            ("+ both", N["oracle_C_k4"], "ck", range(1, 17)),
            ("K=4 chain", N["oracle_full_k4_chain"], "k4", ())]
    ya = 2.50
    ys = [ya + (len(rows) - 1 - i) * GAP for i in range(len(rows))]
    glass_stack(fx, ya - 0.11, ys[0] + 0.11)
    xa = lambda v: XE + v / 1.12 * (XR1 - XE)
    ends = []
    for (lab, v, how, lay), y in zip(rows, ys):
        yy = _p(16, y)[1]
        row_label(y, lab)
        if how == "k4":
            _body(fx, y, 1, 99)
            role = "k4"
        else:
            _rail(fx, y, 1, 16, "k3")
            fx.plot([_p(16.5, y)[0], XE], [yy, yy], color=cel.line("k3"), lw=0.9, zorder=_zp(16) + 0.025)
            for kind in "ck":
                if kind in how:
                    _inject(fx, y, lay, kind)
            role = "k3"
        _beam(fx, y, xa(v), role)
        cel.discs(fx, [xa(v)], [yy], role, 1, d=5.6, lw=0.55, z=6, cast=True)
        ends.append(fx.text(xa(v) + 0.07, yy, f"{v:.2f}", ha="left", va="center", fontsize=FSS, color=MUTED, zorder=7,
                            **(cel.BOLD if how in ("ck", "k4") else {})))
    fig.canvas.draw()
    q = N["oracle_C_k4"] / N["oracle_full_k4_chain"]
    _bracket(fx, xa(0.50), _p(16, ys[5])[1], _p(16, ys[4])[1], [text_right(fx, ends[5], 3), text_right(fx, ends[4], 3)],
             f"{q:.1f}" + r"$\times$ short")
    _ruler(fx, ya - 0.20, [(v, 0 if abs(v * 4 - round(v * 4)) < 1e-6 else 2) for v in np.arange(0, 1.1001, 0.05)],
           [(v, f"{v:g}") for v in (0, 0.25, 0.5, 0.75, 1)], xa, r"final MSE $\div$ K=3 aug., networks 0–3")
    for l in (1, 4, 8, 12, 16):
        x, yy = _p(l, ys[0] + 0.11, 1.0)
        fx.text(x, yy + 0.03, str(l), ha="center", va="bottom", fontsize=FSS, color=MUTED)
    x, yy = _p(1, ys[0] + 0.11, 1.0)
    fx.text(x - 0.09, yy + 0.03, "layer", ha="right", va="bottom", fontsize=FSS, color=MUTED)
    cel.badge(fig, 0.10, ys[0] + 0.20, "a", "gray", 0, gid="tab")

    # ---- b: chain B to layer s0, then the K=4 state, official network 0
    B = D["late"]["B"]
    caps = [N["late4_tree_star0_age3"], N["late4_tree_star0_age2"], N["late4_tree_star0_age1"]]
    rows = [("chain B", B, None, {}), ("K=4 from the input", N["portable_net0"], 0, {}),
            ("$s_0$ = 4", D["late"][4], 4, {"hot": True}),
            ("tree births only", N["late4_tree_star0_age0"], 4, {"hatch": True}),
            (r"slice age $\leq$ 3", caps[0], 4, {"hatch": True, "cap": 3}),
            ("$s_0$ = 6", D["late"][6], 6, {}), ("$s_0$ = 8", D["late"][8], 8, {}),
            (r"$s_0$ = 8, empty $\kappa_4$", D["late"]["empty8"], 8, {"empty": True})]
    yb = 0.52
    ys = [yb + (len(rows) - 1 - i) * GAP for i in range(len(rows))]
    glass_stack(fx, yb - 0.11, ys[0] + 0.11)
    lo, hi = 1.2e-9, 5.0e-7
    xb = lambda v: XE + (np.log10(v) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)) * (XR1 - XE)
    ends = []
    for (lab, v, s0, o), y in zip(rows, ys):
        yy = _p(16, y)[1]
        row_label(y, lab)
        if s0 is None:
            _rail(fx, y, 1, 16, "chainb")
            fx.plot([_p(16.5, y)[0], XE], [yy, yy], color=cel.line("chainb"), lw=0.9, zorder=_zp(16) + 0.025)
            role = "chainb"
        else:
            if s0 > 0:
                _rail(fx, y, 1, s0, "chainb")
            _body(fx, y, s0 + 1, o.get("cap", 99), hatch=o.get("hatch", False), start=0 if o.get("empty") else 1)
            if s0 > 0:
                _ignite(fx, y, s0, hot=o.get("hot", False), hollow=o.get("empty", False))
            role = "k3" if o.get("hatch") else "k4"
        _beam(fx, y, xb(v), role)
        cel.discs(fx, [xb(v)], [yy], role, 1, d=5.6, lw=0.55, z=6, cast=True)
        ends.append(fx.text(xb(v) + 0.07, yy, sci(v), ha="left", va="center", fontsize=FSS, color=MUTED, zorder=7))
    fig.canvas.draw()
    q = N["late4_tree_star0_age0"] / D["late"][4]
    yr = yb - 0.20
    rule(fx, xb(B), yr, _p(16, ys[0])[1] + 0.09, avoid=ends, color=cel.line("chainb"), lw=0.55, ls=(0, (2.4, 1.5)),
         zorder=2.92)
    _bracket(fx, 6.05, _p(16, ys[3])[1], _p(16, ys[2])[1], [text_right(fx, ends[3], 3), text_right(fx, ends[2], 3)],
             f"{q:.1f}" + r"$\times$")
    tk = [(m * 10.0 ** e, 0 if m == 1 else (1 if m in (2, 5) else 2)) for e in range(-9, -6) for m in range(1, 10)
          if lo <= m * 10.0 ** e <= hi]
    _ruler(fx, yr, tk, [(10.0 ** e, rf"$10^{{{e}}}$") for e in (-8, -7)], xb, "final MSE, official network 0")
    cel.badge(fig, 0.10, ys[0] + 0.20, "b", "gray", 0, gid="tab")
    save(fig, output, "fig3_whole_chain_late_start")


# ================================================================ Figure 4: the scored steps, and cost against accuracy
def figure_four(D, N, output):
    H = 3.72
    fig = plt.figure(figsize=(W, H))
    ids = [332957, 334990, 335127, 335162, 335185, 335300, 335301, 336537]
    origin = ["V29", "mixed", "ours", "ours", "ported", "ours", "ours", "fitted"]
    lad = D["_lad"]

    def orole(j):
        return "fit" if origin[j] == "fitted" else ("chainb" if origin[j] == "ours" else "k3")

    # ---- a: eight scorer receipts, the grader's 50 leaderboard networks one trace each
    ax = cel.axes(fig, 0.50, 2.08, 3.92, 3.38)
    ax.set_xlim(0.30, 8.45)
    ax.set_ylim(2.0, 7.4)
    M = np.array([lad[i] for i in ids]) * 1e9
    ax.axvspan(7.55, 8.45, color=cel.wash("focus"), lw=0, zorder=0.3, gid="texture")
    for k in np.argsort(M[0]):
        ax.plot(range(1, 9), M[:, k], color=cel.light("chainb"), lw=0.35, zorder=1.2, alpha=0.9, gid="texture")
    for j in range(8):
        rnk = np.argsort(np.argsort(M[j]))
        cel.discs(ax, j + 1 + ((rnk % 7) - 3) * 0.04 + 0.13, M[j], orole(j), 0, d=2.3, lw=0.3, z=3, spark=False)
    means = M.mean(axis=1)
    cel.cline(ax, range(1, 9), means, "chainb", 1, lw=1.2, rim=2.4, z=4, inner=cel.light("chainb"))
    for j in range(8):
        cel.discs(ax, [j + 1], [means[j]], orole(j), 2 if origin[j] in ("ours", "fitted") else 1, d=6.8, lw=0.6, z=6,
                  cast=True)
    for j in (0, 7):
        say(ax, j + 1, means[j], f"{means[j]:.2f}", ha="right", va="top", transform=nudge(ax, -4.2, -3.2),
            color=cel.deeper(cel.line(orole(j)), 0.16), **(cel.BOLD if j == 7 else {}),
            path_effects=[cel.pe.withStroke(linewidth=2.2, foreground=cel.PAPER)])
    own = D["LAD"]["own"]
    ch = cel.chip(ax, 4.65, 6.62, r"our steps 3, 4, 6, 7: $\times$" + f"{own['rom']:.3f} [{own['ci'][0]:.3f}, "
                  f"{own['ci'][1]:.3f}], {own['better']}/50", "focus", size=FSS + 0.2, weight="bold", level=1, ha="center")
    fig.canvas.draw()
    yb = _boxes(ax, [ch], 0.0)[0][2] - 3.0 * fig.dpi / 72
    ytop = ax.transData.inverted().transform((0, yb))[1]
    for j in (2, 3, 5, 6):                  # fine leaders from the chip to the four steps it multiplies
        ax.plot([j + 1, j + 1], [ytop, means[j] + 0.20], color=cel.line("focus"), lw=0.5, zorder=5.5)
        ax.scatter([j + 1], [ytop], s=2.2 ** 2, color=cel.line("focus"), zorder=5.6)
    ax.set_xticks(range(1, 9), [f"{j + 1}" for j in range(8)])
    ax.set_yticks([2, 3, 4, 5, 6, 7])
    cel.axes_style(ax)
    ax.tick_params(axis="x", length=0, pad=4.5)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.90)
    cel.rulers(ax, y=(1, 0.5, 0.1))
    cel.ylab(ax, r"adjusted error ($\times10^{-9}$), 50 leaderboard networks", dx=-0.06).set_y(
        1.0 + 12.0 / 72 / cel.box_of(ax)[3])
    for j in range(8):
        xi = inch(ax, j + 1, 2.0)[0]
        cel.badge(fig, xi - 0.19, 1.735, origin[j], orole(j), 1 if origin[j] in ("ours", "fitted") else 0, size=FSS,
                  w=0.38, h=0.13)

    # ---- b: the layer-4 complete chain, counted, as budget tiles (one tile = 1,024 units = one network's budget)
    L4 = D["LEDGER_FULL"][4]
    rows = L4["rows"]
    other = sum(v for k, v in rows.items() if not k.startswith(("C.k3", "non-tree")))
    parts = [("other pieces", other, "k3", 1, None),
             (r"C·$\kappa_3$ births", rows["C.k3 births at real kappa3 width, beyond n-wide legs"], "k4", 1, None),
             (r"C·$\kappa_3$ crossing", rows["C.k3 crossing at rank 16"], "k4", 1, "\\"),
             ("non-tree first reads", rows["non-tree first reads, dense, classes merged"], "k4", 2, None)]
    bx = cel.axes(fig, 0.50, 0.30, 3.92, 1.42)
    cols = 11
    bx.set_xlim(-0.3, 21.4)
    bx.set_ylim(6.35, -0.55)
    bx.set_aspect("equal")
    edges, acc = [], 0.0
    for name, v, role, level, hatch in parts:
        edges.append((acc / 1024, (acc + v) / 1024, role, level, hatch))
        acc += v
    total = acc / 1024
    for t in range(int(np.ceil(total))):
        r_, c_ = divmod(t, cols)
        frac = min(1.0, total - t)
        for a, b, role, level, hatch in edges:
            lo, hi = max(a, t), min(b, t + frac)
            if hi <= lo:
                continue
            p = Rectangle((c_ + (lo - t) * 0.86 + 0.07, r_ + 0.07), (hi - lo) * 0.86, 0.86,
                          fc=cel.shades(role)[level], ec="none", zorder=3)
            bx.add_patch(cel.hatched(p, cel.line("k4"), lw=0.3, n=7, kind="\\") if hatch else p)
        bx.add_patch(Rectangle((c_ + 0.07, r_ + 0.07), 0.86 * frac, 0.86, fc="none", ec=cel.line("k4"), lw=0.35,
                               zorder=3.2))
        bx.add_patch(Rectangle((c_ + 0.07, r_ + 0.07), 0.86 * frac, 0.86, fc=cel.SHADE, ec="none", zorder=2.9,
                               transform=offset_copy(bx.transData, fig, x=0.9, y=-0.9, units="points")))
    bx.add_patch(Rectangle((0.07, 0.07), 0.86, 0.86, fc="none", ec=cel.line("focus"), lw=1.0, zorder=4))
    last = int(np.ceil(total)) - 1
    say(bx, last % cols + (total - last) + 0.35, last // cols + 0.5,
        f"{L4['gain_share']:.0%}" + r" in C·$\kappa_3$ and non-tree", ha="left", va="center",
        color=cel.deeper(cel.line("k4"), 0.16), **cel.BOLD)
    bx.axis("off")
    cel.sheet_panel(bx, letter="b", header=7.0, stack=2.0, which="0101", peg_at=0.86)
    say(bx, 11.75, -0.05, f"{L4['total']:,.0f} units = {L4['total'] / 1024:.0f} budgets", ha="left", va="center",
        size=FS, **cel.BOLD)
    key = [("one network's budget", None, None, None, "frame")] + [(f"{nm} {v / 1000:,.1f}k", role, level, hatch, None)
                                                                  for nm, v, role, level, hatch in parts]
    key.append((f"non-tree as rank-1 pairs {N['pair_units_r1_local'] / 1000:.1f}k", "k4", 1, None, "pairs"))
    for k, (text, role, level, hatch, kind) in enumerate(key):
        y = 0.95 + k * 0.90
        if kind == "frame":
            bx.add_patch(Rectangle((11.80, y - 0.32), 0.64, 0.64, fc="none", ec=cel.line("focus"), lw=1.0, zorder=4))
        elif kind == "pairs":
            frac = N["pair_units_r1_local"] / 1024
            bx.add_patch(Rectangle((11.80, y - 0.32), 0.64 * frac, 0.64, fc=cel.shades("k4")[2], ec="none", zorder=3))
            bx.add_patch(Rectangle((11.80, y - 0.32), 0.64, 0.64, fc="none", ec=cel.line("k4"), lw=0.35,
                                   ls=(0, (1.2, 1.0)), zorder=3.1))
        else:
            p = Rectangle((11.80, y - 0.32), 0.64, 0.64, fc=cel.shades(role)[level], ec=cel.line("k4"), lw=0.35,
                          zorder=3)
            bx.add_patch(cel.hatched(p, cel.line("k4"), lw=0.3, n=7, kind="\\") if hatch else p)
        say(bx, 12.70, y, text, ha="left", va="center")

    # ---- c: cost against accuracy on network 0; the diagonal is chain B's score
    cx = cel.axes(fig, 4.62, 0.42, 6.66, 3.38)
    cx.set_xscale("log")
    cx.set_yscale("log")
    xlo, xhi, ylo, yhi = 28, 4.0e6, 0.055, 1.55
    cx.set_xlim(xlo, xhi)
    cx.set_ylim(ylo, yhi)
    B0, rawB = D["B0_UNITS"], D["late"]["B"]
    xs = np.geomspace(xlo, xhi, 200)
    # over the budget the network scores as an all-zero prediction: hatched
    cx.add_patch(cel.hatched(Rectangle((1024, ylo), xhi - 1024, yhi - ylo, fc=cel.wash("k3"), ec="none", zorder=0.3,
                                       gid="texture"), cel.light("k3"), lw=0.35, n=4, kind="/"))
    # below the diagonal and inside the budget: better than chain B's score
    zone = np.c_[np.r_[B0, 1024, 1024, B0], np.r_[1.0, B0 / 1024, ylo, ylo]]
    xs = np.geomspace(102.4, xhi, 200)          # below 102.4 units the score's floor holds: the tie leaves the plot
    cx.add_patch(Polygon(zone, closed=True, fc=cel.wash("chainb"), ec="none", zorder=0.4, gid="texture"))
    cx.plot(xs, B0 / xs, color=cel.line("chainb"), lw=0.8, zorder=2, ls=(0, (3.0, 1.6)))
    cx.plot([1024, 1024], [ylo, yhi], color=cel.LINE, lw=0.8, zorder=2.1)
    fk = N["frontier_fullk4_ratio_net0"]
    lab_floor = say(cx, 1.2e5, fk, r"full K=4, $n^4$", ha="center", va="bottom", transform=nudge(cx, 0, 1.6),
                    color=cel.deeper(cel.line("k4"), 0.12))
    fig.canvas.draw()
    rule(cx, fk, xlo, xhi, avoid=[lab_floor], vertical=False, color=cel.line("k4"), lw=0.55, ls=(0, (1.2, 1.2)),
         zorder=1.8)
    # the in-budget K=4 pieces (Table 6): hollow read at the output, filled fed back
    pieces = [("diag_s12", 0), ("cross4_s1", 0), ("star4_s1", 0), ("star_4_all", 1), ("slice31_4_all", 1),
              ("pair_3_all", 1), ("cross_2_all_4", 1), ("combo_4_all", 1)]
    for key, fed in pieces:
        u, r = N[f"frontier_{key}_units"], N[f"frontier_{key}_ratio"]
        cel.discs(cx, [u], [r], "k4", 0, d=3.2, lw=0.4, z=4, spark=False)
    cel.discs(cx, [B0], [1.0], "chainb", 1, d=6.6, lw=0.6, z=6, cast=True)
    # the complete chain from layer s0 (Table 7a): counted units at the compressed raw (filled) or the exact raw (open,
    # a bound), a dotted line to the units at which it ties chain B
    ticks = []
    for s0 in (0, 4, 6, 8):
        Lg = D["LEDGER_FULL"][s0]
        exact = "raw_c" not in Lg
        r = (Lg["raw"] if exact else Lg["raw_c"]) / rawB
        tie = Lg["tie"] if exact else Lg["tie_c"]
        hot = s0 == 4
        cx.plot([tie, Lg["total"]], [r, r], color=cel.line("focus") if hot else cel.line("k4"), lw=0.7 if hot else 0.45,
                ls=(0, (1.2, 1.2)), zorder=3)
        cx.scatter([Lg["total"]], [r], marker="s", s=4.6 ** 2, fc=cel.PAPER if exact else cel.light("k4"),
                   ec=cel.line("k4"), lw=0.6, zorder=5)
        cx.scatter([tie], [r], marker="|", s=5.0 ** 2, color=cel.line("focus") if hot else cel.line("k4"), lw=0.8,
                   zorder=5)
        say(cx, Lg["total"], r, rf"$s_0$ = {s0}", ha="center", va="top" if s0 in (0, 4) else "bottom",
            transform=nudge(cx, 0, -4.6 if s0 in (0, 4) else 4.6), color=MUTED)
        txt = (r"$\leq$" if exact else "") + f"{tie:,.0f}"
        if hot:
            ticks.append(cel.chip(cx, tie, r, f"{tie:,.0f}", "focus", size=8.4, weight="bold", level=1, ha="right",
                                  va="center", transform=nudge(cx, -5.5)))
        else:
            ticks.append(say(cx, tie, r, txt, ha="right", va="center", transform=nudge(cx, -4.0),
                             color=cel.deeper(cel.line("k4"), 0.14)))
    r4 = D["LEDGER_FULL"][4]["raw_c"] / rawB
    cx.plot([D["LEDGER_FULL"][4]["total"], D["EXEC"]["total"]], [r4, r4], color=cel.line("focus"), lw=0.7,
            ls=(0, (1.2, 1.2)), zorder=3)
    cel.discs(cx, [D["EXEC"]["total"]], [r4], "k4", 1, d=5.6, lw=0.55, z=5, cast=True)
    say(cx, D["EXEC"]["total"], r4, "dense run", ha="right", va="bottom", transform=nudge(cx, 1.0, 4.6), color=MUTED)
    say(cx, 1024, yhi, "budget", ha="left", va="top", transform=nudge(cx, 2.5, -2.5), color=INK)
    kx = 3.3e3                              # what each mark is: metered, counted, derived
    cel.discs(cx, [kx], [0.150], "k4", 1, d=4.6, lw=0.5, z=5, spark=False)
    cx.scatter([kx], [0.124], marker="s", s=4.2 ** 2, fc=cel.PAPER, ec=cel.line("k4"), lw=0.6, zorder=5)
    cx.scatter([kx], [0.102], marker="|", s=5.0 ** 2, color=cel.line("k4"), lw=0.8, zorder=5)
    for y, t in ((0.150, "metered"), (0.124, "counted"), (0.102, "derived")):
        say(cx, kx, y, t, ha="left", va="center", transform=nudge(cx, 4.4), color=MUTED)
    cx.set_xticks([1e2, 1e3, 1e4, 1e5, 1e6])
    cx.set_yticks([0.1, 0.2, 0.5, 1.0], ["0.1", "0.2", "0.5", "chain B"])       # y = 1 is chain B itself
    cx.get_yticklabels()[-1].set_color(cel.deeper(cel.line("chainb"), 0.16))
    cx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cel.axes_style(cx)
    cx.set_xlabel("units per network", labelpad=1.5)
    cel.sheet_panel(cx, letter="c", header=7.0, stack=2.0, which="0101", peg_at=0.82)
    log_ruler(cx, "x", 2, 6)
    cel.ylab(cx, r"raw $\div$ chain B, network 0", dx=-0.10).set_y(1.0 + 12.0 / 72 / cel.box_of(cx)[3])
    save(fig, output, "fig4_scored_and_cost")


# ================================================================ Figure 5: where the 602-unit target is bound
def _red_kind(p):
    g, lab = p["group"], p["label"]
    if g == "chain B":
        return "chainb"
    if "corrector" in g:
        return None
    if g.startswith("B + K=4 pieces"):
        return "fed"
    if "weight" in g and "reverse" not in g:
        return "weight"
    if "Table 7a" in g or "from the input" in g:
        return "counted"
    if g.startswith("full K=4"):
        return "full"
    if "treeall" in lab:
        return "window"
    return "rule"


def _metered_labels(ax, R, c4, m1, p8, p32):
    """Figure 5a's metered hexagons, each named in a two-line capsule on a leader, and the 12.5x between the counted
    layer-4 chain and the dense metered run (one raw) as an arc over the cluster with an arrowhead on each point.
    p8 and p32 sit 2.5% apart in units: p8's capsule stands above, inside the arc, p32's at the foot of the panel and
    the dense run's lower right, so no leader crosses another leader or the arc."""
    fig = ax.figure
    rd = fig.canvas.get_renderer()
    k = fig.dpi / 72                                        # display px per pt
    tr = ax.transData
    at = lambda q: tr.transform((q["units"], q["ratio"]))
    back = lambda xy: tr.inverted().transform(xy)
    edge = cel.line("k4")
    pad = 0.28 * FSS                                        # cel.chip's frame pad, pt
    hx = 2.7                                                # hexagon circumradius at the 5.4 pt marker, pt

    def capsule(x, y, key, dx, dy, ha, va):
        name, value = R["caps"][key]
        t = cel.chip(ax, x, y, r"$\mathbf{" + name.replace(" ", r"\ ") + "}$\n" + value, "k4", size=FSS,
                     ha=ha, va=va, transform=nudge(ax, dx, dy), z=7.5)
        t.set_multialignment("left")
        t.set_linespacing(1.15)
        t.update_bbox_position_size(rd)
        return t.get_bbox_patch().get_window_extent(rd)

    def leader(xpx, y0px, y1px):
        ax.plot(*np.array([back((xpx, y0px)), back((xpx, y1px))]).T, color=edge, lw=0.45, zorder=7.2,
                solid_capstyle="butt")

    P4, Pm, P8, P32 = at(c4), at(m1), at(p8), at(p32)
    # the arc's feet: the counted square's top-right corner and the dense hexagon's top vertex
    foot_l = P4 + np.array([2.2 + 0.3, 2.2 + 0.3]) * k
    foot_r = Pm + np.array([0.0, hx + 0.5]) * k
    base_l, base_r = foot_l + np.array([1.7, 2.8]) * k, foot_r + np.array([0.0, 3.3]) * k
    cx = (base_l[0] + base_r[0]) / 2
    # p8 above, centred under the arc's crown
    b8 = capsule(p8["units"], p8["ratio"], "p8_net0", (cx - P8[0]) / k, hx + 4.5 + pad, "center", "bottom")
    leader(P8[0], P8[1] + (hx + 0.4) * k, b8.y0 - 1.6 * k)        # the capsule's hard shadow hangs 1.2 pt below
    # the dense run lower right, clear of p32's leader by 4.5 pt
    left_m = max(Pm[0] - 6.0 * k, P32[0] + 4.5 * k)
    bm = capsule(m1["units"], m1["ratio"], "m1_net0", (left_m - Pm[0]) / k + pad, -(hx + 7.0 + pad), "left", "top")
    leader(Pm[0], Pm[1] - (hx + 0.4) * k, bm.y1)
    # p32 at the foot of the panel, its right edge on the dense run's
    ylo = ax.get_ylim()[0]
    b32 = capsule(p32["units"], ylo, "p32_net0", (bm.x1 - P32[0]) / k - pad, 6.0 + pad, "right", "bottom")
    leader(P32[0], P32[1] - (hx + 0.4) * k, b32.y1)

    # the arc: a superellipse (exponent 8) from foot to foot whose crown clears p8's capsule and carries the chip
    y0 = max(base_l[1], base_r[1])
    a = (base_r[0] - base_l[0]) / 2
    b = b8.y1 + 7.5 * k - y0
    th = np.linspace(np.pi, 0, 600)
    xs = cx + a * np.sign(np.cos(th)) * np.abs(np.cos(th)) ** (2 / 8)
    ys = y0 + b * np.abs(np.sin(th)) ** (2 / 8)
    xs = np.r_[base_l[0], xs, base_r[0]]
    ys = np.r_[base_l[1], ys, base_r[1]]
    chip = cel.chip(ax, *back((cx, y0 + b)), rf"{R['count_ratio']:.1f}$\times$", "k4", size=FSS, level=0,
                    ha="center", va="center", z=7.5)
    chip.update_bbox_position_size(rd)
    cb = chip.get_bbox_patch().get_window_extent(rd).expanded(1.0, 1.0).padded(1.2 * k)
    keep = ~((xs > cb.x0) & (xs < cb.x1) & (ys > cb.y0) & (ys < cb.y1))
    for run in np.split(np.arange(len(xs)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1):
        if keep[run[0]] and len(run) > 1:
            ax.plot(*back(np.c_[xs[run], ys[run]]).T, color=edge, lw=0.5, zorder=7.1, solid_capstyle="butt")
    for frm, to in ((base_l, foot_l), (base_r, foot_r)):
        ax.add_patch(FancyArrowPatch(back(frm), back(to), arrowstyle="-|>,head_length=0.34,head_width=0.2",
                                     mutation_scale=FSS, color=edge, lw=0.5, shrinkA=0, shrinkB=0, zorder=7.1))


def figure_five(D, N, output):
    # panel a is 0.25 in taller and 0.40 in wider than v23c at the same pt per decade (ylo and xhi move, nothing else):
    # the room below and right of the metered hexagons holds their capsules, the room below 0.1 the key
    H = 3.61
    fig = plt.figure(figsize=(W, H))
    R = D["RED_FIG"]
    pts = [p for p in R["pts"] if p["net"] == 0 and _red_kind(p)]
    B = [p for p in pts if p["group"] == "chain B"][0]
    uB = B["units"]

    # ---- a: final raw against counted units, every measured point of network 0
    ax = cel.axes(fig, 0.60, 0.40, 4.80, 3.21)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xlo, xhi, ylo, yhi = 105, 6.27e6, 0.025, 60
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.add_patch(cel.hatched(Rectangle((1024, ylo), xhi - 1024, yhi - ylo, fc=cel.wash("k3"), ec="none", zorder=0.3,
                                       gid="texture"), cel.light("k3"), lw=0.35, n=4, kind="/"))
    zone = np.c_[np.r_[uB, 1024, 1024, uB], np.r_[1.0, uB / 1024, ylo, ylo]]
    ax.add_patch(Polygon(zone, closed=True, fc=cel.wash("chainb"), ec="none", zorder=0.4, gid="texture"))
    xs = np.geomspace(uB / 1.6, uB / ylo, 200)          # chain B's score: raw / raw_B = units_B / units
    ax.plot(xs, uB / xs, color=cel.line("chainb"), lw=0.8, zorder=2, ls=(0, (3.0, 1.6)))
    ax.plot([1024, 1024], [ylo, yhi], color=cel.LINE, lw=0.8, zorder=2.1)

    style = {"fed": ("chainb", 0), "weight": ("k3", 2), "rule": ("k4", 0), "window": ("k4", 1)}
    for p in pts:
        k = _red_kind(p)
        if k in style:
            role, lev = style[k]
            cel.discs(ax, [p["units"]], [p["ratio"]], role, lev, d=4.4 if p["frontier"] else 3.4, lw=0.4,
                      z=4.2 if p["frontier"] else 3.6, spark=False)
    # the first compression (Table 11): the 20,586-unit rule's births with non-tree births as matrix pairs (gold, the
    # focus path), and both families as pairs (blue diamonds), at units that keep the real kappa3 leg width
    p2 = [q for q in R["pts2"] if q["net"] == 0 and not q["control"]]
    path2 = sorted([q for q in p2 if q["ck"] == 0], key=lambda q: -q["units"])
    both2 = [q for q in p2 if q["ck"] > 0]
    cel.diamonds(ax, [q["units"] for q in both2], [q["ratio"] for q in both2], "k4", 1, d=5.4, lw=0.45, z=4.4)
    # the lower envelope: the best raw at no more than these units, a descending stair, over both batches
    cand = [p for p in pts if p["frontier"]] + [q for q in p2 if q["frontier"]]
    env = sorted([p for p in cand if not any(o["units"] <= p["units"] and o["ratio"] < p["ratio"] for o in cand)],
                 key=lambda p: (p["units"], -p["ratio"]))
    sx, sy = [env[0]["units"]], [env[0]["ratio"]]
    for a, b in zip(env, env[1:]):
        sx += [b["units"], b["units"]]
        sy += [a["ratio"], b["ratio"]]
    cel.cline(ax, sx, sy, "k4", 1, lw=0.9, rim=2.0, z=3.0)
    for p in pts:                                           # counted columns of Table 7a and full K=4 (Table 1)
        k = _red_kind(p)
        if k == "counted":
            exact = "exact" in p["label"]
            ax.scatter([p["units"]], [p["ratio"]], marker="s", s=4.4 ** 2, fc=cel.PAPER if exact else cel.light("k4"),
                       ec=cel.line("k4"), lw=0.6, zorder=5)
        elif k == "full":
            cel.discs(ax, [p["units"]], [p["ratio"]], "k4", 2, d=5.6, lw=0.55, z=5, cast=True)
    cel.discs(ax, [uB], [1.0], "chainb", 1, d=6.6, lw=0.6, z=6, cast=True)
    # the discriminating test, metered end to end by flopscope (Table 12): the compressed layer-4 chain with dense
    # (2,1,1) kernels (m1) and the same chain with its (2,1,1) births as 32 matrix pairs over all 16 layers (p32)
    # and as 8 pairs (p8), hexagons joined as one metered line; m1 sits at the counted chain's raw
    E = R["e2e"]
    m1, p8, p32 = E["m1_net0"], E["p8_net0"], E["p32_net0"]
    cel.cline(ax, [m1["units"], p32["units"], p8["units"]], [m1["ratio"], p32["ratio"], p8["ratio"]], "k4", 2,
              lw=0.7, rim=1.4, z=5.2, shadow=False)
    # p32 and p8 sit 2.5% apart in units and 1.2x apart in raw: at the legend's hexagon size (5.4 pt) the two marks
    # clear each other; p8, the coarser compression, is drawn light and under p32
    for q, fc, z in ((m1, cel.mid("k4"), 5.5), (p8, cel.light("k4"), 5.5), (p32, cel.mid("k4"), 5.6)):
        ax.scatter([q["units"]], [q["ratio"]], marker="h", s=5.4 ** 2, fc=fc, ec=cel.line("k4"), lw=0.6, zorder=z)

    # the counted layer-4 chain, compressed as run, ties chain B at 602 units
    c4 = [p for p in pts if p["label"] == "full from layer 4, compressed as counted"][0]
    t4 = R["tie"](c4)
    ax.plot([t4, c4["units"]], [c4["ratio"], c4["ratio"]], color=cel.line("k4"), lw=0.5, ls=(0, (1.2, 1.2)), zorder=2.6)
    ax.scatter([t4], [c4["ratio"]], marker="|", s=5.0 ** 2, color=cel.line("k4"), lw=0.8, zorder=5)
    say(ax, t4, c4["ratio"], f"{t4:,.0f}", ha="right", va="top", transform=nudge(ax, -2.5, -2.0),
        color=cel.deeper(cel.line("k4"), 0.14), **cel.BOLD)
    say(ax, c4["units"], c4["ratio"], f"{c4['units']:,.0f}", ha="left", va="center", transform=nudge(ax, 8.0),
        color=cel.deeper(cel.line("k4"), 0.14))
    fk = [p for p in pts if _red_kind(p) == "full"][0]
    say(ax, fk["units"], fk["ratio"], "full K=4", ha="center", va="top", transform=nudge(ax, 0, -4.5),
        color=cel.deeper(cel.line("k4"), 0.14))
    _metered_labels(ax, R, c4, m1, p8, p32)
    # the focus: the best state-deletion rule, 20,586 units at 0.615, ties chain B at 276 units (75x its units); the
    # same births with rank-4 non-tree pairs, 3,258 units at 0.788, tie at 215 (15x)
    w = [p for p in pts if p["label"] == "window_s4_b9_treeall"][0]
    q4 = [q for q in path2 if q["nt"] == 4][0]
    tw, t4q = R["tie"](w), q4["tie"]
    ax.plot([tw, w["units"]], [w["ratio"], w["ratio"]], color=cel.mid("focus"), lw=0.6, ls=(0, (1.2, 1.6)),
            zorder=4.5)
    ax.scatter([tw], [w["ratio"]], marker="|", s=5.0 ** 2, color=cel.mid("focus"), lw=0.8, zorder=5)
    ax.plot([t4q, q4["units"]], [q4["ratio"], q4["ratio"]], color=cel.line("focus"), lw=0.8, ls=(0, (1.2, 1.2)),
            zorder=4.6)
    ax.scatter([t4q], [q4["ratio"]], marker="|", s=5.4 ** 2, color=cel.line("focus"), lw=0.9, zorder=5)
    cel.cline(ax, [w["units"]] + [q["units"] for q in path2], [w["ratio"]] + [q["ratio"] for q in path2], "focus", 1,
              lw=0.7, rim=1.6, z=4.7, shadow=False)
    cel.diamonds(ax, [q["units"] for q in path2], [q["ratio"] for q in path2], "focus", 1, d=6.0, lw=0.55, z=6.6)
    cel.discs(ax, [w["units"]], [w["ratio"]], "focus", 1, d=6.0, lw=0.6, z=6.5, cast=True)
    cel.chip(ax, 4300, w["ratio"],
             rf"{w['units'] / tw:.0f}$\times$ $\rightarrow$ {q4['units'] / t4q:.0f}$\times$ the units",
             "focus", size=FSS + 0.4, weight="bold", level=1, ha="center", va="top", transform=nudge(ax, 0, -4.5))
    say(ax, 1024, yhi, "budget", ha="left", va="top", transform=nudge(ax, 2.5, -2.5), color=INK)
    # key: what each mark is
    key = [("fed", "fed K=4 pieces"), ("weight", "cut by read weight"), ("rule", "other deletions"),
           ("window", "early window"), ("pair", "compressed"), ("sq", "counted"), ("hex", "metered")]
    for j, (k, t) in enumerate(key):                # in the empty corner under the 602 run, right of the score line
        kx, ky = 4.6e3, 0.225 / 1.29 ** j
        if k == "hex":
            ax.scatter([kx], [ky], marker="h", s=5.4 ** 2, fc=cel.mid("k4"), ec=cel.line("k4"), lw=0.6, zorder=5)
        elif k == "sq":
            ax.scatter([kx], [ky], marker="s", s=4.2 ** 2, fc=cel.PAPER, ec=cel.line("k4"), lw=0.6, zorder=5)
        elif k == "pair":
            cel.diamonds(ax, [kx], [ky], "k4", 1, d=5.4, lw=0.45, z=5)
        else:
            role, lev = style[k]
            cel.discs(ax, [kx], [ky], role, lev, d=4.2, lw=0.4, z=5, spark=False)
        say(ax, kx, ky, t, ha="left", va="center", transform=nudge(ax, 4.2), color=MUTED)
    ax.set_xticks([1e3, 1e4, 1e5, 1e6])
    ax.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
                  ["0.1", "0.2", "0.5", "chain B", "2", "5", "10", "20", "50"])
    ax.get_yticklabels()[3].set_color(cel.deeper(cel.line("chainb"), 0.16))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cel.axes_style(ax)
    ax.set_xlabel("units per network, counted or metered", labelpad=1.5)
    cel.sheet_panel(ax, letter="a", header=7.0, stack=2.0, which="0101", peg_at=0.84)
    log_ruler(ax, "x", 2, 6)
    cel.ylab(ax, r"final raw $\div$ chain B, network 0", dx=-0.07).set_y(1.0 + 12.0 / 72 / cel.box_of(ax)[3])

    # the specimen: which births the 20,586-unit rule keeps, layer by layer (its plan, lab/reduce_s1)
    fx = canvas(fig, z=9)
    cx0, cy0, cw, ch = 3.20, 2.27, 0.88, 0.80
    cel.card(fig, cx0, cy0, cw, ch, role="gold")
    rows = [("tree", range(4, 14)), ("non-tree", range(4, 10)), ("star", range(4, 10)), (r"C·$\kappa_3$", ())]
    cs = 0.042
    gx = cx0 + cw - 0.05 - 10 * cs
    for i, (nm, kept) in enumerate(rows):
        yy = cy0 + ch - 0.335 - i * (0.10 if matplotlib.__version__.startswith("3.10.") else 0.088)
        fx.text(gx - 0.04, yy + cs * 0.42, nm, ha="right", va="center", fontsize=FSS, color=INK)
        for j, layer in enumerate(range(4, 14)):
            on = layer in kept
            fam = "focus" if nm == "non-tree" else "k4"
            fx.add_patch(Rectangle((gx + j * cs, yy), cs * 0.84, cs * 0.84, fc=cel.shades(fam)[3] if on else cel.PAPER,
                                   ec=cel.line(fam), lw=0.3, zorder=9))
    fx.text(cx0 + cw / 2, cy0 + ch - 0.05, f"births kept, units\n{w['units']:,.0f} $\\rightarrow$ {q4['units']:,.0f}", ha="center", va="top", fontsize=FSS,
            color=cel.deeper(cel.line("focus"), 0.16), linespacing=1.0, **cel.BOLD)
    fx.text(gx, cy0 + 0.035, "4", ha="left", va="bottom", fontsize=FSS, color=MUTED)
    fx.text(gx + 10 * cs, cy0 + 0.035, "13", ha="right", va="bottom", fontsize=FSS, color=MUTED)
    fx.text(gx - 0.04, cy0 + 0.035, "layer", ha="right", va="bottom", fontsize=FSS, color=MUTED)
    lx, ly = inch(ax, w["units"], w["ratio"])
    fx.plot([cx0 + 0.12, lx + 0.015], [cy0, ly + 0.05], color=cel.line("focus"), lw=0.5, zorder=8)

    # ---- b: the layer-4 column by family: share of the units, of the first-order read weight, and the gain carried
    bx = cel.axes(fig, 5.25, 0.40, 6.62, 3.21)
    bx.set_xlim(0, 1)
    bx.set_ylim(0, 3.2)
    bx.axis("off")
    U, RW, G = R["units"], R["rw"]["v1_n0"], R["gain"]
    tot = sum(U.values())
    rest_u = tot - U["ck3"] - U["nontree"]
    fams = [(r"C·$\kappa_3$ births", U["ck3"] / tot, RW["ck3"], G["ck3"], "k4"),
            ("non-tree births", U["nontree"] / tot, RW["nontree"], G["nontree"], "k4"),
            (f"the other {rest_u:,.0f} units", rest_u / tot, RW["tree"] + RW["star"], 1 / G["tree_alone_over_B"], "k3")]
    sx0, sx1 = 0.04, 0.43                                  # shares 0-70%
    gx0, gx1 = 0.56, 0.74                                  # gain, log 0.9-2.2
    gpos = lambda g: gx0 + (gx1 - gx0) * (np.log(g) - np.log(0.9)) / (np.log(2.2) - np.log(0.9))
    spos = lambda s: sx0 + (sx1 - sx0) * s / 0.70
    titles = []
    for i, (nm, su, rw, g, role) in enumerate(fams):
        y = 2.42 - i * 0.98
        titles.append(say(bx, sx0, y + 0.30, nm, ha="left", va="bottom", color=INK, **cel.BOLD))
        cel.hbar(bx, y + 0.11, sx0, spos(su), role, 1, h=4.6, cast=0.8)
        hot = i == 1
        cel.hbar(bx, y - 0.13, sx0, spos(rw), "focus" if hot else "k3", 1 if hot else 0, h=4.6, cast=0.8)
        say(bx, spos(su), y + 0.11, f"{su:.0%}", ha="left", va="center", transform=nudge(bx, 2.6),
            color=cel.deeper(cel.line(role), 0.14))
        say(bx, spos(rw), y - 0.13, f"{rw:.0%}", ha="left", va="center", transform=nudge(bx, 2.6),
            color=cel.deeper(cel.line("focus"), 0.16) if hot else MUTED, **(cel.BOLD if hot else {}))
        lo, hi = sorted((gpos(1.0), gpos(g)))
        cel.hbar(bx, y - 0.01, lo, hi, role if g > 1 else "k3", 2 if g > 1 else 0, h=6.0, cast=0.8)
        say(bx, gpos(max(g, 1.0)), y - 0.01, rf"{g:.2f}$\times$" if g > 1 else rf"{1 / g:.2f}$\times$ worse",
            ha="left", va="center", transform=nudge(bx, 2.6), color=INK if g > 1 else MUTED)
    rule(bx, gpos(1.0), 0.16, 2.92, avoid=titles, color=cel.LINE, lw=0.5, zorder=2)   # broken where a family title runs
    say(bx, sx0, 2.98, "units", ha="left", va="bottom", color=cel.deeper(cel.line("k4"), 0.14), **cel.BOLD)
    say(bx, 0.225, 2.98, "read weight", ha="left", va="bottom", color=MUTED)
    say(bx, gpos(1.0), 2.98, "gain", ha="left", va="bottom", color=INK, transform=nudge(bx, 3.2))
    cel.sheet_panel(bx, letter="b", header=7.0, stack=2.0, which="0101", peg_at=0.70)
    cel.ylab(bx, "60,549 units by state family", dx=0.0).set_y(1.0 + 12.0 / 72 / cel.box_of(bx)[3])
    save(fig, output, "fig5_where_602_is_bound")


def render_all(namespace, output):
    setup()
    output = Path(output)
    N = {row["key"]: row["value"] for row in namespace["ROWS"]}
    for figure in (figure_one, figure_two, figure_three, figure_four, figure_five):
        figure(namespace, N, output)
