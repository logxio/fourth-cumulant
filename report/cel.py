"""Cel layer of the report's figures: flat fill, one hard shadow inside the mark, a highlight on key marks,
line art in the family's middle tone, and vector screentone for bands and panel grounds.

Every color comes from palette.py. A mark has three levels: pale (fill L, shadow M, line D), emphasized (fill M,
shadow D, line one step deeper) and accent (fill D, the saturated reading a figure points at; at most three a
figure). No new hue enters: deeper steps keep the family's hue and saturation and only lower its lightness.
"""
import colorsys

import matplotlib
import numpy as np
from matplotlib import hatch as _hatch
from matplotlib import patheffects as pe
from matplotlib.collections import PathCollection
from matplotlib.colors import to_hex, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform, blended_transform_factory, offset_copy

from palette import HUES, INK, MUTED, hue, mix

# Screentone dots are near-zero strokes; path simplification would drop them.
matplotlib.rcParams['path.simplify'] = False

PAPER = '#FFFFFF'
LINE = HUES['gray'][0]                    # axis line art
TICK = HUES['gray'][1]
GROUND = '#DCDFE4'                        # panel screentone dots, between gray L and M
SHADE = mix(HUES['gray'][2], HUES['gray'][1], 0.45)   # a cel's hard shadow on the paper
BOLD = {'fontweight': 'bold', 'fontfamily': 'STIXGeneral'}   # the text face has no bold cut on this system


def deeper(c, dl=0.10):
    """Same hue and saturation, lightness lowered by dl."""
    h, l, s = colorsys.rgb_to_hls(*to_rgb(c))
    return to_hex(colorsys.hls_to_rgb(h, max(0.0, l - dl), s))


def ramp(role, level=0):
    """(fill, shadow, line) of one mark."""
    d, m, l, _ = HUES[hue(role)]
    if level == 0:
        return l, m, d
    if level == 1:
        return m, d, deeper(d, 0.12)
    return d, deeper(d, 0.11), deeper(d, 0.22)


def wash(role):
    return HUES[hue(role)][3]


def light(role):
    return HUES[hue(role)][2]


def mid(role):
    return HUES[hue(role)][1]


def line(role):
    return HUES[hue(role)][0]


# ---------------------------------------------------------------- screentone
class _ToneDots(_hatch.HatchPatternBase):
    """A '.' in a hatch string: a staggered grid of round dots, 6 rows per inch per '.', the dot diameter set by
    the patch's hatch line width (each dot is a near-zero stroke with a round cap)."""

    def __init__(self, hatch, density):
        n = self.num_rows = hatch.count('.') * density
        self.num_shapes = (n // 2 + 1) * (n + 1) + (n // 2) * n if n else 0
        self.num_vertices = 2 * self.num_shapes

    def set_vertices_and_codes(self, vertices, codes):
        n = self.num_rows
        off = 1.0 / n
        pts = []
        for row in range(n + 1):
            cols = np.linspace(0, 1, n + 1) if row % 2 == 0 else np.linspace(off / 2, 1 - off / 2, n)
            pts.extend((c, row * off) for c in cols)
        pts = np.asarray(pts)
        vertices[0::2] = pts
        vertices[1::2] = pts + [off * 1e-3, 0]
        codes[0::2] = Path.MOVETO
        codes[1::2] = Path.LINETO


_hatch._hatch_types[_hatch._hatch_types.index(_hatch.SmallFilledCircles)] = _ToneDots


def toned(patch, color, rows=5, dot=0.5):
    """Give any patch a screentone: rows x 6 dots per inch, dot diameter in pt."""
    patch.set_hatch('.' * rows)
    if hasattr(patch, "set_hatchcolor"):
        patch.set_hatchcolor(color)
    else:  # Matplotlib 3.10 exposes the hatch color on the artist, without a setter.
        patch._hatch_color = to_rgba(color)
    patch.set_hatch_linewidth(dot)
    patch.set_capstyle('round')
    return patch


def tone_rect(ax, x, y, w, h, color, rows=5, dot=0.5, z=0.2, transform=None, edge=None, lw=0.35, **kw):
    p = Rectangle((x, y), w, h, transform=transform or ax.transData, facecolor='none',
                  edgecolor=edge or 'none', lw=lw if edge else 0, zorder=z, **kw)
    ax.add_patch(toned(p, color, rows, dot))
    return p


def ground(ax, color=GROUND, rows=5, dot=0.5):
    """Pale screentone over the whole plotting area."""
    return tone_rect(ax, 0, 0, 1, 1, color, rows, dot, z=0.05, transform=ax.transAxes)


# Screentone in three dot sizes, the manga way of stepping light to dark with hard edges, plus diagonal hatching
# for the darkest shadow. Tier 1 is a panel ground, tier 2 a band or zone, tier 3 a shadow or dominated region.
TIERS = {1: (5, 0.42), 2: (5, 0.80), 3: (5, 1.15)}


def tier(patch, color, level=1):
    return toned(patch, color, *TIERS[level])


def tier_rect(ax, x, y, w, h, color, level=1, z=0.2, transform=None, **kw):
    return tone_rect(ax, x, y, w, h, color, *TIERS[level], z=z, transform=transform, **kw)


def hatched(patch, color, lw=0.3, n=4, kind='/'):
    """Diagonal hatching (n x 6 lines per inch), used only inside the darkest shadows."""
    patch.set_hatch(kind * n)
    if hasattr(patch, "set_hatchcolor"):
        patch.set_hatchcolor(color)
    else:
        patch._hatch_color = to_rgba(color)
    patch.set_hatch_linewidth(lw)
    return patch


def drop(artist, dx=1.4, dy=-1.4, color=SHADE):
    """Hard cast shadow of a patch, offset in pt to the lower right (light from the upper left)."""
    artist.set_path_effects([pe.SimplePatchShadow(offset=(dx, dy), shadow_rgbFace=color, alpha=1, hatch=None),
                             pe.Normal()])
    return artist


# ---------------------------------------------------------------- unit conversions
def _axes_inches(ax):
    fw, fh = ax.figure.get_size_inches()
    b = ax.get_position()
    return b.width * fw, b.height * fh


def pt_dx(ax, pt):
    """pt as a data-x length (linear axis, limits already set)."""
    x0, x1 = ax.get_xlim()
    return pt / 72 * (x1 - x0) / _axes_inches(ax)[0]


def pt_dy(ax, pt):
    y0, y1 = ax.get_ylim()
    return pt / 72 * (y1 - y0) / _axes_inches(ax)[1]


# ---------------------------------------------------------------- marks
def _arc(cx, cy, t0, t1, n):
    t = np.radians(np.linspace(t0, t1, n))
    return np.column_stack([cx + np.cos(t), cy + np.sin(t)])


def _closed(pts):
    pts = np.vstack([pts, pts[:1]])
    codes = np.full(len(pts), Path.LINETO, dtype=Path.code_type)
    codes[0], codes[-1] = Path.MOVETO, Path.CLOSEPOLY
    return Path(pts, codes)


def _crescent(a=0.30):
    """Unit disc minus the same disc moved up-left by a: the hard shadow on the lower right."""
    c1 = np.array([-a, a])
    h = np.sqrt(1 - (a * np.sqrt(2) / 2) ** 2)
    mid_ = c1 / 2
    u = np.array([1, 1]) / np.sqrt(2)
    p_hi, p_lo = mid_ + h * u, mid_ - h * u
    t_hi, t_lo = np.degrees(np.arctan2(*p_hi[::-1])), np.degrees(np.arctan2(*p_lo[::-1]))
    outer = _arc(0, 0, t_hi, t_lo - 360 if t_lo > t_hi else t_lo, 40)
    q_lo, q_hi = p_lo - c1, p_hi - c1
    s_lo, s_hi = np.degrees(np.arctan2(*q_lo[::-1])), np.degrees(np.arctan2(*q_hi[::-1]))
    inner = _arc(c1[0], c1[1], s_lo, s_hi if s_hi > s_lo else s_hi + 360, 30)
    return _closed(np.vstack([outer, inner]))


DISC = _closed(_arc(0, 0, 0, 360, 48)[:-1])
CAST = _closed(_arc(0.26, -0.26, 0, 360, 48)[:-1])     # the disc's hard shadow on the sheet below
CRESCENT = _crescent()
SPARK = _closed(np.column_stack([-0.40 + 0.22 * np.cos(np.radians(np.arange(0, 360, 18))),
                                 0.40 + 0.22 * np.sin(np.radians(np.arange(0, 360, 18)))]))
STAR = _closed(np.array([(np.cos(a) * r, np.sin(a) * r) for a, r in
                         zip(np.radians(np.arange(90, 450, 45)), [1, 0.28] * 4)]))
STAR_SHADE = _closed(np.array([[0, 0], [0, -1], [0.198, -0.198], [1, 0]]))  # lower-right facets
DIAMOND = _closed(np.array([[0, 1], [1, 0], [0, -1], [-1, 0]]))
DIAMOND_SHADE = _closed(np.array([[0.62, 0.38], [1, 0], [0, -1], [-0.38, -0.62]]))
DIAMOND_SPARK = _closed(np.array([[-0.28, 0.50], [-0.12, 0.34], [-0.28, 0.18], [-0.44, 0.34]]))


def _marks(ax, xs, ys, layers, d, z, transform=None):
    """Draw every point's layers before the next point, so overlapping marks occlude like cut-out cels.
    layers: [(path, face, edge, lw)], paths in a unit box of half-width 1; d is the mark's width in pt."""
    xs, ys = np.atleast_1d(xs), np.atleast_1d(ys)
    n, k = len(xs), len(layers)
    offsets = np.repeat(np.column_stack([xs, ys]), k, axis=0)
    paths = [l[0] for l in layers]
    face = [l[1] for l in layers]
    edge = [l[2] for l in layers]
    lws = [l[3] for l in layers]
    # a unit-radius path scaled by sqrt(size) pt: size (d/2)^2 gives a mark d pt wide
    col = PathCollection(paths, sizes=[(d / 2) ** 2], offsets=offsets, offset_transform=transform or ax.transData,
                         facecolors=face, edgecolors=edge, linewidths=lws, zorder=z)
    col.set_transform(IdentityTransform())
    ax.add_collection(col, autolim=False)
    return col


def discs(ax, xs, ys, role, level=0, d=5.0, lw=0.6, z=5, spark=True, transform=None, colors=None, cast=False):
    """Small cel discs: flat fill, crescent shadow lower right, highlight dot upper left, line ring. cast=True adds
    the disc's hard shadow on the sheet below; marks drawn later cover earlier ones with their shadow too."""
    f, s, l = colors or ramp(role, level)
    layers = [(CAST, SHADE, 'none', 0)] if cast else []
    layers += [(DISC, f, l, lw), (CRESCENT, s, 'none', 0)]
    if spark:
        layers.append((SPARK, PAPER, 'none', 0))
    layers.append((DISC, 'none', l, lw))
    return _marks(ax, xs, ys, layers, d, z, transform)


def diamonds(ax, xs, ys, role, level=0, d=6.5, lw=0.6, z=5, colors=None):
    f, s, l = colors or ramp(role, level)
    layers = [(DIAMOND, f, 'none', 0), (DIAMOND_SHADE, s, 'none', 0), (DIAMOND_SPARK, PAPER, 'none', 0),
              (DIAMOND, 'none', l, lw)]
    return _marks(ax, xs, ys, layers, d, z)


def hbar(ax, y, lo, hi, role, level=0, h=3.4, lw=0.6, z=3, spark=False, shadow=0.42, colors=None, tone=None,
         cast=0.0, gleam=False, caps=0.0):
    """Horizontal cel bar from lo to hi at y: the outline sits inside lo..hi so the ends are the data.
    tone: screentone tier laid over the face (an interval band); cast: pt of hard shadow on the sheet below, kept
    inside lo..hi so it never reads as a longer bar; gleam: a thin highlight line along the top; caps: pt height of
    fine end brackets."""
    f, s, l = colors or ramp(role, level)
    hy, ix = pt_dy(ax, h) / 2, pt_dx(ax, lw / 2)
    x0, w = lo + ix, hi - lo - 2 * ix
    if cast:
        c = pt_dx(ax, cast)
        ax.add_patch(Rectangle((x0 + c, y - hy - pt_dy(ax, cast)), max(w - c, 0), pt_dy(ax, cast) + hy, fc=SHADE,
                               ec='none', lw=0, zorder=z - 0.02))
    ax.add_patch(Rectangle((x0, y - hy), w, 2 * hy, fc=f, ec='none', lw=0, zorder=z))
    if tone:
        ax.add_patch(tier(Rectangle((x0, y - hy), w, 2 * hy, fc='none', ec='none', lw=0, zorder=z + 0.005), s, tone))
    ax.add_patch(Rectangle((x0, y - hy), w, 2 * hy * shadow, fc=s, ec='none', lw=0, zorder=z + 0.01))
    if gleam:
        ax.plot([x0 + pt_dx(ax, 1.6), x0 + w - pt_dx(ax, 1.6)], [y + hy * 0.55] * 2, color=PAPER, lw=0.45,
                solid_capstyle='round', zorder=z + 0.02)
    if spark:
        ax.plot([x0 + pt_dx(ax, 2.0), x0 + pt_dx(ax, 6.5)], [y + hy * 0.42] * 2, color=PAPER, lw=0.7,
                solid_capstyle='round', zorder=z + 0.02)
    ax.add_patch(Rectangle((x0, y - hy), w, 2 * hy, fc='none', ec=l, lw=lw, zorder=z + 0.03))
    if caps:
        cy = pt_dy(ax, caps) / 2
        for e in (lo, hi):
            ax.plot([e, e], [y - cy, y + cy], color=l, lw=0.5, zorder=z + 0.04, solid_capstyle='butt')


def cline(ax, x, y, role, level=0, lw=1.0, rim=2.2, z=3, shadow=True, ls='-', color=None, rim_color=None,
          inner=None):
    """A line with a pale rim stroke under it (cel edge light) and a hard shadow copy offset to the lower right.
    inner: a hairline of that color drawn along the stroke's middle (the C tier's fine inner line)."""
    f, s, l = ramp(role, level)
    c = color or l
    ax.plot(x, y, color=rim_color or PAPER, lw=lw + rim, solid_capstyle='round', solid_joinstyle='round', zorder=z)
    if shadow:
        tr = offset_copy(ax.transData, ax.figure, x=0.45, y=-0.55, units='points')
        ax.plot(x, y, color=s, lw=lw, ls=ls, transform=tr, solid_capstyle='round', solid_joinstyle='round',
                zorder=z + 0.01)
    ax.plot(x, y, color=c, lw=lw, ls=ls, solid_capstyle='round', solid_joinstyle='round', zorder=z + 0.02)
    if inner:
        ax.plot(x, y, color=inner, lw=max(0.22, lw * 0.26), ls=ls, solid_capstyle='round', solid_joinstyle='round',
                zorder=z + 0.03)


# ---------------------------------------------------------------- axes and panel furniture
def axes_style(ax, left=True, bottom=True):
    """Open L-frame in mid-gray line art: spines thicker than ticks (outer heavy, inner light)."""
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_visible(bottom)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(LINE)
        ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(colors=LINE, labelcolor=INK, width=0.45, length=2.2, direction='out', pad=2)
    if not left:
        ax.tick_params(axis='y', length=0, pad=3)
    ax.set_axisbelow(True)


def ruler(ax, ticks, levels=None, length=(3.4, 2.2, 1.3), axis='x', color=None, lw=0.4, top=False, z=1):
    """A ruler of tick marks under the x axis (or left of the y axis): levels[i] 0 = major, 1 = half, 2 = fine.
    top=True hangs the same marks down from the top edge (or in from the right edge), inside the panel."""
    color = color or LINE
    levels = levels if levels is not None else [2] * len(ticks)
    if axis == 'x':
        tr = blended_transform_factory(ax.transData, ax.transAxes)
        h_ax = _axes_inches(ax)[1] * 72
        y0, sgn = (1, -1) if top else (0, -1)
        for t, lv in zip(ticks, levels):
            ax.plot([t, t], [y0, y0 + sgn * length[lv] / h_ax], color=color, lw=lw if lv else lw + 0.1,
                    transform=tr, clip_on=False, zorder=z, solid_capstyle='butt')
    else:
        tr = blended_transform_factory(ax.transAxes, ax.transData)
        w_ax = _axes_inches(ax)[0] * 72
        x0, sgn = (1, -1) if top else (0, -1)
        for t, lv in zip(ticks, levels):
            ax.plot([x0, x0 + sgn * length[lv] / w_ax], [t, t], color=color, lw=lw if lv else lw + 0.1, transform=tr,
                    clip_on=False, zorder=z, solid_capstyle='butt')


def ruler_ticks(lo, hi, major, half, fine):
    """Ruler marks between lo and hi: (positions, levels 0 major, 1 half, 2 fine)."""
    pos, lv = [], []
    for k in range(int(np.ceil(lo / fine - 1e-9)), int(np.floor(hi / fine + 1e-9)) + 1):
        t = round(k * fine, 6)
        pos.append(t)
        if abs(t / major - round(t / major)) < 1e-6:
            lv.append(0)
        elif half and abs(t / half - round(t / half)) < 1e-6:
            lv.append(1)
        else:
            lv.append(2)
    return pos, lv


def rulers(ax, x=None, y=None):
    """The C tier frame ruler on all four edges: under the x axis and left of the y axis the full ruler, hanging
    inside from the top and the right edge a lighter copy. x, y: (major, half, fine) steps; None leaves an edge
    plain (a category axis). The tick marks themselves are switched off, the ruler replaces them."""
    if x:
        pos, lv = ruler_ticks(*ax.get_xlim(), *x)
        ax.tick_params(axis='x', length=0, pad=4.2)
        ruler(ax, pos, lv)
        ruler(ax, pos, lv, top=True, color=TICK, length=(2.6, 1.8, 1.1), lw=0.35, z=1.65)
    if y:
        lo, hi = sorted(ax.get_ylim())
        pos, lv = ruler_ticks(lo, hi, *y)
        ax.tick_params(axis='y', length=0, pad=4.2)
        ruler(ax, pos, lv, axis='y')
        ruler(ax, pos, lv, axis='y', top=True, color=TICK, length=(2.6, 1.8, 1.1), lw=0.35, z=1.65)


def fine_grid(ax, xs=(), ys=(), color=None, lw=0.3, z=0.1):
    color = color or HUES['gray'][2]
    for x in xs:
        ax.axvline(x, color=color, lw=lw, zorder=z)
    for y in ys:
        ax.axhline(y, color=color, lw=lw, zorder=z)


def brackets(fig, x, y, w, h, arm=5, lw=0.6, color=None, z=6):
    """Corner brackets (registration marks) at the four corners of a box in inches; arm in pt."""
    color = color or LINE
    a = arm / 72
    tr = fig.dpi_scale_trans
    for cx, cy, sx, sy in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        fig.add_artist(Line2D([cx + sx * a, cx, cx], [cy, cy, cy + sy * a], transform=tr, color=color, lw=lw,
                              zorder=z, solid_capstyle='butt'))


def badge(fig, x, y, text, role='gray', level=0, size=7, w=None, h=None, z=8, gid=None):
    """A small cel chip at (x, y) inches, lower-left corner: flat face, hard shadow, line frame, bold label."""
    f, s, l = ramp(role, level)
    w = w if w is not None else (0.13 + 0.055 * (len(text) - 1))
    h = h if h is not None else 0.13
    tr = fig.dpi_scale_trans
    fig.add_artist(FancyBboxPatch((x + 1.1 / 72, y - 1.1 / 72), w, h, boxstyle='round,pad=0,rounding_size=0.018',
                                  transform=tr, fc=s, ec='none', zorder=z, gid=gid))
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.018', transform=tr, fc=f,
                                  ec=l, lw=0.6, zorder=z + 0.1, gid=gid))
    fig.text((x + w / 2) / fig.get_size_inches()[0], (y + h / 2 - 0.004) / fig.get_size_inches()[1], text,
             fontsize=size, **BOLD, color=deeper(l, 0.18) if role != 'gray' else INK, ha='center',
             va='center', zorder=z + 0.2, gid=gid)


def axes(fig, x0, y0, x1, y1):
    """Axes from inch corners; the face is transparent so the sheet behind it shows."""
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x0 / fw, y0 / fh, (x1 - x0) / fw, (y1 - y0) / fh])
    ax.patch.set_facecolor('none')
    return ax


def box_of(ax):
    fw, fh = ax.figure.get_size_inches()
    b = ax.get_position()
    return b.x0 * fw, b.y0 * fh, b.width * fw, b.height * fh


def _rect(fig, x, y, w, h, **kw):
    r = Rectangle((x, y), w, h, transform=fig.dpi_scale_trans, **kw)
    fig.add_artist(r)
    return r


def sheet(fig, x, y, w, h, drop_pt=2.2, stack=0.0, z=-5, frame=True):
    """A cel sheet at (x, y, w, h) inches. Its hard shadow falls drop_pt to the lower right as flat shade with fine
    hatching; stack > 0 puts one more sheet that far behind it, so the panel reads as a stack."""
    d = drop_pt / 72
    if stack:
        s = stack / 72
        _rect(fig, x + s + d, y - s - d, w, h, fc=SHADE, ec='none', lw=0, zorder=z - 0.4)
        _rect(fig, x + s, y - s, w, h, fc=PAPER, ec=TICK, lw=0.45, zorder=z - 0.3)
    sh = _rect(fig, x + d, y - d, w, h, fc=SHADE, ec='none', lw=0, zorder=z - 0.2)
    hatched(sh, HUES['gray'][1], lw=0.32, n=5)
    _rect(fig, x, y, w, h, fc=PAPER, ec=LINE if frame else 'none', lw=0.8 if frame else 0, zorder=z)
    fig._cel_sheets = getattr(fig, '_cel_sheets', []) + [(x, y, w, h)]     # read by figcheck
    if frame:
        i = 1.7 / 72
        _rect(fig, x + i, y + i, w - 2 * i, h - 2 * i, fc='none', ec=TICK, lw=0.3, zorder=z + 0.05)
    return x, y, w, h


def sheet_panel(ax, letter=None, header=8.0, stack=0.0, which='1111', grid=(), ygrid=(), peg_at=0.5, drop_pt=2.2):
    """The C tier panel: the axes area on a cel sheet. Above the data a header strip of `header` pt carries the
    peg holes (at peg_at of the width); a letter tab sits on the upper left corner, registration marks off the
    corners (which masks any that would meet tick labels). Spines give way to the sheet's double frame; the ground is
    tier 1 screentone with a fine grid. Returns the sheet box in inches."""
    fig = ax.figure
    x, y, w, h = box_of(ax)
    hd = header / 72
    sheet(fig, x, y, w, h + hd, drop_pt=drop_pt, stack=stack)
    if header:
        fig.add_artist(Line2D([x, x + w], [y + h, y + h], transform=fig.dpi_scale_trans, color=TICK, lw=0.35,
                              zorder=-4.6))
        pegs(fig, x + w * peg_at, y + h + hd / 2, s=min(0.92, (header - 2.4) / 4.0))
    ax.spines[['left', 'bottom', 'top', 'right']].set_visible(False)
    tier_rect(ax, 0, 0, 1, 1, GROUND, 1, z=0.05, transform=ax.transAxes)
    fine_grid(ax, xs=grid, ys=ygrid, color=HUES['gray'][2], lw=0.3)
    corners(fig, x, y, w, h + hd, which=which)
    if letter:
        badge(fig, x - 0.05, y + h + hd - 0.075, letter, 'gray', 0, gid='tab')
    return x, y, w, h + hd


def pegs(fig, cx, cy, s=1.0, z=-4.5):
    """Animation peg holes centred at (cx, cy) inches: a round hole between two slots, each punched through the
    sheet (shade inside, a crescent of deeper shade on the upper left wall, a line rim)."""
    tr = fig.dpi_scale_trans
    for dx, w, h in ((-8.6, 6.6, 2.5), (0, 2.8, 2.8), (8.6, 6.6, 2.5)):
        w_, h_ = w * s / 72, h * s / 72
        x0, y0 = cx + dx * s / 72 - w_ / 2, cy - h_ / 2
        r = min(w_, h_) / 2
        fig.add_artist(FancyBboxPatch((x0, y0), w_, h_, boxstyle=f'round,pad=0,rounding_size={r}', transform=tr,
                                      fc=HUES['gray'][3], ec='none', zorder=z))
        fig.add_artist(FancyBboxPatch((x0, y0 + h_ * 0.42), w_, h_ * 0.58,
                                      boxstyle=f'round,pad=0,rounding_size={r * 0.9}', transform=tr,
                                      fc=HUES['gray'][2], ec='none', zorder=z + 0.01))
        fig.add_artist(FancyBboxPatch((x0, y0), w_, h_, boxstyle=f'round,pad=0,rounding_size={r}', transform=tr,
                                      fc='none', ec=LINE, lw=0.45, zorder=z + 0.02))


def register(fig, cx, cy, r=1.7, arm=3.0, z=-4.5):
    """A registration mark: a ring and a cross, in the frame's line tone."""
    tr = fig.dpi_scale_trans
    t = np.linspace(0, 2 * np.pi, 33)
    fig.add_artist(Line2D(cx + r / 72 * np.cos(t), cy + r / 72 * np.sin(t), transform=tr, color=LINE, lw=0.35,
                          zorder=z))
    a = arm / 72
    for xs, ys in (([cx - a, cx + a], [cy, cy]), ([cx, cx], [cy - a, cy + a])):
        fig.add_artist(Line2D(xs, ys, transform=tr, color=LINE, lw=0.35, zorder=z, solid_capstyle='butt'))


def corners(fig, x, y, w, h, gap=3.2, which='1111'):
    """Registration marks just outside the four corners of a box (inches): lower left, lower right, upper left,
    upper right; which masks corners that would meet tick labels."""
    g = gap / 72
    for on, (cx, cy) in zip(which, ((x - g, y - g), (x + w + g, y - g), (x - g, y + h + g), (x + w + g, y + h + g))):
        if on == '1':
            register(fig, cx, cy)


def _wedges(n, r0, seed):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.06, 0.06, n)
    verts, codes = [], []
    for a, rin, rout, dw in zip(t, r0 + rng.uniform(0, 0.22, n), 1 - rng.uniform(0, 0.18, n),
                                rng.uniform(0.018, 0.04, n)):
        pts = [(rout * np.cos(a - dw), rout * np.sin(a - dw)), (rout * np.cos(a + dw), rout * np.sin(a + dw)),
               (rin * np.cos(a), rin * np.sin(a))]
        verts += pts + [pts[0]]
        codes += [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    return Path(np.array(verts), np.array(codes, dtype=Path.code_type))


def burst(ax, x, y, role, d=46, r0=0.36, n=40, z=0.3, seed=11, color=None, transform=None):
    """Manga focus lines: n thin wedges pointing at (x, y) from a d-pt circle down to r0 of it, in the family's
    pale tone. Clipped to the axes, drawn under the rulers and the data."""
    col = PathCollection([_wedges(n, r0, seed)], sizes=[(d / 2) ** 2], offsets=[(x, y)],
                         offset_transform=transform or ax.transData, facecolors=[color or light(role)],
                         edgecolors='none', linewidths=0, zorder=z, gid='texture')
    col.set_transform(IdentityTransform())
    ax.add_collection(col, autolim=False)
    return col


def _star(k, r_in, seed):
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2 * np.pi, 2 * k, endpoint=False) + np.pi / 2
    r = np.where(np.arange(2 * k) % 2 == 0, 1 - rng.uniform(0, 0.16, 2 * k), r_in + rng.uniform(-0.05, 0.05, 2 * k))
    return _closed(np.column_stack([r * np.cos(a), r * np.sin(a)]))


def flash(ax, x, y, role, d=22, k=11, r_in=0.66, z=0.32, seed=5, transform=None):
    """A flat burst shape behind a headline reading: wash fill, pale line, hard-edged spikes."""
    col = PathCollection([_star(k, r_in, seed)], sizes=[(d / 2) ** 2], offsets=[(x, y)],
                         offset_transform=transform or ax.transData, facecolors=[wash(role)],
                         edgecolors=[light(role)], linewidths=0.45, zorder=z)
    col.set_transform(IdentityTransform())
    ax.add_collection(col, autolim=False)
    return col


def chip(ax, x, y, text, role, size=7, ha='left', va='center', level=0, transform=None, weight='normal', z=7,
         pad=0.28, face=PAPER, color=None):
    """A value tag on the top layer: paper face, frame in the family's line tone, hard shadow on the layer below."""
    f, s, l = ramp(role, level)
    t = ax.text(x, y, text, fontsize=size, color=color or deeper(l, 0.14), ha=ha, va=va, zorder=z, **(BOLD if weight == 'bold' else {}),
                transform=transform or ax.transData,
                bbox=dict(boxstyle=f'round,pad={pad},rounding_size=0.2', fc=face, ec=l, lw=0.55))
    drop(t.get_bbox_patch(), 1.2, -1.2, s)
    return t


def view(ax, x0, x1, y0, y1, role='gray', z=4.5):
    """The viewfinder on the main panel: a hairline box with corner brackets around the magnified data."""
    c = deeper(line(role), 0.04)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc='none', ec=c, lw=0.35, ls=(0, (1.2, 1.0)), zorder=z))
    bx, by = (x1 - x0) * 0.22, (y1 - y0) * 0.22
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        ax.plot([cx + sx * bx, cx, cx], [cy, cy, cy + sy * by], color=c, lw=0.7, zorder=z + 0.01,
                solid_capstyle='butt', solid_joinstyle='miter')


def loupe(ax, rect, xlim, ylim, role='gray', drop_pt=1.6, links=(('ul', 'ur'), ('ll', 'lr'))):
    """A magnified inset: a small cel sheet at rect (inches x, y, w, h) showing xlim, ylim of the parent, a
    viewfinder on the parent and leader lines, each from a viewfinder corner to an inset corner."""
    from matplotlib.patches import ConnectionPatch
    fig = ax.figure
    x, y, w, h = rect
    sheet(fig, x, y, w, h, drop_pt=drop_pt, z=4.0)
    ins = axes(fig, x, y, x + w, y + h)
    ins.set_zorder(4.2)
    ins.set_xlim(xlim)
    ins.set_ylim(ylim)
    ins.set_xticks([])
    ins.set_yticks([])
    ins.spines[['left', 'bottom', 'top', 'right']].set_visible(False)
    view(ax, xlim[0], xlim[1], ylim[0], ylim[1], role)
    c = deeper(line(role), 0.04)
    pos = {'ll': (0, 0), 'lr': (1, 0), 'ul': (0, 1), 'ur': (1, 1)}
    for a, b in links:
        u, v = pos[a]
        con = ConnectionPatch(xyA=(xlim[u], ylim[v]), coordsA=ax.transData, xyB=pos[b], coordsB=ins.transAxes,
                              color=c, lw=0.4, ls=(0, (1.6, 1.2)), zorder=3.9)
        fig.add_artist(con)
    return ins


def card(fig, x, y, w, h, role='gray', drop_pt=1.2, z=8, r=0.02, fc=PAPER):
    """A small rounded card with a hard shadow (legend items, paper strips)."""
    f, s, l = ramp(role, 0)
    tr = fig.dpi_scale_trans
    p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', transform=tr, fc=fc, ec=l, lw=0.5,
                       zorder=z)
    fig.add_artist(drop(p, drop_pt, -drop_pt, s if role != 'gray' else SHADE))
    return p


def numeral(fig, cx, cy, n, role='gray', size=6.8, r=4.2, z=8.5):
    """A numbered cel dot: the card's index in a ring."""
    f, s, l = ramp(role, 1)
    tr = fig.dpi_scale_trans
    t = np.linspace(0, 2 * np.pi, 41)
    fig.add_artist(Polygon(np.column_stack([cx + r / 72 * np.cos(t), cy + r / 72 * np.sin(t)]), closed=True,
                           transform=tr, fc=f, ec=l, lw=0.5, zorder=z))
    fw, fh = fig.get_size_inches()
    fig.text(cx / fw, (cy - 0.004) / fh, str(n), fontsize=size, **BOLD,
             color=PAPER if role != 'gray' else INK, ha='center', va='center', zorder=z + 0.1)


def lanes(ax, ys, half=0.30, color=None, edge=None, z=0.08):
    """Flat pale lanes under the rows of a dot-and-interval panel, hairline edges."""
    x0, x1 = ax.get_xlim()
    for y in ys:
        ax.add_patch(Rectangle((x0, y - half), x1 - x0, 2 * half, fc=color or HUES['gray'][3], ec='none', lw=0,
                               zorder=z))
        for e in (y - half, y + half):
            ax.plot([x0, x1], [e, e], color=edge or HUES['gray'][2], lw=0.35, zorder=z + 0.01)


def ylab(ax, text, dx=0.0, size=7):
    """Axis title set horizontally above the axis; no figure text is rotated."""
    return ax.text(dx, 1.035, text, transform=ax.transAxes, ha='left', va='bottom', fontsize=size, color=INK)


def irow(ax, y, p, lo, hi, role, step=2, hot=False, h=4.8, d=7.6, caps=7.5, z=3):
    """A dot-and-interval row: the interval a cel bar whose fill takes depth step `step` of its family (tier 2
    screentone over it, cast shadow under it, end brackets), the point a cel disc; hot gives the panel's one accent."""
    cs = shades(role)
    f = cs[step]
    s = deeper(f, 0.08) if step < 4 else deeper(f, 0.06)
    l = line(role) if step < 3 else deeper(cs[4], 0.08)
    hbar(ax, y, lo, hi, role, 0, h=h, tone=2, cast=1.0, gleam=True, caps=caps, z=z, colors=(f, s, l))
    discs(ax, [p], [y], role, 2 if hot else 1, d=d, lw=0.65, z=z + 3, cast=True)


# ================================================================ depth steps, microtext and specimen tags
MICRO = HUES['gray'][0]
MARK = HUES['gold'][3]                    # champagne underlay of the sentence a page is shown for


def shades(role):
    """Five depth steps of one family, pale to deep: wash, light, mid, stroke, stroke deepened."""
    w_, l_, m_, d_ = wash(role), light(role), mid(role), line(role)
    return [w_, l_, m_, d_, deeper(d_, 0.10)]


def depth_of(v, lo, hi, n=5):
    """Index 0..n-1 of v in n equal steps over lo..hi (the step a mark's fill takes from shades())."""
    t = (v - lo) / (hi - lo) if hi > lo else 0.0
    return int(min(n - 1, max(0, np.floor(t * n))))


def _latex(s):
    for a, b in (('\\', ''), ('%', r'\%'), ('$', r'\$'), ('&', r'\&'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'),
                 ('}', r'\}'), ('~', '-'), ('^', '')):
        s = s.replace(a, b)
    return s


def _wrap(text, width_in, size):
    import textwrap
    return textwrap.wrap(text, max(8, int(width_in * 72 / (0.47 * size))))


def micro(fig, x, y, lines, size=2.3, lead=1.22, color=MICRO, bold=False, z=9.3):
    """Real text lines set as microtext from (x, y) inches downwards (baseline of the first line at y)."""
    fw, fh = fig.get_size_inches()
    for k, ln in enumerate(lines):
        fig.text(x / fw, (y - k * size * lead / 72) / fh, _latex(ln), fontsize=size, color=color, ha='left',
                 va='baseline', **(BOLD if bold else {}), zorder=z)
    return y - len(lines) * size * lead / 72


def eyelet(fig, cx, cy, r=1.25, z=9.5):
    """A metal eyelet: rim in the line tone, punched hole, a hard white highlight on the upper left of the rim."""
    tr = fig.dpi_scale_trans
    t = np.linspace(0, 2 * np.pi, 33)
    rr = r / 72
    fig.add_artist(Polygon(np.column_stack([cx + rr * np.cos(t), cy + rr * np.sin(t)]), closed=True, transform=tr,
                           fc=HUES['gray'][1], ec=deeper(LINE, 0.12), lw=0.35, zorder=z))
    fig.add_artist(Polygon(np.column_stack([cx + 0.48 * rr * np.cos(t), cy + 0.48 * rr * np.sin(t)]), closed=True,
                           transform=tr, fc=PAPER, ec='none', zorder=z + 0.01))
    a = np.linspace(np.radians(110), np.radians(200), 12)
    fig.add_artist(Line2D(cx + 0.78 * rr * np.cos(a), cy + 0.78 * rr * np.sin(a), transform=tr, color=PAPER, lw=0.45,
                          zorder=z + 0.02, solid_capstyle='round'))


def tag(fig, x, y, text, role='gray', size=6.8, h=0.125, z=9.4, string_to=None, level=0):
    """A paper hang tag: pointed left end, metal eyelet, hard shadow, the text in bold;
    string_to (inches) ties it with a thin string."""
    f, s, l = ramp(role, level)
    w = 0.10 + len(text) * size * 0.50 / 72
    pts = np.array([(0, h / 2), (0.32 * h, h), (w, h), (w, 0), (0.32 * h, 0)]) + [x, y]
    p = Polygon(pts, closed=True, transform=fig.dpi_scale_trans, fc=f, ec=l, lw=0.5, zorder=z, joinstyle='round')
    fig.add_artist(drop(p, 1.0, -1.0, s))
    fig.add_artist(Polygon(np.array([(0.32 * h, 0), (w, 0), (w, h * 0.30), (0.32 * h, h * 0.30), (0.12 * h, h * 0.22)])
                           + [x, y], closed=True, transform=fig.dpi_scale_trans, fc=s, ec='none', zorder=z + 0.01))
    ex, ey = x + 0.30 * h, y + h / 2
    eyelet(fig, ex, ey, z=z + 0.05)
    fw, fh = fig.get_size_inches()
    fig.text((x + 0.52 * h + (w - 0.52 * h) / 2) / fw, (y + h / 2 - 0.003) / fh, text, fontsize=size, **BOLD,
             color=deeper(l, 0.16), ha='center', va='center', zorder=z + 0.1)
    if string_to is not None:
        sx, sy = string_to
        mx, my = (ex + sx) / 2, max(ey, sy) + 0.04
        t = np.linspace(0, 1, 20)
        bx = (1 - t) ** 2 * ex + 2 * (1 - t) * t * mx + t ** 2 * sx
        by = (1 - t) ** 2 * ey + 2 * (1 - t) * t * my + t ** 2 * sy
        fig.add_artist(Line2D(bx, by, transform=fig.dpi_scale_trans, color=deeper(LINE, 0.06), lw=0.4, zorder=z + 0.06))
    return x + w


def ring(ax, x, y, d=8.5, role='gray', z=6.6, transform=None):
    """The specimen ring: a hollow ring around the data point a specimen card is drawn for."""
    c = deeper(line(role), 0.10)
    col = PathCollection([DISC], sizes=[(d / 2) ** 2], offsets=[(x, y)], offset_transform=transform or ax.transData,
                         facecolors='none', edgecolors=[c], linewidths=0.6, zorder=z)
    col.set_transform(IdentityTransform())
    ax.add_collection(col, autolim=False)


def leader(ax, x, y, fig_xy, role='gray', z=6.5, d=8.5):
    """A hairline from a specimen card (fig inches) to its data point, ringed at the point."""
    from matplotlib.patches import ConnectionPatch
    c = deeper(line(role), 0.10)
    fig = ax.figure
    # drawn with the panel's own artists, so labels on paper above it in z cover it
    con = ConnectionPatch(xyA=fig_xy, coordsA=fig.dpi_scale_trans, xyB=(x, y), coordsB=ax.transData, color=c,
                          lw=0.45, shrinkB=d / 2, zorder=z)
    ax.add_artist(con)
    con.set_clip_on(False)
    ring(ax, x, y, d=d, role=role, z=z + 0.1)


def depth_key(fig, x, y, role, lo, hi, w=0.42, h=0.06, size=6.8, z=8.5, label=None):
    """The depth scale of a figure: five cel cells of one family, pale to deep, end values beside it."""
    tr = fig.dpi_scale_trans
    cs = shades(role)
    cw = w / len(cs)
    fig.add_artist(drop(Rectangle((x, y), w, h, transform=tr, fc=PAPER, ec='none', zorder=z - 0.01), 1.0, -1.0))
    for k, c in enumerate(cs):
        fig.add_artist(Rectangle((x + k * cw, y), cw, h, transform=tr, fc=c, ec='none', zorder=z))
    fig.add_artist(Rectangle((x, y), w, h, transform=tr, fc='none', ec=LINE, lw=0.45, zorder=z + 0.01))
    for k in range(1, len(cs)):
        fig.add_artist(Line2D([x + k * cw] * 2, [y, y + h], transform=tr, color=PAPER, lw=0.4, zorder=z + 0.02))
    fw, fh = fig.get_size_inches()
    fig.text((x - 0.025) / fw, (y + h / 2) / fh, lo, fontsize=size, color=MUTED, ha='right', va='center', zorder=z)
    fig.text((x + w + 0.025) / fw, (y + h / 2) / fh, hi, fontsize=size, color=MUTED, ha='left', va='center', zorder=z)
    if label:
        fig.text((x + w / 2) / fw, (y + h + 0.02) / fh, label, fontsize=size, color=MUTED, ha='center', va='bottom',
                 zorder=z)
