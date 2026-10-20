"""Mechanical legibility check of a drawn figure: every readable text must stand clear of every other text, of every
line, frame, bar, cell and mark, and stay inside its own panel sheet and the figure.

check(fig) draws the figure and returns a list of hits, each one line of plain text:
  text×text    two text boxes intersect (a text with a bbox, a chip, counts with its whole box)
  text×line    a text box is crossed by a drawn line or by the outline of a visible patch (frame, bar, cell, badge,
               bracket, arrow) or overlaps a drawn mark (disc, diamond, scatter marker, line marker)
  outside      a text whose anchor lies on a panel sheet runs past that sheet's inner frame, or any text runs past
               the figure edge
  small        a text set below MIN pt (exponents inside a label are part of their label and not counted)
Artists with gid 'texture' (focus-line bursts, dense data traces) are not obstacles: a label may sit over them. The
letter tab's own text (gid 'tab') sits on the frame corner by design and skips the line and frame tests. Patches with
neither a face nor an edge (screentone and hatch overlays) have no outline.
Text boxes are shrunk by TOL pt on each side before testing, so a hit is at least a visible overlap.
"""
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.path import Path
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Affine2D, Bbox

TOL = 0.3      # pt
MIN = 6.8      # pt, the smallest readable label at print size (figures go in at their own width)


def _none(c):
    if c is None:
        return True
    if isinstance(c, str):
        return c.lower() == 'none'
    c = np.asarray(c, dtype=float).ravel()
    return c.size == 0 or (c.size >= 4 and c[3] == 0)


def _visible(a):
    return a.get_visible() and (a.axes is None or a.axes.get_visible())


def _texts(fig):
    """Every readable text: figure and axes texts, annotations, axis labels and the tick labels actually drawn."""
    out, tick_texts = [], set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_texts.update({id(tick.label1), id(tick.label2)})
            if not ax.axison:
                continue
            for tick in axis._update_ticks():
                for lab in (tick.label1, tick.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        out.append((lab, f'tick {lab.get_text()}'))
    for t in fig.findobj(Text):
        if id(t) in tick_texts or not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        out.append((t, t.get_text()))
    seen, uniq = set(), []
    for t, s in out:
        if id(t) not in seen:
            seen.add(id(t))
            uniq.append((t, s))
    return uniq


def _box(t, r, shrink):
    patch = t.get_bbox_patch()
    if patch is not None:
        t.update_bbox_position_size(r)
    b = patch.get_window_extent(r) if patch is not None else t.get_window_extent(r)
    if patch is not None:            # a chip's hard shadow belongs to the chip
        for fx in patch.get_path_effects():
            off = getattr(fx, '_offset', None)
            if off is not None:
                dx, dy = (o * t.figure.dpi / 72 for o in off)
                b = Bbox.union([b, Bbox.from_extents(b.x0 + dx, b.y0 + dy, b.x1 + dx, b.y1 + dy)])
    return Bbox.from_extents(b.x0 + shrink, b.y0 + shrink, b.x1 - shrink, b.y1 - shrink)


def _hit(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _clip(artist, box):
    """The part of box the artist can paint into (its clip box), or None."""
    if artist.get_clip_on() and artist.get_clip_box() is not None:
        c = artist.get_clip_box()
        x0, y0, x1, y1 = max(box.x0, c.x0), max(box.y0, c.y0), min(box.x1, c.x1), min(box.y1, c.y1)
        if x0 >= x1 or y0 >= y1:
            return None
        return Bbox.from_extents(x0, y0, x1, y1)
    return box


def _name(a, g=None, dpi=72):
    gid = a.get_gid()
    kind = type(a).__name__
    where = 'fig' if a.axes is None else f'axes@{a.axes.get_position().x0:.2f},{a.axes.get_position().y0:.2f}'
    e = g if isinstance(g, Bbox) else (g.get_extents() if g is not None else None)
    at = '' if e is None else f' at {e.x0 / dpi:.2f}-{e.x1 / dpi:.2f} x {e.y0 / dpi:.2f}-{e.y1 / dpi:.2f} in'
    return f'{kind}{"[" + gid + "]" if gid else ""} ({where}){at}'


def _say(s):
    return s.replace('\n', ' / ')


def _shapes(fig, r):
    """(artist, kind, geometry): kind 'path' (display path, tested by its strokes), 'mark' (display path, tested
    filled) or 'box' (display bbox)."""
    dpi = fig.dpi
    out = []
    furniture = set()     # tick marks and grid lines belong to their tick labels
    for ax in fig.axes:
        furniture.add(id(ax.patch))
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                furniture.update({id(tick.tick1line), id(tick.tick2line), id(tick.gridline)})
    arrows = [t.arrow_patch for t in fig.findobj(Annotation) if getattr(t, 'arrow_patch', None) is not None]
    for a in fig.findobj(lambda x: isinstance(x, (Line2D, Patch, PathCollection))) + arrows:
        if not _visible(a) or a.get_gid() == 'texture' or id(a) in furniture:
            continue
        if isinstance(a, Line2D):
            xy = a.get_transform().transform(np.column_stack(a.get_data()).astype(float))
            ls = a.get_linestyle()
            if ls not in ('None', 'none', '', ' ') and a.get_linewidth() > 0:
                out.append((a, 'path', Path(xy)))
            m = a.get_marker()
            if m not in (None, 'None', 'none', '', ' '):
                ms, mew = a.get_markersize() * dpi / 72, max(a.get_markeredgewidth(), 0.3) * dpi / 72
                w, h = (ms, mew) if m == '_' else ((mew, ms) if m == '|' else (ms, ms))
                for x, y in xy:
                    out.append((a, 'box', Bbox.from_extents(x - w / 2, y - h / 2, x + w / 2, y + h / 2)))
        elif isinstance(a, FancyArrowPatch):
            try:
                paths, _ = a._get_path_in_displaycoord()
            except Exception:
                continue
            for p in paths:
                out.append((a, 'path', p))
        elif isinstance(a, Patch):
            if _none(a.get_facecolor()) and (_none(a.get_edgecolor()) or a.get_linewidth() == 0):
                continue
            p = a.get_transform().transform_path(a.get_path())
            out.append((a, 'path', p))
        elif isinstance(a, PathCollection):
            paths = a.get_paths()
            offs = a.get_offsets()
            if len(paths) == 0 or len(offs) == 0:
                continue
            trs = a.get_transforms()
            ot = a.get_offset_transform()
            base = a.get_transform()
            n = max(len(paths), len(offs))
            for i in range(n):
                p = paths[i % len(paths)]
                m = Affine2D(trs[i % len(trs)]) if len(trs) else Affine2D()
                fc = a.get_facecolor()
                ec = a.get_edgecolor()
                fci = fc[i % len(fc)] if len(fc) else None
                eci = ec[i % len(ec)] if len(ec) else None
                if _none(fci) and _none(eci):
                    continue
                ox, oy = ot.transform(np.asarray(offs[i % len(offs)], dtype=float))
                pp = (m + base + Affine2D().translate(ox, oy)).transform_path(p)
                out.append((a, 'mark', pp))
    return out


def check(fig, name=''):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    tol = TOL * dpi / 72
    texts = _texts(fig)
    boxes = [(t, _say(s), _box(t, r, tol)) for t, s in texts]
    hits = []
    for t, s, b in boxes:
        if t.get_fontsize() < MIN - 1e-6:
            hits.append(f'{name} small     "{s}"  at {t.get_fontsize():.1f} pt')
    # i: text x text
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _hit(boxes[i][2], boxes[j][2]):
                hits.append(f'{name} text×text  "{boxes[i][1]}"  ×  "{boxes[j][1]}"')
    # ii: text x line / outline / mark
    shapes = _shapes(fig, r)
    for t, s, b in boxes:
        if t.get_gid() == 'tab':
            continue
        own = t.get_bbox_patch()
        for a, kind, g in shapes:
            if a is own:
                continue
            bb = _clip(a, b)
            if bb is None:
                continue
            if kind == 'box':
                if _hit(bb, g):
                    hits.append(f'{name} text×line  "{s}"  ×  {_name(a, g, dpi)} marker')
                    break
                continue
            ext = g.get_extents()
            if not _hit(bb, ext):
                continue
            if g.intersects_bbox(bb, filled=(kind == 'mark')):
                hits.append(f'{name} text×line  "{s}"  ×  {_name(a, g, dpi)}')
    # iii: outside the panel sheet or the figure
    fb = fig.bbox
    sheets = [Bbox.from_bounds(*(np.array(sb) * dpi)) for sb in getattr(fig, '_cel_sheets', [])]
    inset = 1.7 * dpi / 72
    for t, s, b in boxes:
        if not (fb.x0 <= b.x0 and b.x1 <= fb.x1 and fb.y0 <= b.y0 and b.y1 <= fb.y1):
            hits.append(f'{name} outside   "{s}"  past the figure edge')
        if t.get_gid() == 'tab':
            continue
        ax_, ay_ = t.get_transform().transform(t.get_position())
        home = [sb for sb in sheets if sb.x0 <= ax_ <= sb.x1 and sb.y0 <= ay_ <= sb.y1]
        if not home:
            continue
        sb = min(home, key=lambda q: q.width * q.height)
        if not (sb.x0 + inset <= b.x0 and b.x1 <= sb.x1 - inset and sb.y0 + inset <= b.y0 and b.y1 <= sb.y1 - inset):
            hits.append(f'{name} outside   "{s}"  past its panel frame')
    return hits


def report(fig, name):
    hits = check(fig, name)
    print(f'figcheck {name}: {len(hits)} hit(s)')
    for h in hits:
        print('  ' + h)
    return hits
