"""Counted units of a reduce_s1 plan, row for row the Table 7a ledger (report §5.3).

Convention of Table 7a: births at layers s0..13, each read at every later layer
through 14 (a "pair" is one (birth, read)); reads at layer 15 and births at 14
are not counted. A plan keeps a subset of states, each for a prefix of its reads,
and pays only the rows its kept pairs and births use:

  chain B (metered)                168, every column
  C.k3 birth, per pair             7 (9n b - n)/n + 4*16 = 63 b + 57   (real kappa3 width + rank-16 crossing)
  non-tree birth, per birth        3 (2 n^4 - n^3) / 2^31 = 3070.5       (dense first read, all classes)
  tree slices, per pair            26                                    (26-matmul old-slice read)
  feedback, per tree pair          row total / pairs                     (old core + young tree)
  K=4 sources, per two-leg pair    row total / (2 pairs)                 (star and C.k3 machinery)
  young-leg transport              row total / (2 young births)          (two-leg births at b <= 12)

The three small rows are allocated in proportion; at a full plan every row
equals Table 7a's to the unit.
"""
from __future__ import annotations

N = 1024
NONTREE = 3 * (2 * N ** 4 - N ** 3) / 2 ** 31
B_UNITS = 168
# runs/cloud/k4s5/cost.json rows (start: sources, young-leg, feedback = old proxy + young tree + young projection)
SMALL = {0: (510.0494395219721, 234, 45.015 + 386 + 88.0),
         4: (310.8742464943789, 162, 33.011 + 286 + 64.53333333333333),
         6: (226.9904704503715, 126, 27.009 + 234 + 52.8),
         8: (153.49725360469893, 90, 21.007 + 182 + 41.06666666666666)}
LAST_READ = 14


def pairs(s0):
    return [(b, l) for b in range(s0, 14) for l in range(b + 1, LAST_READ + 1)]


def small_rows(s0):
    """Sources, young-leg and feedback totals for a full plan from s0 (per-pair rate of the nearest
    measured start at or below s0 for starts the micro-benchmark did not run)."""
    if s0 in SMALL:
        return SMALL[s0]
    base = max(k for k in SMALL if k <= s0)
    src, young, fb = SMALL[base]
    p0, p1 = len(pairs(base)), len(pairs(s0))
    young_births = lambda s: len([b for b in range(s, 13)])
    return (src * p1 / p0, 18 * young_births(s0), fb * p1 / p0)


def reads(life, b):
    """Counted reads of a state born at b with life A (A >= 14 - b: to the end)."""
    return max(0, min(life, LAST_READ - b))


def units(plan_life, s0):
    """plan_life: {family: {birth: life}} with families star, ck3, tree, nontree.
    The B seed travels with the tree slices born at s0 (one slice source per layer, as in Table 7a)."""
    src, young, fb = small_rows(s0)
    full_pairs = len(pairs(s0))
    young_births = len([b for b in range(s0, 13)])
    rows = dict.fromkeys(("chain B (metered)", "K=4 sources", "young-leg transport", "feedback",
                          "tree slice reads", "C.k3 births and crossing", "non-tree first reads"), 0.0)
    rows["chain B (metered)"] = B_UNITS
    for fam, births in plan_life.items():
        for b, life in births.items():
            b, life = int(b), int(life)
            if b < s0 or b > 13:
                continue
            r = reads(life, b)
            if fam == "ck3":
                rows["C.k3 births and crossing"] += r * (63 * b + 57)
            if fam in ("ck3", "star"):
                rows["K=4 sources"] += src * r / (2 * full_pairs)
                if b <= 12 and r:
                    rows["young-leg transport"] += young / (2 * young_births)
            if fam == "tree":
                rows["tree slice reads"] += 26 * r
                rows["feedback"] += fb * r / full_pairs
            if fam == "nontree" and r:
                rows["non-tree first reads"] += NONTREE
    return sum(rows.values()), rows


def full_life(s0, last_birth=13):
    return {fam: {b: 14 - b for b in range(s0, last_birth + 1)} for fam in ("star", "ck3", "tree", "nontree")}


if __name__ == "__main__":
    # Table 7a check: 60,549 / 47,287 / 33,611 (and 77,930 from layer 0, where no kappa3 source lives at b = 0)
    for s0, want in ((4, 60549), (6, 47287), (8, 33611)):
        total, rows = units(full_life(s0), s0)
        print(s0, round(total), want, {k: round(v) for k, v in rows.items()})
        assert abs(total - want) < 1.5, (s0, total)
    for s0 in (10, 12, 13):
        total, rows = units(full_life(s0), s0)
        print(s0, round(total), {k: round(v) for k, v in rows.items()})
