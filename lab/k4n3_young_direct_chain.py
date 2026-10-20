"""Age-1 young (2,1,1) feedback computed from graph-term factors.

The slice group born at layer L carries T = K(2,1,1) (plus the (2,2) birth
slice).  Read at layer L+1 with transport Q, its four dslices need only

    F1  = ((Q∘Q) ⊗ Q ⊗ Q) · sym_jk(T)      (n³, the (2,1,1) dslice body)
    A21 = F1[i, i, j]                       (n×n)
    dB  = 6 Σ_ab Q_ia Q_ib sym_jk(T)[r,a,b] (n×n)

    (4,) = rowsum(Q∘Q ∘ dB)       (3,1) = 3 A21 + ½ (Q∘dB) Qᵀ
    (2,2) = ((Q∘Q) dBᵀ + dB (Q∘Q)ᵀ)/6
    (2,1,1) = F1 + Σ_r dB[i,r] Q[j,r] Q[k,r] / 6      (all zero-repeated)

Tree graph terms (two matrix edges at a centre vertex) are merged by a shared
edge and give A21/dB with a fixed number of n×n products.  Terms that are
linear maps of the full κ3 tensor or of the previous K4 (2,1,1) slice, and the
covariance triangle, are evaluated densely here (validation only) and tagged.
The chain mirrors ``k4n3_tucker_age.run_chain`` with a dtype argument.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

assert os.environ.get("AW_SPEC") == "star,three2,slice4+slice31,!slice22+slice211"
import k4full
from k4n3_ablate5 import graph_class
from k4n3_tucker_age import mode_product, reconstruct, shared_basis
from mlp_kprop.diagslice import _zero_repeated, diagslice, zero_repeated
from mlp_kprop.partitions import vec_part_coef
from mlp_kprop.tensor_utils import symmetrize

NS = k4full.nonlin.__globals__
TREE = ("tree2_c", "tree2_k3", "tree2_k4", "parallel3_c")
P211 = (2, 1, 1)
TRIANGLE = ((0, 1, 1), (1, 0, 1), (1, 1, 0))


@dataclass
class Edge:
    """Matrix E[x_centre, x_leaf] with a hashable description of its source."""
    mat: torch.Tensor
    key: tuple


@dataclass(eq=False)
class Tree:
    """coef · wc[c] · E0[c, x0] w0[x0] · E1[c, x1] w1[x1] in birth order (r, a, b).

    ``centre`` is 0 (doubled axis) or 1 (one of the symmetric pair).  For
    centre 0 the leaves are the pair; for centre 1 the leaves are the doubled
    axis (edge ``e0``) and the other pair axis (edge ``e1``).
    """
    cls: str
    centre: int
    coef: float
    wc: torch.Tensor
    e0: Edge
    w0: torch.Tensor
    w0key: tuple
    e1: Edge
    w1: torch.Tensor
    w1key: tuple

    def side(self, which: int) -> torch.Tensor:
        e, w = (self.e0, self.w0) if which == 0 else (self.e1, self.w1)
        return e.mat * w[None, :]

    def side_key(self, which: int) -> tuple:
        return (self.e0.key, self.w0key) if which == 0 else (self.e1.key, self.w1key)

    def dense(self) -> torch.Tensor:
        a = self.coef * self.wc[:, None] * self.side(0)
        b = self.side(1)
        if self.centre == 0:
            t = a[:, :, None] * b[:, None, :]
        else:  # centre on axis 1: T[r, c, b] = a[c, r] b[c, b]
            t = a.T[:, :, None] * b[None, :, :]
        return zero_repeated(t)


@dataclass
class Birth:
    """All additive pieces of one newborn (2,1,1) slice."""
    trees: list[Tree] = field(default_factory=list)
    raw111: list = field(default_factory=list)
    dense: dict[str, torch.Tensor] = field(default_factory=dict)
    d22: torch.Tensor | None = None
    total: torch.Tensor | None = None


def raw_tree(vec_part, count, K, wick_lookup, power):
    """Centre axis, coefficient, centre Wick and [(leaf axis, Edge[centre, leaf], Wick, key)]."""
    edges = {}
    for v in vec_part:
        axes = tuple(i for i in range(3) if v[i])
        assert len(axes) == 2, vec_part
        mat = diagslice(K[sum(v)], tuple(v[i] for i in axes), output_zero_repeated=True)
        desc = (sum(v), v[axes[0]], v[axes[1]])
        if axes in edges:
            old_mat, old_desc = edges[axes]
            edges[axes] = (old_mat * mat, tuple(sorted(old_desc + (desc,))))
        else:
            edges[axes] = (mat, (desc,))
    assert len(edges) == 2, vec_part
    pairs = list(edges)
    centre = (set(pairs[0]) & set(pairs[1])).pop()
    total = tuple(sum(v[i] for v in vec_part) for i in range(3))
    wick = [wick_lookup(int(total[i]), int(power[i])) for i in range(3)]
    leaves = []
    for pair in pairs:
        mat, desc = edges[pair]
        leaf = pair[0] if pair[1] == centre else pair[1]
        if pair[0] != centre:
            mat = mat.T
            desc = tuple((d, b, a) for d, a, b in desc)
        leaves.append((leaf, Edge(mat, desc), wick[leaf], (int(total[leaf]), int(power[leaf]))))
    leaves.sort(key=lambda x: x[0])
    coef = float(count * vec_part_coef(vec_part, divide_fac=True))
    return centre, coef, wick[centre], leaves


def tree_from_term(vec_part, count, K, wick_lookup, cls) -> Tree:
    centre, coef, wc, ((la, ea, wa, ka), (lb, eb, wb, kb)) = raw_tree(
        vec_part, count, K, wick_lookup, P211)
    if centre == 0:      # leaves 1, 2 (sorted)
        return Tree(cls, 0, coef, wc, ea, wa, ka, eb, wb, kb)
    assert la == 0       # centre on a pair axis: e0 to the doubled axis
    return Tree(cls, 1, coef, wc, ea, wa, ka, eb, wb, kb)


def trees_from_p111(raw, p1: torch.Tensor, cls: str) -> list[Tree]:
    """−2 p1[i] · sym6(T111) as three (2,1,1) trees, each with a third of the weight."""
    centre, coef, wc, ((_, ex, wx, kx), (_, ey, wy, ky)) = raw
    s = -2 * p1
    c3 = coef / 3
    tag = lambda k: k + ("p1",)
    return [Tree(cls + "@111", 0, c3, wc * s, ex, wx, kx, ey, wy, ky),
            Tree(cls + "@111", 1, c3, wc, ex, wx * s, tag(kx), ey, wy, ky),
            Tree(cls + "@111", 1, c3, wc, ey, wy * s, tag(ky), ex, wx, kx)]


class Products:
    """Counts n×n×n (and n×n×k) products by name."""

    def __init__(self):
        self.count = Counter()

    def mm(self, name, a, b):
        self.count[name] += a.shape[0] * a.shape[1] * b.shape[1] / a.shape[0] ** 3 if a.shape[0] else 0
        return a @ b


def merge_groups(trees: list[Tree], centre: int) -> list[tuple[int, list[Tree]]]:
    """Greedy: repeatedly take the side key shared by most remaining trees."""
    left = [t for t in trees if t.centre == centre]
    groups = []
    while left:
        votes = Counter()
        for t in left:
            for side in (0, 1):
                votes[(side if centre == 1 else 0, t.side_key(side))] += 1
        (side, key), _ = votes.most_common(1)[0]
        if centre == 0:
            members = [t for t in left if key in (t.side_key(0), t.side_key(1))]
        else:
            members = [t for t in left if t.side_key(side) == key]
        groups.append((side, key, members))
        left = [t for t in left if t not in members]
    return groups


def tree_parts(q: torch.Tensor, trees: list[Tree], d22: torch.Tensor | None,
               prod: Products, f1: torch.Tensor | None):
    """A21 and dB of all tree terms (+ slice22) with merged products."""
    n = q.shape[0]
    qq = q * q
    a21 = torch.zeros((n, n), dtype=q.dtype, device=q.device)
    db = torch.zeros_like(a21)
    lq = torch.zeros_like(a21)   # left factors of every product ending in @ q.T (one product)
    zg = torch.zeros_like(a21)   # Dg: δ_ab Z[r, a]
    z0 = torch.zeros_like(a21)   # D0: δ_ra Z[r, b] (sym)
    if d22 is not None:
        zg += 0.5 * d22
    for t in trees:   # leaf collision corrections, elementwise per term
        a = t.coef * t.wc[:, None] * t.side(0)
        b = t.side(1)
        if t.centre == 0:
            zg -= a * b
        else:
            z0 -= (a * b).T
    for _, key, members in merge_groups(trees, 0):
        shared = None
        msum = torch.zeros_like(a21)
        for t in members:
            mine = 1 if t.side_key(0) == key else 0     # side kept in the sum
            if shared is None:
                shared = t.side(1 - mine)
            msum += t.coef * t.wc[:, None] * t.side(mine)
        A = prod.mm("tree_P0", q, msum.T)
        B = prod.mm("tree_P0", q, shared.T)
        a21 += 0.5 * (prod.mm("tree_P0", qq * A, B.T) + prod.mm("tree_P0", qq * B, A.T))
        db += 6 * A * B
        if f1 is not None:
            f1 += 0.5 * (torch.einsum("ir,jr,kr->ijk", qq, A, B)
                         + torch.einsum("ir,jr,kr->ijk", qq, B, A))
    for side, key, members in merge_groups(trees, 1):
        msum = torch.zeros_like(a21)
        nsum = torch.zeros_like(a21)
        for t in members:
            a = t.coef * t.wc[:, None] * t.side(0)
            b = t.side(1)
            if side == 0:   # shared doubled-axis edge, sum the pair edges
                msum = t.side(0)
                nsum += t.coef * t.wc[:, None] * b
            else:
                nsum = b
                msum += a
        G = prod.mm("tree_P1", qq, msum.T)
        H = prod.mm("tree_P1", q, nsum.T)
        a21 += 0.5 * prod.mm("tree_P1", G * q, H.T)
        lq += 0.5 * (G * H)
        db += 6 * prod.mm("tree_P1", q * H, msum)
        if f1 is not None:
            f1 += 0.5 * (torch.einsum("ia,ja,ka->ijk", G, q, H)
                         + torch.einsum("ia,ja,ka->ijk", G, H, q))
    G = prod.mm("tree_D", qq, zg)
    lq += G * q
    db += 6 * prod.mm("tree_D", qq, zg.T)
    if f1 is not None:
        f1 += torch.einsum("ia,ja,ka->ijk", G, q, q)
    P = prod.mm("tree_D", q, z0.T)
    a21 += 0.5 * prod.mm("tree_D", qq * q, P.T)
    lq += 0.5 * (qq * P)
    db += 6 * q * P
    if f1 is not None:
        f1 += 0.5 * (torch.einsum("ir,jr,kr->ijk", qq, q, P)
                     + torch.einsum("ir,jr,kr->ijk", qq, P, q))
    return a21, db, lq


def dense_parts(q: torch.Tensor, t: torch.Tensor, f1: torch.Tensor | None):
    qq = q * q
    ts = 0.5 * (t + t.transpose(1, 2))
    half = torch.einsum("ja,kb,rab->rjk", q, q, ts)
    F = torch.einsum("ir,rjk->ijk", qq, half)
    if f1 is not None:
        f1 += F
    a21 = torch.diagonal(F, dim1=0, dim2=1).T
    db = 6 * torch.diagonal(half, dim1=1, dim2=2).T
    return a21.contiguous(), db.contiguous()


def slices_from(q, a21, db, f1, lq=None, prod=None):
    """a21 here excludes lq @ q.T; the (3,1) right-q products share one product."""
    qq = q * q
    left = 0.5 * (q * db) + (3 * lq if lq is not None else 0)
    mm = prod.mm if prod is not None else (lambda name, a, b: a @ b)
    d22 = mm("slices", qq, db.T)
    out = {(4,): (qq * db).sum(1),
           (3, 1): 3 * a21 + mm("slices", left, q.T),
           (2, 2): (d22 + d22.T) / 6}
    if f1 is not None:
        out[P211] = f1 + torch.einsum("ir,jr,kr->ijk", db, q, q) / 6
    return {p: zero_repeated(v) for p, v in out.items()}


def tree_tensor(trees: list[Tree], n, device, dtype) -> torch.Tensor:
    """Dense sym/zero-repeated sum of tree terms with batched outer products (no n³ temporaries)."""
    acc = torch.zeros((n, n, n), device=device, dtype=dtype)
    for t in trees:
        a = (t.coef * t.wc[:, None] * t.side(0)).to(dtype)
        b = t.side(1).to(dtype)
        view = acc if t.centre == 0 else acc.permute(1, 0, 2)
        view.baddbmm_(a[:, :, None], b[:, None, :])
    acc.add_(acc.transpose(1, 2).clone()).mul_(0.5)
    _zero_repeated(acc)
    return acc


class Recorder:
    """Hooks into the reference nonlinear step and splits the (2,1,1) birth."""

    def __init__(self, light=False, birth_mode=None, drop_triangle=False, basis_fn=None):
        self.light, self.birth_mode, self.drop_triangle = light, birth_mode, drop_triangle
        self.basis_fn = basis_fn
        self.with111 = True
        self.split = {}
        self.current: Birth | None = None
        self.K = None
        self.terms = defaultdict(lambda: 0.)
        self.sub_new = None
        self.conv_in = None
        self.conv_out = None
        self.conv_slices = None
        self.births: dict[int, Birth] = {}
        self.check = {}
        self.norms = {}

    def install(self):
        self.saved = {k: NS[k] for k in ("eval_part", "multiply_wicks", "DS_pK_to_K")}
        self.saved_add = k4full.SK4.add_groups_
        self.saved_from = k4full.SK4.from_dstensor
        rec = self

        def eval_part(K, *args, **kwargs):
            rec.K = K
            if rec.drop_triangle and args[0] == TRIANGLE and \
                    inspect.currentframe().f_back.f_locals.get("int_part") == P211:
                return None
            return rec.saved["eval_part"](K, *args, **kwargs)

        def multiply_wicks(K_part, k, p, wick_lookup):
            out = rec.saved["multiply_wicks"](K_part, k, p, wick_lookup)
            if tuple(p) == (1, 1, 1):
                loc = inspect.currentframe().f_back.f_locals
                cls = graph_class(loc["vec_part"])
                if cls in TREE:
                    rec.current.raw111.append((cls, raw_tree(
                        loc["vec_part"], loc["count"], rec.K, wick_lookup, (1, 1, 1))))
                if not rec.light:
                    rec.terms111[cls] = rec.terms111[cls] + loc["count"] * out
            if tuple(p) == P211:
                loc = inspect.currentframe().f_back.f_locals
                vec_part, count = loc["vec_part"], loc["count"]
                cls = graph_class(vec_part)
                if cls in TREE:
                    rec.current.trees.append(
                        tree_from_term(vec_part, count, rec.K, wick_lookup, cls))
                if not rec.light:
                    rec.terms[cls] = rec.terms[cls] + count * out
            return out

        def ds_pk_to_k(pK_ds, *args, **kwargs):
            out = rec.saved["DS_pK_to_K"](pK_ds, *args, **kwargs)
            rec.p11 = pK_ds[2].get_slice((1, 1)).clone()
            rec.p1 = pK_ds[1].get_slice((1,)).clone()
            if rec.light:
                return out
            rec.conv_in = pK_ds[4].get_slice(P211).clone()
            rec.conv_slices = (pK_ds[1].get_slice((1,)).clone(),
                               pK_ds[2].get_slice((1, 1)).clone(),
                               pK_ds[3].get_slice((1, 1, 1)).clone())
            rec.conv_out = out[4].get_slice(P211).clone()
            return out

        def add_groups_(sk4, groups):
            if sk4.repeated.slices:
                new = sk4._dslices(groups)
                if not rec.light:
                    rec.sub_new = new[P211].clone()
                for part in sk4.repeated.slices:
                    sk4.repeated.slices[part] = sk4.repeated.slices[part] + new[part]
            sk4.groups.extend(groups)

        def from_dstensor(ds):
            rec.add_conv_path()
            if rec.birth_mode:
                rec.split_birth(ds)
            out = rec.saved_from(ds)
            if not rec.light:
                rec.finish(ds, out)
            return out

        NS["eval_part"] = eval_part
        NS["multiply_wicks"] = multiply_wicks
        NS["DS_pK_to_K"] = ds_pk_to_k
        k4full.SK4.add_groups_ = add_groups_
        k4full.SK4.from_dstensor = staticmethod(from_dstensor)

    def uninstall(self):
        NS.update(self.saved)
        k4full.SK4.add_groups_ = self.saved_add
        k4full.SK4.from_dstensor = staticmethod(self.saved_from)

    def start_layer(self):
        self.current = Birth()
        self.terms = defaultdict(lambda: 0.)
        self.terms111 = defaultdict(lambda: 0.)
        self.sub_new = None

    def add_conv_path(self):
        """pK->K: −2 p11[i,j] p11[i,k] (centre-0 tree) and −2 p1[i] p111 tree terms."""
        one = torch.ones(self.p11.shape[0], device=self.p11.device, dtype=self.p11.dtype)
        e = Edge(self.p11, (("p11",),))
        self.current.trees.append(Tree("conv_path", 0, -2.0, one, e, one, ("1",), e, one, ("1",)))
        if self.with111:
            for cls, raw in self.current.raw111:
                self.current.trees.extend(trees_from_p111(raw, self.p1, cls))

    def split_birth(self, ds):
        """Keep the tree part of the newborn (2,1,1) exact; project the rest on a k3 basis."""
        total = ds.slices[P211]
        n = total.shape[0]
        tree = tree_tensor(self.current.trees, n, total.device, total.dtype)
        target = total if self.birth_mode == "all_layer" else total - tree
        basis = self.basis_fn(target, fresh=self.birth_mode == "inh_fresh")
        core = target
        for axis in range(3):
            core = mode_product(core, basis, axis)
        for axis in range(3):
            core = reconstruct(core, basis, axis)
        res = target - core
        del core
        self.split = dict(total=float(total.norm()), tree=float(tree.norm()),
                          projected_part=float(target.norm()), residual=float(res.norm()))
        new = total - res
        del res, tree, target
        _zero_repeated(new)
        ds.slices[P211] = new

    def finish(self, ds, sk4):
        b = self.current
        b.total = ds.get_slice(P211).clone()
        b.d22 = ds.get_slice((2, 2)).clone()
        pk = {cls: zero_repeated(symmetrize(t, vec=P211)) for cls, t in self.terms.items()
              if torch.is_tensor(t)}
        p1, p11, p111 = self.conv_slices
        conv = zero_repeated(self.conv_out - self.conv_in)
        path = zero_repeated(-2 * p11[:, :, None] * p11[:, None, :])
        conv_p111 = zero_repeated(-2 * p1[:, None, None] * p111)
        sub = zero_repeated(self.conv_out - b.total)
        sub_new = zero_repeated(self.sub_new) if self.sub_new is not None else torch.zeros_like(sub)
        for cls, t in pk.items():
            if cls not in TREE:
                b.dense[cls] = t
        tree111 = sum((zero_repeated(-2 * p1[:, None, None] * symmetrize(t, vec=(1, 1, 1)))
                       for c, t in self.terms111.items() if c in TREE and torch.is_tensor(t)),
                      torch.zeros_like(conv_p111))
        b.dense["conv_p111"] = conv_p111 - tree111
        b.dense["sub_new"] = -sub_new
        b.dense["sub_old"] = -(sub - sub_new)
        tree_sum = sum(zero_repeated(symmetrize(t.dense(), vec=P211)) for t in b.trees)
        tree_ref = sum(pk.get(c, 0.) for c in TREE) + path + tree111
        rebuilt = tree_sum + sum(b.dense.values())
        scale = float(b.total.abs().max().clamp_min(1e-300))
        self.check = dict(
            tree_factor=float((tree_sum - tree_ref).abs().max()) / scale,
            conv_split=float((conv - path - conv_p111).abs().max()) / scale,
            rebuilt=float((rebuilt - b.total).abs().max()) / scale,
            tree_batched=float((tree_tensor(b.trees, b.total.shape[0], b.total.device,
                                            b.total.dtype) - tree_sum).abs().max()) / scale)
        self.norms = {c: float(t.norm()) for c, t in b.dense.items()}
        self.norms["tree"] = float(tree_sum.norm())
        self.norms["total"] = float(b.total.norm())
        slice_group = [g for g in sk4.groups if "slice211" in g.types]
        assert len(slice_group) == 1
        self.births[id(slice_group[0].Bb)] = b


def young_slices(q, birth: Birth, prod: Products, want_f1: bool, dense_classes=None):
    """Direct four dslices of one age-1 slice group; returns (slices, per-class |A21|,|dB|)."""
    n = q.shape[0]
    f1 = torch.zeros((n, n, n), dtype=q.dtype, device=q.device) if want_f1 else None
    a21, db, lq = tree_parts(q, birth.trees, birth.d22, prod, f1)
    parts = {"tree": (float((a21 + lq @ q.T).norm()), float(db.norm()))}
    for cls, t in birth.dense.items():
        if dense_classes is not None and cls not in dense_classes:
            continue
        x, y = dense_parts(q, t, f1)
        parts[cls] = (float(x.norm()), float(y.norm()))
        a21 = a21 + x
        db = db + y
    return slices_from(q, a21, db, f1, lq, prod), parts


def run_chain(weights, device, dtype, *, m0f, rank=128, m0=2, old_rank=128,
              layers=0, young="reference", log=None, light=False, birth_mode=None,
              drop_triangle=False, old_low="exact"):
    """``young``: reference (dense group) or direct (factor formulas, checked).

    ``birth_mode`` rewrites each newborn (2,1,1) slice before it becomes a group:
    inh_layer/inh_fresh keep the tree terms exact and project the remainder on
    the layer's old-feedback basis / a basis refit to the remainder; all_layer
    projects the whole slice on the layer basis.
    """
    n, length = weights[0].shape[0], layers or len(weights)
    mlp = k4full.MLP(hidden_dim=n, num_layers=len(weights) + 1, nonlin="relu",
                     init_kind="manual", w_var=2.0).to(device=device, dtype=dtype)
    for i, weight in enumerate(weights):
        mlp.Ws[i].weight.data = torch.from_numpy(weight.T.copy()).to(device=device, dtype=dtype)
    cumulants = k4full.coerce_input({1: torch.zeros(n, device=device, dtype=dtype),
                                     2: torch.eye(n, device=device, dtype=dtype)},
                                    k_max=4, kind=k4full.Kind.SIMPLE)
    wick = k4full.WICK_COEF_D["relu"]
    original = k4full.SK4._dslices
    def basis_fn(target, fresh):
        if projected_layer == layer and projected_q is not None and not fresh:
            return projected_q
        prior = projected_q if projected_layer == layer else transported_q
        return shared_basis(target, prior, rank)

    rec = Recorder(light=light, birth_mode=birth_mode, drop_triangle=drop_triangle,
                   basis_fn=basis_fn)
    birth: dict[int, int] = {}
    means, rel, records = [], [], []
    prior_q = transported_q = projected_q = None
    projected_layer = -1
    layer = -1

    def aged(self, groups):
        nonlocal prior_q, projected_layer, projected_q
        slice_groups = [g for g in groups if "slice211" in g.types]
        if not slice_groups:
            return original(self, groups)
        age = lambda g: layer - birth.get(id(g.Bb), layer)
        old = [g for g in slice_groups if age(g) > m0f] if rank < n else []
        direct = [g for g in slice_groups if g not in old and young == "direct"
                  and id(g.Bb) in rec.births and age(g) == 1]
        rest = [g for g in groups if g not in old and g not in direct]
        out = original(self, rest) if rest else {
            p: torch.zeros((n,) * len(p), device=device, dtype=dtype) for p in k4full.PARTS}
        for g in direct:
            prod = Products()
            got, parts = young_slices(g.Q, rec.births[id(g.Bb)], prod, True)
            ref = original(self, [g])
            err = {str(p): float((got[p] - ref[p]).abs().max() /
                                 ref[p].abs().max().clamp_min(1e-300)) for p in ref}
            records[-1].update(direct_rel_err=err, products=dict(prod.count), parts=parts)
            for p in out:
                out[p] = out[p] + got[p]
        if old:
            aged_out = original(self, old)
            s211 = aged_out[P211]
            if projected_layer != layer:
                projected_q = shared_basis(s211, transported_q, rank)
                projected_layer = layer
            core = s211
            for axis in range(3):
                core = mode_product(core, projected_q, axis)
            proj = core
            for axis in range(3):
                proj = reconstruct(proj, projected_q, axis)
            proj = k4full.zero_repeated(proj)
            rel[layer] = float((proj - s211).norm() / s211.norm().clamp_min(1e-30))
            aged_out[P211] = proj
            if old_low == "drop":   # probe: old slice groups lose (4,), (3,1), (2,2)
                for part in ((4,), (3, 1), (2, 2)):
                    aged_out[part] = torch.zeros_like(aged_out[part])
            prior_q = projected_q
            for p in out:
                out[p] = out[p] + aged_out[p]
        return out

    k4full.SK4._dslices = aged
    rec.install()
    try:
        for layer in range(length):
            k4full.LAST_LAYER = layer == length - 1
            rel.append(0.0)
            records.append(dict(layer=layer))
            rec.start_layer()
            weight = mlp.Ws[layer].weight
            transported_q = weight @ prior_q if prior_q is not None else None
            cumulants = k4full.linear_kprop(cumulants, weight, k_max=4,
                                            set_metric=mlp.init_scale[layer])
            if transported_q is not None:
                w1 = wick(mean=cumulants[1].core, var=cumulants[2].core.diag(), k=1, p=1)
                transported_q = w1[:, None] * transported_q
            cumulants = k4full.nonlin(cumulants, nonlin_wick_coef=wick)
            means.append(cumulants[1].to_tensor().double().cpu().numpy())
            records[-1].update(birth_check=rec.check, norms=rec.norms, split=rec.split,
                               projection_rel=rel[layer])
            groups = cumulants[4].groups
            for group in groups:
                birth.setdefault(id(group.Bb), layer)
            live = {id(g.Bb) for g in groups}
            for key in [k for k in rec.births if k not in live or layer - birth[k] > 1]:
                del rec.births[key]
            old = [g for g in groups if g.types[0] in ("star", "three2")
                   and layer - birth[id(g.Bb)] > m0]
            if old and old_rank < n:
                unique = {birth[id(g.Bb)]: g.Q for g in old}
                sketch = weight[:, :min(old_rank, n)].contiguous()
                for _ in range(2):
                    image = sum((q @ (q.T @ sketch) for q in unique.values()),
                                torch.zeros_like(sketch))
                    sketch, _ = torch.linalg.qr(image, mode="reduced")
                for g in old:
                    g.Q = sketch @ (sketch.T @ g.Q)
            if log:
                print(json.dumps(records[-1]), file=log, flush=True)
    finally:
        k4full.SK4._dslices = original
        rec.uninstall()
    return np.stack(means), rel, records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--m0f", type=int, default=1)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--dtype", default="f64")
    p.add_argument("--official-index", type=int, default=-1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--birth-mode", default=None,
                   choices=(None, "inh_layer", "inh_fresh", "all_layer"))
    p.add_argument("--drop-triangle", action="store_true")
    p.add_argument("--reuse-ref", default=None, help="reference prediction .npy from an earlier run")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    dtype = torch.float64 if args.dtype == "f64" else torch.float32
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.official_index >= 0:
        official(args, dtype, out)
        return
    rng = np.random.default_rng(args.seed)
    n = args.width
    weights = [(rng.standard_normal((n, n)) * (2 / n) ** .5).astype("f")
               for _ in range(args.layers)]
    tag = f"_{args.birth_mode}" if args.birth_mode else ""
    tag += "_notri" if args.drop_triangle else ""
    stem = f"direct_m0f{args.m0f}_k{args.rank}_n{n}_seed{args.seed}_{args.dtype}{tag}"
    t0 = time.monotonic()
    if args.reuse_ref:
        ref = np.load(args.reuse_ref)
    else:
        ref, _, _ = run_chain(weights, args.device, dtype, m0f=args.m0f, rank=args.rank,
                              layers=args.layers, young="reference")
    t1 = time.monotonic()
    with open(out / f"{stem}.log", "w") as log:
        got, got_rel, records = run_chain(
            weights, args.device, dtype, m0f=args.m0f, rank=args.rank, layers=args.layers,
            young="direct" if not args.birth_mode else "reference", log=log,
            birth_mode=args.birth_mode, drop_triangle=args.drop_triangle,
            light=bool(args.birth_mode))
    diff = np.max(np.abs(got - ref), axis=1)
    scale = np.max(np.abs(ref), axis=1)
    receipt = dict(width=n, layers=args.layers, seed=args.seed, m0f=args.m0f, rank=args.rank,
                   dtype=args.dtype, birth_mode=args.birth_mode,
                   drop_triangle=args.drop_triangle, per_layer_max_abs=diff.tolist(),
                   per_layer_rel=(diff / scale).tolist(), max_abs=float(diff.max()),
                   final_rms_vs_reference=float(np.sqrt(np.mean((got[-1] - ref[-1]) ** 2))),
                   reference_s=t1 - t0, direct_s=time.monotonic() - t1,
                   memory=k4full._mem())
    checks = [r["birth_check"] for r in records if r.get("birth_check")]
    if checks:
        receipt["birth_check_max"] = {k: max(c[k] for c in checks) for k in checks[0]}
    errs = [max(r["direct_rel_err"].values()) for r in records if "direct_rel_err" in r]
    receipt["direct_err_max"] = max(errs) if errs else None
    np.save(out / f"{stem}_ref.npy", ref)
    np.save(out / f"{stem}.npy", got)
    (out / f"{stem}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def official(args, dtype, out):
    from mlpdata import load_truth, weights_from_seed
    truth = load_truth("mini")
    weights = weights_from_seed(truth["mlp_seed"][args.official_index])
    target = truth["all_layer_means"][args.official_index].astype(np.float64)
    tag = f"_{args.birth_mode}" if args.birth_mode else ""
    tag += "_notri" if args.drop_triangle else ""
    stem = f"birth_m0f{args.m0f}_k{args.rank}{tag}_mlp{args.official_index}"
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    with open(out / f"{stem}.log", "w") as log:
        pred, rel, records = run_chain(
            weights, args.device, dtype, m0f=args.m0f, rank=args.rank, layers=args.layers,
            log=log, light=True, birth_mode=args.birth_mode, drop_triangle=args.drop_triangle)
    errors = np.mean((pred - target[:len(pred)]) ** 2, axis=1)
    receipt = dict(m0f=args.m0f, rank=args.rank, birth_mode=args.birth_mode,
                   drop_triangle=args.drop_triangle, index=args.official_index,
                   width=weights[0].shape[0], layers=len(pred), final_mse=float(errors[-1]),
                   per_layer_mse=errors.tolist(), projection_rel=rel,
                   split=[r.get("split") for r in records],
                   elapsed_s=time.monotonic() - t0, memory=k4full._mem(),
                   spec=os.environ["AW_SPEC"])
    np.save(out / f"{stem}.npy", pred)
    (out / f"{stem}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
