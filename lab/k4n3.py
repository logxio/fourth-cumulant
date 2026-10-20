"""Matrix-core K4 source contractions for the portable research chain.

The source is Sym(sum_ab core[a,b] u_a v_a s_b t_b).  `same_slices`
contracts its repeated-index slices with matrix products.  The (2,1,1)
slice is returned as two lazy three-leg sources, so callers can feed a
factored K3 representation without allocating an n^3 array.  A dense
reference exists only in `check` for numerical comparison.
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass

import torch


@dataclass
class ThreeLeg:
    first: torch.Tensor
    second: torch.Tensor
    third: torch.Tensor
    weight: float

    def block(self, start: int, stop: int) -> torch.Tensor:
        # Validation/debug only. Production consumers keep these three legs.
        a = self.first[start:stop]
        return self.weight * (
            torch.einsum("ir,jr,kr->ijk", a, self.second, self.third)
            + torch.einsum("ir,jr,kr->ijk", a, self.third, self.second)
        ) / 2


@dataclass
class Crossed211:
    """Lazy crossed (2,1,1) terms from a low-rank matrix core."""
    pairs: list[tuple[torch.Tensor, torch.Tensor]]

    def block(self, start: int, stop: int) -> torch.Tensor:
        n = self.pairs[0][0].shape[0]
        out = torch.zeros((stop - start, n, n),
                          device=self.pairs[0][0].device,
                          dtype=self.pairs[0][0].dtype)
        for x, y in self.pairs:
            out += (x[start:stop, :, None] * y[start:stop, None, :]
                    + y[start:stop, :, None] * x[start:stop, None, :]) / 3
        return out


@dataclass
class CoreSource:
    u: torch.Tensor
    v: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor
    core: torch.Tensor
    tag: str = "source"

    def __post_init__(self) -> None:
        n, ra = self.u.shape
        rb = self.s.shape[1]
        assert self.v.shape == (n, ra)
        assert self.s.shape == self.t.shape == (n, rb)
        assert self.core.shape == (ra, rb)

    def transported(self, q: torch.Tensor) -> CoreSource:
        # A birth group can share q and transformed legs across its sources.
        return CoreSource(q @ self.u, q @ self.v, q @ self.s, q @ self.t,
                          self.core, self.tag)

    def same_slices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[ThreeLeg, ThreeLeg]]:
        """Exact (4), (3,1), same-side (2,2), lazy same-side (2,1,1)."""
        u, v, s, t, z = self.u, self.v, self.s, self.t, self.core
        da, db = u * v, s * t
        az = da @ z
        bz = db @ z.T
        d4 = (az * db).sum(dim=1)
        d31 = (
            ((az * s) @ t.T + (az * t) @ s.T)
            + ((bz * u) @ v.T + (bz * v) @ u.T)
        ) / 4
        h22 = az @ db.T
        d22 = (h22 + h22.T) / 6
        d211 = (ThreeLeg(az, s, t, 1 / 6), ThreeLeg(bz, u, v, 1 / 6))
        return d4, d31, d22, d211

    def crossed_slices(self, rank: int) -> tuple[torch.Tensor, Crossed211, float]:
        """Rank-r core approximation for crossed 22 and 211; 4r matrix products.

        Matrices in Crossed211 are evaluation scratch. Persisting them for every
        birth layer would violate the intended memory budget.
        """
        uc, sv, vh = torch.linalg.svd(self.core, full_matrices=False)
        r = min(rank, sv.numel())
        f = uc[:, :r] * sv[:r].sqrt()
        g = vh[:r].T * sv[:r].sqrt()
        energy = float(sv[:r].square().sum() / sv.square().sum())
        d22, crossed211 = self.crossed_slices_with_factors(f, g)
        return d22, crossed211, energy

    def crossed_slices_with_factors(self, f: torch.Tensor,
                                    g: torch.Tensor) -> tuple[torch.Tensor, Crossed211]:
        """Reuse one static core SVD across all later transports."""
        n = self.u.shape[0]
        d22 = torch.zeros((n, n), device=self.u.device, dtype=self.u.dtype)
        pairs = []
        for q in range(f.shape[1]):
            fq, gq = f[:, q], g[:, q]
            x = ((self.u * fq) @ self.v.T + (self.v * fq) @ self.u.T) / 2
            y = ((self.s * gq) @ self.t.T + (self.t * gq) @ self.s.T) / 2
            d22 += (2 / 3) * x * y
            pairs.append((x, y))
        return d22, Crossed211(pairs)

    def dense_reference(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Small-width independent oracle; never used for a competition-width run."""
        u, v, s, t, z = self.u, self.v, self.s, self.t, self.core
        a = (torch.einsum("ia,ja->ija", u, v) + torch.einsum("ia,ja->ija", v, u)) / 2
        b = (torch.einsum("ib,jb->ijb", s, t) + torch.einsum("ib,jb->ijb", t, s)) / 2
        db = torch.einsum("iib->ib", b)
        da = torch.einsum("iia->ia", a)
        d4 = torch.einsum("ia,ab,ib->i", da, z, db)
        d31 = (torch.einsum("ia,ab,ijb->ij", da, z, b)
               + torch.einsum("ija,ab,ib->ij", a, z, db)) / 2
        d22 = (torch.einsum("ia,ab,jb->ij", da, z, db)
               + torch.einsum("ja,ab,ib->ij", da, z, db)) / 6
        d211 = (torch.einsum("ia,ab,jkb->ijk", da, z, b)
                + torch.einsum("jka,ab,ib->ijk", a, z, db)) / 6
        return d4, d31, d22, d211


@dataclass
class LowRankSource:
    """K4 core stored as F @ G.T, with F/G of shape n by r."""
    u: torch.Tensor
    v: torch.Tensor
    s: torch.Tensor
    t: torch.Tensor
    f: torch.Tensor
    g: torch.Tensor

    @classmethod
    def from_core(cls, source: CoreSource, rank: int) -> tuple[LowRankSource, float]:
        uc, sv, vh = torch.linalg.svd(source.core, full_matrices=False)
        r = min(rank, sv.numel())
        low = cls(source.u, source.v, source.s, source.t,
                  uc[:, :r] * sv[:r].sqrt(), vh[:r].T * sv[:r].sqrt())
        return low, float(sv[:r].square().sum() / sv.square().sum())

    def same_slices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[ThreeLeg, ThreeLeg]]:
        u, v, s, t, f, g = self.u, self.v, self.s, self.t, self.f, self.g
        da, db = u * v, s * t
        left, right = da @ f, db @ g
        az, bz = left @ g.T, right @ f.T
        d4 = (left * right).sum(1)
        d31 = (((az * s) @ t.T + (az * t) @ s.T)
               + ((bz * u) @ v.T + (bz * v) @ u.T)) / 4
        h22 = left @ right.T
        d22 = (h22 + h22.T) / 6
        return d4, d31, d22, (ThreeLeg(az, s, t, 1 / 6),
                                ThreeLeg(bz, u, v, 1 / 6))

    def crossed_slices(self) -> tuple[torch.Tensor, Crossed211]:
        u, v, s, t, f, g = self.u, self.v, self.s, self.t, self.f, self.g
        n, rank = f.shape[0], f.shape[1]
        d22 = torch.zeros((n, n), device=u.device, dtype=u.dtype)
        pairs = []
        for q in range(rank):
            fq, gq = f[:, q], g[:, q]
            x = ((u * fq) @ v.T + (v * fq) @ u.T) / 2
            y = ((s * gq) @ t.T + (t * gq) @ s.T) / 2
            d22 += (2 / 3) * x * y
            pairs.append((x, y))
        return d22, Crossed211(pairs)


@dataclass
class DiagonalSource:
    """Full symmetric four-leg source; star and diagonal slice births use this."""
    legs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    weight: torch.Tensor
    tag: str = "diagonal"

    def slices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[ThreeLeg, ...]]:
        legs, w = self.legs, self.weight
        d4 = (legs[0] * legs[1] * legs[2] * legs[3] * w).sum(1)
        d31 = sum(
            (math.prod([legs[j] for j in range(4) if j != i]) * w) @ legs[i].T
            for i in range(4)
        ) / 4
        pairs = list(itertools.combinations(range(4), 2))
        d22 = sum(
            (legs[a] * legs[b] * w) @
            (legs[c] * legs[d]).T
            for a, b in pairs
            for c, d in [tuple(j for j in range(4) if j not in (a, b))]
        ) / 6
        d211 = tuple(
            ThreeLeg(legs[a] * legs[b] * w, legs[c], legs[d], 1 / 6)
            for a, b in pairs
            for c, d in [tuple(j for j in range(4) if j not in (a, b))]
        )
        return d4, d31, d22, d211


@dataclass
class StarSource:
    """Four-leg star Sym(A,P,A,A), including all crossed pairings."""
    a: torch.Tensor
    p: torch.Tensor
    weight: torch.Tensor

    def slices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[ThreeLeg, ThreeLeg]]:
        a, p, w = self.a, self.p, self.weight
        aa = a * a
        ap = a * p
        aaa = aa * a
        d4 = (aaa * p * w).sum(1)
        d31 = (3 * ((aa * p * w) @ a.T) + (aaa * w) @ p.T) / 4
        root = w.abs().sqrt()
        h22 = (ap * root) @ (aa * (root * w.sign())).T
        d22 = (h22 + h22.T) / 2
        d211 = (ThreeLeg(ap * w, a, a, 0.5),
                ThreeLeg(aa * w, p, a, 0.5))
        return d4, d31, d22, d211


def star_birth(a: torch.Tensor, p: torch.Tensor, w3: torch.Tensor) -> StarSource:
    return StarSource(a, p, 4 * w3)


def three2_birth(a: torch.Tensor, p: torch.Tensor,
                 x3: torch.Tensor, p3: torch.Tensor, y3: torch.Tensor,
                 w2: torch.Tensor) -> CoreSource:
    # 12 C_ij κ3_jkl, with the K3 source represented by (X3,P3,Y3).
    return CoreSource(a, p, p3, y3, 12 * w2[:, None] * x3, "three2")


def path_birth(a: torch.Tensor, p: torch.Tensor,
               c_off: torch.Tensor, w2: torch.Tensor) -> CoreSource:
    # 12 A_ij w2_j C_jk w2_k P_kl; crossed terms require a rank reduction.
    z = 12 * w2[:, None] * c_off * w2[None, :]
    return CoreSource(a, p, p, a, z, "path")


def slice22_birth(k22: torch.Tensor) -> CoreSource:
    n = k22.shape[0]
    eye = torch.eye(n, device=k22.device, dtype=k22.dtype)
    return CoreSource(eye, eye, eye, eye, 3 * k22, "slice22")


def slice211_birth(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> CoreSource:
    """A three-leg CP source for K211 becomes a K4 matrix-core birth."""
    n = x.shape[0]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    return CoreSource(eye, eye, y, z, 6 * x, "slice211")


def check(n: int, rank: int, device: str, reference: bool) -> None:
    torch.manual_seed(41)
    dt = torch.float64
    scale = 1 / math.sqrt(n)
    legs = [torch.randn(n, rank, device=device, dtype=dt) * scale for _ in range(4)]
    z = torch.randn(rank, rank, device=device, dtype=dt) * scale
    source = CoreSource(*legs, z, "parity")
    t0 = time.monotonic()
    d4, d31, d22, lazy = source.same_slices()
    ref = source.dense_reference()
    d211 = lazy[0].block(0, n) + lazy[1].block(0, n)
    err = [float((x - y).abs().max()) for x, y in zip((d4, d31, d22, d211), ref)]
    print(f"n={n} rank={rank} device={device} max_abs={err} elapsed={time.monotonic()-t0:.2f}s", flush=True)
    assert max(err) < 1e-11, err
    if reference:
        from k4full import Group, PARTS, group_dslices
        u, v, s, t = legs
        b = (torch.einsum("ib,jb->ijb", s, t)
             + torch.einsum("ib,jb->ijb", t, s)) / 2
        birth = (b.reshape(n * n, rank) @ z.T).reshape(n, n, rank)
        group = Group(("star",), False, u, v, birth, torch.eye(n, device=device, dtype=dt))
        out = {p: torch.zeros((n,) * len(p), device=device, dtype=dt) for p in PARTS}
        t1 = time.monotonic()
        group_dslices(group, out)
        err_ref = [float((x - out[p]).abs().max()) for x, p in zip((d4, d31, d22, d211), PARTS)]
        print(f"k4full_same max_abs={err_ref} elapsed={time.monotonic()-t1:.2f}s", flush=True)
        assert max(err_ref) < 1e-11, err_ref

        # A diagonal core is the star graph's exact full, crossed source.
        weight = torch.randn(rank, device=device, dtype=dt) * scale
        diag = DiagonalSource(tuple(legs), weight, "star")
        d4, d31, d22, lazy = diag.slices()
        d211 = sum(x.block(0, n) for x in lazy)
        birth_diag = b * weight.reshape(1, 1, rank)
        group = Group(("star",), True, u, v, birth_diag,
                      torch.eye(n, device=device, dtype=dt))
        out = {p: torch.zeros((n,) * len(p), device=device, dtype=dt) for p in PARTS}
        group_dslices(group, out)
        err_diag = [float((x - out[p]).abs().max()) for x, p in zip((d4, d31, d22, d211), PARTS)]
        print(f"k4full_diagonal_full max_abs={err_diag}", flush=True)
        assert max(err_diag) < 1e-11, err_diag

        star = StarSource(u, v, weight)
        sd4, sd31, sd22, slazy = star.slices()
        sbirth = ((torch.einsum("ia,ja->ija", u, u)) * weight.reshape(1, 1, rank))
        group = Group(("star",), True, u, v, sbirth,
                      torch.eye(n, device=device, dtype=dt))
        out = {p: torch.zeros((n,) * len(p), device=device, dtype=dt) for p in PARTS}
        group_dslices(group, out)
        serr = [float((x - out[p]).abs().max()) for x, p in zip(
            (sd4, sd31, sd22, slazy[0].block(0, n) + slazy[1].block(0, n)), PARTS)]
        print(f"k4full_star max_abs={serr}", flush=True)
        assert max(serr) < 1e-11, serr

        # Exact low-rank core exercises the n^3-per-rank crossed contraction.
        low_rank = min(16, rank)
        f = torch.randn(rank, low_rank, device=device, dtype=dt) * scale
        g = torch.randn(rank, low_rank, device=device, dtype=dt) * scale
        low = CoreSource(*legs, f @ g.T, "three2")
        s4, s31, s22, s211 = low.same_slices()
        x22, x211, energy = low.crossed_slices(low_rank)
        birth = (b.reshape(n * n, rank) @ low.core.T).reshape(n, n, rank)
        group = Group(("three2",), True, u, v, birth,
                      torch.eye(n, device=device, dtype=dt))
        out = {p: torch.zeros((n,) * len(p), device=device, dtype=dt) for p in PARTS}
        group_dslices(group, out)
        err_cross = [float((x - out[p]).abs().max()) for x, p in zip(
            (s4, s31, s22 + x22,
             s211[0].block(0, n) + s211[1].block(0, n) + x211.block(0, n)), PARTS)]
        print(f"k4full_lowrank_cross rank={low_rank} core_energy={energy:.9f} max_abs={err_cross}", flush=True)
        assert max(err_cross) < 1e-11, err_cross

        compressed, _ = LowRankSource.from_core(low, low_rank)
        c4, c31, c22, clazy = compressed.same_slices()
        cx22, cx211 = compressed.crossed_slices()
        err_compressed = [float((x - out[p]).abs().max()) for x, p in zip(
            (c4, c31, c22 + cx22,
             clazy[0].block(0, n) + clazy[1].block(0, n) + cx211.block(0, n)), PARTS)]
        print(f"k4full_lowrank_stored max_abs={err_compressed}", flush=True)
        assert max(err_compressed) < 1e-11, err_compressed


def ledger(layers: int, rank: int) -> None:
    """Optimistic one-star/one-three2-group per birth; excludes K3 feedback."""
    evaluated = layers * (layers + 1) // 2
    transported = layers * (layers - 1) // 2
    rows = [
        ("star slices 31+22", 3, evaluated),
        ("three2 lowrank same 31", 4, evaluated),
        ("three2 crossed rank-r", 4 * rank, evaluated),
        ("star two-leg transport if not shared", 2, transported),
        ("three2 four-leg transport if not shared", 4, transported),
    ]
    total = 0.0
    print("class | matmul/source | source-layers | 2n^3 units")
    for name, mm, count in rows:
        units = mm * count / 2
        total += units
        print(f"{name} | {mm} | {count} | {units:.0f}")
    thin_count = 5 * rank * evaluated
    thin_units = thin_count / 1024
    shared_full = evaluated * (3 + 4 + 4 * rank) / 2
    print(f"three2 thin products: {5*evaluated} n×n×r calls at r={rank}, {thin_units:.2f} units at n=1024")
    print(f"shared_legs_full_units={shared_full:.0f} unshared_full_units={total:.0f} "
          f"shared_strassen_plus_v29={shared_full*0.51+thin_units+260:.0f} "
          f"unshared_strassen_plus_v29={total*0.51+thin_units+260:.0f}")
    print("layer | active_each_class | star_mm | three2_same_mm | three2_cross_mm | thin_nnr")
    for layer in range(1, layers + 1):
        print(f"{layer} | {layer} | {3*layer} | {4*layer} | {4*rank*layer} | {5*layer}")


def bench(n: int, rank: int, device: str) -> None:
    torch.manual_seed(41)
    dt = torch.float32
    tensors = [torch.randn((n, n), device=device, dtype=dt) / math.sqrt(n)
               for _ in range(4)]
    w = torch.randn(n, device=device, dtype=dt)
    f = torch.randn((n, rank), device=device, dtype=dt) / math.sqrt(n)
    g = torch.randn((n, rank), device=device, dtype=dt) / math.sqrt(n)
    star = StarSource(tensors[0], tensors[1], w)
    three2 = LowRankSource(*tensors, f, g)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def measure(name: str, fn) -> None:
        if device.startswith("cuda"):
            fn()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.monotonic()
        result = fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak = float("nan")
        print(f"{name} n={n} rank={rank} device={device} "
              f"seconds={time.monotonic()-t0:.4f} gpu_peak_gib={peak:.4f}", flush=True)
        return result

    measure("star_slices", star.slices)
    measure("three2_same", three2.same_slices)
    measure("three2_crossed", three2.crossed_slices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "ledger", "bench"))
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--layers", type=int, default=16)
    args = parser.parse_args()
    if args.command == "check":
        check(args.n, args.rank, args.device, args.reference)
    elif args.command == "ledger":
        ledger(args.layers, args.rank)
    else:
        bench(args.n, args.rank, args.device)
