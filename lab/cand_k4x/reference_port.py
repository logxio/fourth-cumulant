from __future__ import annotations

# MIT License
# Copyright (c) 2026 504aldo
# Adapted from the pool-based Strassen implementation in subs/s06_stack2/estimator.py.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to inclusion of this notice.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

import flopscope.numpy as fnp
STRASSEN_MIN = 8
STRASSEN_FUSE_P = 343
MAX_EL = 1 << 60

class _VH:
    """V37: a scratch view together with its sub-views, made once. A slice of a remote array
    is a client/server round trip (half of all calls of V30 were slices of the pooled
    Strassen buffers); the pools persist across layers and MLPs, so every sub-view of a
    pooled buffer is created on first use and reused afterwards."""
    __slots__ = ("a", "s7", "q4", "h0", "sw", "qs")

    def __init__(self, a):
        self.a = a
        self.s7 = None    # the seven [:, i*P:(i+1)*P] slabs
        self.q4 = None    # quadrants of the last two axes: (00, 01, 10, 11)
        self.h0 = None    # holder of a[0]
        self.sw = None    # swapaxes(a, -1, -2)
        self.qs = None    # swapped views of the quadrants (hub leaf right operands)

    def sev(self):
        s = self.s7
        if s is None:
            a = self.a
            P = a.shape[1] // 7
            s = self.s7 = tuple(a[:, i * P:(i + 1) * P] for i in range(7))
        return s

    def quad(self):
        qd = self.q4
        if qd is None:
            a = self.a
            h, q = a.shape[-2] // 2, a.shape[-1] // 2
            qd = self.q4 = (a[..., :h, :q], a[..., :h, q:], a[..., h:, :q], a[..., h:, q:])
        return qd

    def first(self):
        e = self.h0
        if e is None:
            e = self.h0 = _VH(self.a[0])
        return e

    def swapped(self):
        s = self.sw
        if s is None:
            s = self.sw = fnp.swapaxes(self.a, -1, -2)
        return s

    def quad_swapped(self, i):
        d = self.qs
        if d is None:
            d = self.qs = {}
        s = d.get(i)
        if s is None:
            s = d[i] = fnp.swapaxes(self.quad()[i], -1, -2)
        return s


class _Strassen:
    """Recursive Strassen on batched operands. Layouts (last two axes = the matrix):
      plain: X (bx, P, m, kd) @ Y (by, P, kd, w) -> out (by, P, m, w), bx in {1, by}
      hub:   X (k, P, m, kd), Y (k, P, w, kd)    -> out (P, m, w) = sum_k X_k @ Y_k^T
    Every combo / product is ONE flopscope op over the whole batch, written with out=
    into pooled buffers shared by role (F53: fresh result buffers dominate residual).
    The seven products: M1=(X11+X22)(Y11+Y22) M2=(X21+X22)Y11 M3=X11(Y12-Y22)
    M4=X22(Y21-Y11) M5=(X11+X12)Y22 M6=(X21-X11)(Y11+Y12) M7=(X12-X22)(Y21+Y22);
    C11=M1+M4-M5+M7 C12=M3+M5 C21=M2+M4 C22=M1-M2+M3+M6.  For the hub kernel the
    right operand is Y^T, whose blocks are (Y^T)11=Y[..,:h,:q], (Y^T)12=Y[..,h:,:q],
    (Y^T)21=Y[..,:h,q:], (Y^T)22=Y[..,h:,q:] (no transposes: the kernel contracts j).
    V37: operands travel as _VH holders (cached sub-views); the op stream of every family
    is unchanged (same ops, same order, same buffers)."""

    def __init__(self, dtype, bmax):
        self.dtype = dtype
        self.bmax = int(bmax)
        self.flat = {}    # V30: (role, P) -> flat scratch buffer shared by every block shape
        self.views = {}   # V30: ((role, P), c, h, q) -> (cap, c, h, q) view of its prefix
        self.hold = {}    # V37: ((role, P), c, h, q, b) -> _VH of the (b, c, h, q) prefix
        self.smin = STRASSEN_MIN   # V30: smallest leaf block side (per family, see _predict_core)

    def clear(self):
        self.flat = {}
        self.views = {}
        self.hold = {}

    def _bufh(self, key, shape4):
        """V30: scratch buffer (b, c, h, q) for role key[0] at recursion width key[1].
        The pruned families run at a different block shape (h, q) on every layer, so the
        pools can no longer be keyed by shape (they would pile up) nor rebuilt per layer
        (freeing GBs is residual time). One flat buffer per (role, P) serves every shape
        through a cached reshaped view of its prefix. Buffers are pure scratch inside one
        family call. V37: returns the cached holder of the (b, c, h, q) prefix."""
        b, c, h, q = (int(x) for x in shape4)
        rk = (key[0], key[1])
        hk = (rk, c, h, q, b)
        e = self.hold.get(hk)
        if e is not None:
            return e
        vk = (rk, c, h, q)
        v = self.views.get(vk)
        if v is None or v.shape[0] < b:
            per = c * h * q
            f = self.flat.get(rk)
            if f is None or f.shape[0] < per * b:
                f = fnp.empty((per * b,), dtype=self.dtype)
                self.flat[rk] = f
                self.views = {k_: v_ for k_, v_ in self.views.items() if k_[0] != rk}
                self.hold = {k_: v_ for k_, v_ in self.hold.items() if k_[0] != rk}
            cap = f.shape[0] // per
            v = fnp.reshape(f[:cap * per], (cap, c, h, q))
            self.views[vk] = v
        e = _VH(v if v.shape[0] == b else v[:b])
        self.hold[hk] = e
        return e

    def level(self, m, kd, w, lev):
        """V28: largest level <= lev at which every block side divides and stays
        >= the leaf minimum (a family whose sides do not allow `lev` runs shallower
        instead of falling back to dense)."""
        while lev > 0 and not self._ok(m, kd, w, lev):
            lev -= 1
        return lev

    def _ok(self, m, kd, w, lev):
        d = 2 ** lev
        return (lev > 0 and m % d == 0 and kd % d == 0 and w % d == 0
                and min(m, kd, w) // d >= self.smin)

    @staticmethod
    def _fit(lev, *elems):
        """V37: largest level <= lev whose biggest stored combo buffer stays within MAX_EL
        at five levels (combos are stored at depths 1..lev-1 when the leaf is fused, i.e.
        from level 4 on, and at depths 1..lev below that)."""
        top = max(elems)
        cap = MAX_EL * 1.75 ** 4
        while lev > 1 and top * 1.75 ** (lev - 1 if lev >= 4 else lev) > cap:
            lev -= 1
        return lev

    def _combos(self, Xh, kind, key, tq=False):
        """Seven Strassen combos of the quadrants (X11,X12,X21,X22) of X (b, P, m, kd) into
        a pooled (b, 7P, h, q) buffer (combo i = slab [:, i*P:(i+1)*P]). tq: the quadrants
        are taken in the transposed block order (hub right operand)."""
        b, P, m, kd = Xh.a.shape
        Bh = self._bufh(key, (b, 7 * P, m // 2, kd // 2))
        o = Bh.sev()
        if tq:
            X11, X21, X12, X22 = Xh.quad()
        else:
            X11, X12, X21, X22 = Xh.quad()
        if kind == "L":
            fnp.add(X11, X22, out=o[0]); fnp.add(X21, X22, out=o[1])
            fnp.copyto(o[2], X11);       fnp.copyto(o[3], X22)
            fnp.add(X11, X12, out=o[4]); fnp.subtract(X21, X11, out=o[5])
            fnp.subtract(X12, X22, out=o[6])
        else:
            fnp.add(X11, X22, out=o[0]); fnp.copyto(o[1], X11)
            fnp.subtract(X12, X22, out=o[2]); fnp.subtract(X21, X11, out=o[3])
            fnp.copyto(o[4], X22);       fnp.add(X11, X12, out=o[5])
            fnp.add(X21, X22, out=o[6])
        return Bh

    @staticmethod
    def _assemble(Mh, outh):
        """M (b, 7P, h, w) products -> out (b, P, m, w) quadrants (8 ops)."""
        M = Mh.sev()
        C11, C12, C21, C22 = outh.quad()
        fnp.add(M[0], M[3], out=C11); fnp.subtract(C11, M[4], out=C11)
        fnp.add(C11, M[6], out=C11)
        fnp.add(M[2], M[4], out=C12)
        fnp.add(M[1], M[3], out=C21)
        fnp.subtract(M[0], M[1], out=C22); fnp.add(C22, M[2], out=C22)
        fnp.add(C22, M[5], out=C22)

    def mm(self, X, Y, out, lev):
        """plain: X (bx, P, m, kd), Y (by, P, kd, w) -> out (by, P, m, w). Operands may be
        arrays or _VH holders (persistent operands: pass a holder to reuse its sub-views)."""
        Xh = X if isinstance(X, _VH) else _VH(X)
        Yh = Y if isinstance(Y, _VH) else _VH(Y)
        bx, P, m, kd = Xh.a.shape
        by, w = Yh.a.shape[0], Yh.a.shape[3]
        lev = self._fit(lev, bx * P * m * kd, by * P * kd * w, max(bx, by) * P * m * w)
        self._mm(Xh, Yh, out if isinstance(out, _VH) else _VH(out), lev)

    def _mm(self, Xh, Yh, outh, lev):
        bx, P, m, kd = Xh.a.shape
        by, w = Yh.a.shape[0], Yh.a.shape[3]
        if not self._ok(m, kd, w, lev):
            fnp.matmul(Xh.a, Yh.a, out=outh.a)
            return
        h, q, v = m // 2, kd // 2, w // 2
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            # V28: fused leaf (deep levels only) -- one product at a time into the out quadrants
            self._leaf_mm(Xh.quad(), Yh.quad(), outh.quad(), bx, by, P, h, q, v)
            return
        Xc = self._combos(Xh, "L", ("L", P))
        Yc = self._combos(Yh, "R", ("R", P))
        # V28: the products reuse this level's left-combo buffer when the callee recurses
        # (its combos consume Xc before anything is written into out); a fused-leaf callee
        # reads X while writing out, so it keeps a separate M. Square blocks only.
        callee_recurses = self._ok(h // 2, q // 2, v // 2, lev - 2)
        mkey = ("L", P) if (callee_recurses and q == v) else ("M", P)
        Mh = self._bufh(mkey, (by, 7 * P, h, v))
        self._mm(Xc, Yc, Mh, lev - 1)
        self._assemble(Mh, outh)

    def _leaf_mm(self, XQ, YQ, OQ, bx, by, P, h, q, v):
        """Fused leaf of mm: M6->C22, M4->C11, M1 (+C11, +C22), M2->C21 (C22-=), M3->C12
        (C22+=), M5 (C11-=, C12+=), M7 (C11+=): 7 matmul writes + 8 adds, one combo
        pair alive at a time (copies for the raw-quadrant operands keep every matmul
        input contiguous)."""
        X11, X12, X21, X22 = XQ
        Y11, Y12, Y21, Y22 = YQ
        C11, C12, C21, C22 = OQ
        Lb = self._bufh(("L1", P), (bx, P, h, q)).a
        Rb = self._bufh(("R1", P), (by, P, q, v)).a
        Mb = self._bufh(("M1", P), (by, P, h, v)).a
        # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb)
        fnp.matmul(Lb, Rb, out=C22)                                   # M6 -> C22
        fnp.add(X21, X22, out=Lb)
        fnp.matmul(Lb, Y11, out=C21)                                  # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb)
        fnp.matmul(X22, Rb, out=C11)                                  # M4 -> C11
        fnp.add(C21, C11, out=C21)                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb)
        fnp.matmul(X11, Rb, out=C12)                                  # M3 -> C12
        fnp.add(C22, C12, out=C22)                                    # C22 += M3
        fnp.add(X11, X12, out=Lb)
        fnp.matmul(Lb, Y22, out=Mb)                                   # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb)
        fnp.matmul(Lb, Rb, out=Mb)                                    # M7
        fnp.add(C11, Mb, out=C11)

    def hub(self, X, Y, out, lev):
        """hub: X (k, P, m, kd), Y (k, P, w, kd) -> out (P, m, w) = sum_k X_k Y_k^T.
        `out` is the (P, m, w) array or a _VH holder of its (1, P, m, w) view."""
        if isinstance(out, _VH):
            outh = out
        else:
            outh = _VH(out[None])
            outh.h0 = _VH(out)
        Xh = X if isinstance(X, _VH) else _VH(X)
        Yh = Y if isinstance(Y, _VH) else _VH(Y)
        k, P, m, kd = Xh.a.shape
        w = Yh.a.shape[2]
        lev = self._fit(lev, k * P * m * kd, k * P * w * kd, P * m * w)
        self._hub(Xh, Yh, outh, lev)

    def _hub(self, Xh, Yh, outh, lev):
        k, P, m, kd = Xh.a.shape
        w = Yh.a.shape[2]
        if not self._ok(m, kd, w, lev):
            # dense: batched GEMM over (k, P) then a k-sum (the einsum form
            # 'kpij,kpcj->pic' takes a slow non-BLAS path: 0.5 s per call at 512^2)
            T = self._bufh(("K", P), (k, P, m, w)).a
            fnp.matmul(Xh.a, Yh.swapped(), out=T)
            fnp.sum(T, axis=0, out=outh.first().a)
            return
        h, q, v = m // 2, kd // 2, w // 2
        if P >= STRASSEN_FUSE_P and not self._ok(h, q, v, lev - 1):
            self._leaf_hub(Xh, Yh, outh.first().quad(), k, P, h, q, v)
            return
        Xc = self._combos(Xh, "L", ("L", P))
        # (Y^T) blocks expressed on the un-transposed Y (rows c, cols j)
        Yc = self._combos(Yh, "R", ("R", P), tq=True)
        Mh = self._bufh(("HM", P), (1, 7 * P, h, v))
        self._hub(Xc, Yc, Mh, lev - 1)
        self._assemble(Mh, outh)

    def _leaf_hub(self, Xh, Yh, OQ, k, P, h, q, v):
        """Fused leaf of hub (same product order as _leaf_mm); each product is a batched
        GEMM over (k, P) into T then a k-sum into its quadrant / the (P, h, v) scratch."""
        X11, X12, X21, X22 = Xh.quad()
        Y11, Y21, Y12, Y22 = Yh.quad()       # transposed block order (see class note)
        C11, C12, C21, C22 = OQ
        Lb = self._bufh(("L1", P), (k, P, h, q)).a
        Rh = self._bufh(("R1", P), (k, P, v, q))
        Rb = Rh.a
        RbT = Rh.swapped()
        Y11T = Yh.quad_swapped(0)
        Y22T = Yh.quad_swapped(3)
        T = self._bufh(("K", P), (k, P, h, v)).a
        Mb = self._bufh(("KS", P), (1, P, h, v)).first().a

        def prod(L_, RT_, dst):
            fnp.matmul(L_, RT_, out=T)
            fnp.sum(T, axis=0, out=dst)

        fnp.subtract(X21, X11, out=Lb); fnp.add(Y11, Y12, out=Rb); prod(Lb, RbT, C22)   # M6 -> C22
        fnp.add(X21, X22, out=Lb); prod(Lb, Y11T, C21)                                # M2 -> C21
        fnp.subtract(C22, C21, out=C22)                                               # C22 -= M2
        fnp.subtract(Y21, Y11, out=Rb); prod(X22, RbT, C11)                           # M4 -> C11
        fnp.add(C21, C11, out=C21)                                                    # C21 += M4
        fnp.add(X11, X22, out=Lb); fnp.add(Y11, Y22, out=Rb); prod(Lb, RbT, Mb)       # M1
        fnp.add(C11, Mb, out=C11); fnp.add(C22, Mb, out=C22)
        fnp.subtract(Y12, Y22, out=Rb); prod(X11, RbT, C12)                           # M3 -> C12
        fnp.add(C22, C12, out=C22)                                                    # C22 += M3
        fnp.add(X11, X12, out=Lb); prod(Lb, Y22T, Mb)                                 # M5
        fnp.subtract(C11, Mb, out=C11); fnp.add(C12, Mb, out=C12)
        fnp.subtract(X12, X22, out=Lb); fnp.add(Y21, Y22, out=Rb); prod(Lb, RbT, Mb)  # M7
        fnp.add(C11, Mb, out=C11)



"""Flopscope K4 matrix-core operators matching the torch research interfaces.

The small lazy (2,1,1) blocks are diagnostic only. Production retains their
three legs. All arithmetic on competition arrays stays inside flopscope.numpy.
"""

import itertools
import os
from dataclasses import dataclass, field

import flopscope.numpy as fnp


def _strassen_type():
    return _Strassen


def square_mm(a, b, levels=None):
    """Batched Strassen on a square product; all arithmetic remains metered.

    The single batch at each recursion depth keeps the number of Python calls
    independent of the 7**levels leaf products.
    """
    n = a.shape[-1]
    assert a.shape == b.shape == (n, n)
    if levels is None:
        levels = min(5, max(0, n.bit_length()-5)) if n >= 512 else 0
    if levels == 0 or n % (1 << levels):
        return fnp.matmul(a,b)

    def recurse(x,y,lev):
        if lev == 0:
            return fnp.matmul(x,y)
        batch,dim,_ = x.shape
        h=dim//2
        a11,a12,a21,a22=x[:,:h,:h],x[:,:h,h:],x[:,h:,:h],x[:,h:,h:]
        b11,b12,b21,b22=y[:,:h,:h],y[:,:h,h:],y[:,h:,:h],y[:,h:,h:]
        left=fnp.concatenate((a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22),axis=0)
        right=fnp.concatenate((b11+b22,b11,b12-b22,b21-b11,b22,b11+b12,b21+b22),axis=0)
        m=fnp.reshape(recurse(left,right,lev-1),(7,batch,h,h))
        m1,m2,m3,m4,m5,m6,m7=(m[j] for j in range(7))
        c11=m1+m4-m5+m7
        c12=m3+m5
        c21=m2+m4
        c22=m1-m2+m3+m6
        return fnp.concatenate((fnp.concatenate((c11,c12),axis=2),
                                fnp.concatenate((c21,c22),axis=2)),axis=1)
    return recurse(a[None],b[None],levels)[0]


_STRASSEN = {}


def _bank_memory(label):
    """Opt-in Linux memory trace for large source contractions."""
    if os.environ.get("K4_MEM_DIAG") != "1":
        return
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            fields = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                      for line in status if line.startswith(("VmSize:", "VmRSS:"))}
        pools = sum(int(buffer.size) * int(buffer.dtype.itemsize)
                    for engine in _STRASSEN.values()
                    for buffer in engine.flat.values())
        print(f"k4bank mem {label} vms={fields.get('VmSize')} "
              f"rss={fields.get('VmRSS')} strassen_pool_gib={pools / (1 << 30):.3f}",
              flush=True)
    except (OSError, AttributeError, TypeError):
        pass


def mm(a,b):
    """Use pooled Strassen only when both operands form a large square product."""
    if a.ndim == b.ndim == 2 and a.shape == b.shape and a.shape[0] == a.shape[1] and a.shape[0] >= 512:
        key = str(a.dtype)
        engine = _STRASSEN.get(key)
        if engine is None:
            engine = _STRASSEN[key] = _strassen_type()(a.dtype,1)
        out = fnp.empty(a.shape,dtype=a.dtype)
        engine.mm(a[None,None],b[None,None],out[None,None],5)
        return out
    return fnp.matmul(a,b)


@dataclass
class ThreeLeg:
    first: object
    second: object
    third: object
    weight: float

    def block(self, start: int, stop: int):
        a = self.first[start:stop]
        return self.weight * (fnp.einsum('ir,jr,kr->ijk', a, self.second, self.third)
                              + fnp.einsum('ir,jr,kr->ijk', a, self.third, self.second)) / 2


@dataclass
class Crossed211:
    pairs: list

    def block(self, start: int, stop: int):
        n = self.pairs[0][0].shape[0]
        out = fnp.zeros((stop-start, n, n), dtype=self.pairs[0][0].dtype)
        for x, y in self.pairs:
            out = out + (x[start:stop, :, None] * y[start:stop, None, :]
                         + y[start:stop, :, None] * x[start:stop, None, :]) / 3
        return out


@dataclass
class CoreSource:
    u: object
    v: object
    s: object
    t: object
    core: object
    tag: str = 'source'

    def __post_init__(self):
        n, ra = self.u.shape
        rb = self.s.shape[1]
        assert self.v.shape == (n, ra)
        assert self.s.shape == self.t.shape == (n, rb)
        assert self.core.shape == (ra, rb)

    def transported(self, q):
        return CoreSource(q @ self.u, q @ self.v, q @ self.s, q @ self.t,
                          self.core, self.tag)

    def same_slices(self):
        u, v, s, t, z = self.u, self.v, self.s, self.t, self.core
        da, db = u*v, s*t
        if self.tag == "slice211_ij":
            diagonal = fnp.diag(z)
            az, bz = da * diagonal, db * diagonal
        else:
            az, bz = mm(da,z), mm(db,z.T)
        d4 = fnp.sum(az*db, axis=1)
        d31 = (mm(az*s,t.T) + mm(az*t,s.T)
               + mm(bz*u,v.T) + mm(bz*v,u.T)) / 4
        h22 = mm(az,db.T)
        d22 = (h22 + h22.T) / 6
        return d4, d31, d22, (ThreeLeg(az, s, t, 1/6), ThreeLeg(bz, u, v, 1/6))

    def crossed_slices(self, rank: int):
        uc, sv, vh = fnp.linalg.svd(self.core, full_matrices=False, k=rank)
        r = min(rank, sv.shape[0])
        root = fnp.sqrt(sv[:r])
        f, g = uc[:, :r]*root, vh[:r].T*root
        energy = float(fnp.sum(sv[:r]*sv[:r]) / fnp.sum(self.core*self.core))
        d22, pairs = fnp.zeros((self.u.shape[0],)*2, dtype=self.u.dtype), []
        for q in range(r):
            x = (mm(self.u*f[:, q],self.v.T) + mm(self.v*f[:, q],self.u.T))/2
            y = (mm(self.s*g[:, q],self.t.T) + mm(self.t*g[:, q],self.s.T))/2
            d22 = d22 + (2/3)*x*y
            pairs.append((x, y))
        return d22, Crossed211(pairs), energy

    def crossed_slices_with_factors(self, f, g):
        return LowRankSource(self.u, self.v, self.s, self.t, f, g).crossed_slices()


@dataclass
class LowRankSource:
    u: object
    v: object
    s: object
    t: object
    f: object
    g: object

    @classmethod
    def from_core(cls, source: CoreSource, rank: int):
        uc, sv, vh = fnp.linalg.svd(source.core, full_matrices=False, k=rank)
        r = min(rank, sv.shape[0])
        root = fnp.sqrt(sv[:r])
        low = cls(source.u, source.v, source.s, source.t,
                  uc[:, :r]*root, vh[:r].T*root)
        return low, float(fnp.sum(sv[:r]*sv[:r]) / fnp.sum(source.core*source.core))

    def same_slices(self):
        u, v, s, t, f, g = self.u, self.v, self.s, self.t, self.f, self.g
        da, db = u*v, s*t
        left, right = da @ f, db @ g
        az, bz = left @ g.T, right @ f.T
        d4 = fnp.sum(left*right, axis=1)
        d31 = (mm(az*s,t.T) + mm(az*t,s.T)
               + mm(bz*u,v.T) + mm(bz*v,u.T)) / 4
        h22 = left @ right.T
        return d4, d31, (h22+h22.T)/6, (ThreeLeg(az,s,t,1/6), ThreeLeg(bz,u,v,1/6))

    def crossed_slices(self):
        n, rank = self.u.shape[0], self.f.shape[1]
        d22, pairs = fnp.zeros((n,n), dtype=self.u.dtype), []
        for q in range(rank):
            x = (mm(self.u*self.f[:,q],self.v.T) + mm(self.v*self.f[:,q],self.u.T))/2
            y = (mm(self.s*self.g[:,q],self.t.T) + mm(self.t*self.g[:,q],self.s.T))/2
            d22 = d22 + (2/3)*x*y
            pairs.append((x,y))
        return d22, Crossed211(pairs)


@dataclass
class DiagonalSource:
    legs: tuple
    weight: object
    tag: str = 'diagonal'

    def slices(self):
        legs, w = self.legs, self.weight
        d4 = fnp.sum(legs[0]*legs[1]*legs[2]*legs[3]*w, axis=1)
        d31 = sum(mm(fnp.prod(fnp.stack([legs[j] for j in range(4) if j != i]), axis=0)*w,legs[i].T)
                  for i in range(4))/4
        pairs = list(itertools.combinations(range(4),2))
        d22 = sum(mm(legs[a]*legs[b]*w,(legs[c]*legs[d]).T)
                  for a,b in pairs for c,d in [tuple(j for j in range(4) if j not in (a,b))])/6
        d211 = tuple(ThreeLeg(legs[a]*legs[b]*w,legs[c],legs[d],1/6)
                     for a,b in pairs for c,d in [tuple(j for j in range(4) if j not in (a,b))])
        return d4,d31,d22,d211


@dataclass
class StarSource:
    a: object
    p: object
    weight: object

    def slices(self):
        a,p,w = self.a,self.p,self.weight
        aa, ap = a*a, a*p
        aaa = aa*a
        d4 = fnp.sum(aaa*p*w, axis=1)
        d31 = (3*mm(aa*p*w,a.T) + mm(aaa*w,p.T))/4
        root = fnp.sqrt(fnp.abs(w))
        h22 = mm(ap*root,(aa*(root*fnp.sign(w))).T)
        d22 = (h22+h22.T)/2
        return d4,d31,d22,(ThreeLeg(ap*w,a,a,.5),ThreeLeg(aa*w,p,a,.5))


def star_birth(a,p,w3):
    return StarSource(a,p,4*w3)


def three2_birth(a,p,x3,p3,y3,w2):
    return CoreSource(a,p,p3,y3,12*w2[:,None]*x3,'three2')


def path_birth(a,p,c_off,w2):
    return CoreSource(a,p,p,a,12*w2[:,None]*c_off*w2[None,:],'path')


def slice22_birth(k22):
    eye = fnp.eye(k22.shape[0],dtype=k22.dtype)
    return CoreSource(eye,eye,eye,eye,3*k22,'slice22')


def slice211_birth(x,y,z):
    eye = fnp.eye(x.shape[0],dtype=x.dtype)
    return CoreSource(eye,eye,y,z,6*x,'slice211')


def k3c_slices(c,x,y,z):
    diag = fnp.diag(c)
    d3 = diag*fnp.sum(x*y*z,axis=1)
    d21 = (c*((x*y) @ z.T + (x*z) @ y.T) + (diag[:,None]*y*z) @ x.T)/3
    return d3,d21


def cp_from_weighted_covariance(c,x,y,z,rank):
    xs,ys,zs,ws = [],[],[],[]
    for j in range(x.shape[1]):
        weighted = (y[:,j,None]*c)*z[None,:,j]
        u,s,vh = fnp.linalg.svd(weighted,full_matrices=False,k=rank)
        keep = min(rank,s.shape[0])
        xs.append(fnp.broadcast_to(x[:,j:j+1],(x.shape[0],keep)))
        ys.append(u[:,:keep]); zs.append(vh[:keep].T); ws.append(s[:keep])
    return fnp.concatenate(xs,axis=1),fnp.concatenate(ys,axis=1),fnp.concatenate(zs,axis=1),fnp.concatenate(ws)


def _hub(pairs):
    """One pooled contraction for every full-width matrix pair in a family."""
    if not pairs:
        raise ValueError("empty contraction")
    n = pairs[0][0].shape[0]
    if n >= 512 and len(pairs) > 8:
        return sum((_hub(pairs[start:start + 8])
                    for start in range(0, len(pairs), 8)),
                   fnp.zeros((n, n), dtype=pairs[0][0].dtype))
    if n < 512 or len(pairs) == 1:
        return sum((mm(x, y.T) for x, y in pairs),
                   fnp.zeros((n, n), dtype=pairs[0][0].dtype))
    left = fnp.stack([x for x, _ in pairs], axis=0)
    right = fnp.stack([y for _, y in pairs], axis=0)
    out = fnp.empty((n, n), dtype=left.dtype)
    key = (str(left.dtype), len(pairs))
    engine = _STRASSEN.get(key)
    if engine is None:
        engine = _STRASSEN[key] = _strassen_type()(left.dtype, len(pairs))
    # A depth-five pool for four or more 1024-wide pairs retains several GiB
    # per batch size.  Depth four keeps the family batched and caps the final
    # seven-way scratch expansion; the contraction itself is unchanged.
    levels = 4 if n >= 1024 and len(pairs) >= 4 else 5
    engine.hub(left[:, None], right[:, None], out[None], levels)
    return out


def _crossed_batch(products):
    """Four matrix products make each crossed (2,2)/(2,1,1) pair."""
    n = products[0][0].shape[0]
    if n < 512:
        values = [mm(a, b.T) for a, b in products]
    else:
        left = fnp.stack([a for a, _ in products], axis=0)
        right = fnp.stack([b.T for _, b in products], axis=0)
        values = fnp.empty((len(products), n, n), dtype=left.dtype)
        key = (str(left.dtype), len(products))
        engine = _STRASSEN.get(key)
        if engine is None:
            engine = _STRASSEN[key] = _strassen_type()(left.dtype, len(products))
        engine.mm(left[None], right[None], values[None], 5)
    d22 = fnp.zeros((n, n), dtype=products[0][0].dtype)
    pairs = []
    for j in range(0, len(products), 4):
        x = (values[j] + values[j + 1]) / 2
        y = (values[j + 2] + values[j + 3]) / 2
        d22 = d22 + (2 / 3) * x * y
        pairs.append((x, y))
    return d22, Crossed211(pairs)


def _young_stars(sources):
    n = sources[0].a.shape[0]
    d4 = fnp.zeros((n,), dtype=sources[0].a.dtype)
    p31, p22, terms = [], [], []
    for source in sources:
        a, p, w = source.a, source.p, source.weight
        aa, ap = a * a, a * p
        aaa = aa * a
        d4 = d4 + fnp.sum(aaa * p * w, axis=1)
        p31.extend(((3 * aa * p * w / 4, a), (aaa * w / 4, p)))
        root = fnp.sqrt(fnp.abs(w))
        p22.append((ap * root, aa * (root * fnp.sign(w))))
        terms.extend((ThreeLeg(ap * w, a, a, .5), ThreeLeg(aa * w, p, a, .5)))
    h22 = _hub(p22)
    return d4, _hub(p31), (h22 + h22.T) / 2, terms


def _young_lows(sources):
    n = sources[0].u.shape[0]
    d4 = fnp.zeros((n,), dtype=sources[0].u.dtype)
    p31, lefts, rights, terms, products = [], [], [], [], []
    for source in sources:
        u, v, s, t, f, g = (source.u, source.v, source.s,
                           source.t, source.f, source.g)
        left, right = (u * v) @ f, (s * t) @ g
        az, bz = left @ g.T, right @ f.T
        d4 = d4 + fnp.sum(left * right, axis=1)
        p31.extend(((az * s / 4, t), (az * t / 4, s),
                    (bz * u / 4, v), (bz * v / 4, u)))
        lefts.append(left)
        rights.append(right)
        terms.extend((ThreeLeg(az, s, t, 1 / 6),
                      ThreeLeg(bz, u, v, 1 / 6)))
        for j in range(f.shape[1]):
            products.extend(((u * f[:, j], v), (v * f[:, j], u),
                             (s * g[:, j], t), (t * g[:, j], s)))
    left = fnp.concatenate(lefts, axis=1)
    right = fnp.concatenate(rights, axis=1)
    h22 = left @ right.T
    crossed22, crossed211 = _crossed_batch(products)
    return (d4, _hub(p31), (h22 + h22.T) / 6 + crossed22,
            terms + [crossed211])


@dataclass
class CP3:
    """Ordered rank factors for a three-index slice."""
    terms: list = field(default_factory=list)

    def add(self, x, y, z, coefficient=1.):
        if x.ndim == 1:
            x, y, z = x[:, None], y[:, None], z[:, None]
        if x.shape != y.shape or y.shape != z.shape:
            raise ValueError("CP3 legs must have identical shapes")
        self.terms.append((coefficient * x, y, z))

    def add_path(self, first, second, center, coefficient=1.):
        if center not in (0, 1, 2) or first.shape != second.shape:
            raise ValueError("center must be an axis and edge matrices must match")
        n = first.shape[0]
        eye = fnp.eye(n, dtype=first.dtype)
        neighbors = [axis for axis in range(3) if axis != center]
        legs = [None, None, None]
        legs[center] = eye
        legs[neighbors[0]] = first.T
        legs[neighbors[1]] = second.T
        self.add(*legs, coefficient=coefficient)

    def extend(self, other):
        self.terms.extend(other.terms)

    def scaled(self, coefficient):
        return CP3([(coefficient * x, y, z) for x, y, z in self.terms])

    def scale_axis(self, axis, coefficient):
        out = CP3()
        for term in self.terms:
            legs = list(term)
            legs[axis] = legs[axis] * coefficient[:, None]
            out.add(*legs)
        return out

    def permute(self, axes):
        out = CP3()
        for term in self.terms:
            out.add(*(term[axis] for axis in axes))
        return out

    def repeated(self):
        if not self.terms:
            raise ValueError("Empty CP3 has no width")
        n = self.terms[0][0].shape[0]
        d3 = fnp.zeros((n,), dtype=self.terms[0][0].dtype)
        d21 = fnp.zeros((n, n), dtype=self.terms[0][0].dtype)
        for x, y, z in self.terms:
            d3 = d3 + fnp.sum(x * y * z, axis=1)
            d21 = d21 + (x * y) @ z.T
        return d3, d21

    def to_symmetric_factored(self):
        if not self.terms:
            raise ValueError("Empty CP3 has no width")
        return tuple(fnp.concatenate([term[axis] for term in self.terms], axis=1)
                     for axis in range(3))

    def zero_repeated_from_slices(self, d4, d31, d22):
        n = d4.shape[0]
        eye = fnp.eye(n, dtype=d4.dtype)
        out = CP3(list(self.terms))
        out.add(eye, eye, d31.T, -1)
        out.add(eye, d31.T, eye, -1)
        out.add(d22, eye, eye, -1)
        out.add(eye * d4[:, None], eye, eye, 2)
        return out

    def zero_repeated(self):
        if not self.terms:
            return CP3()
        n = self.terms[0][0].shape[0]
        eye = fnp.eye(n, dtype=self.terms[0][0].dtype)
        ij = fnp.zeros((n, n), dtype=eye.dtype)
        ik = fnp.zeros((n, n), dtype=eye.dtype)
        jk = fnp.zeros((n, n), dtype=eye.dtype)
        triple = fnp.zeros((n,), dtype=eye.dtype)
        for x, y, z in self.terms:
            ij = ij + (x * y) @ z.T
            ik = ik + (x * z) @ y.T
            jk = jk + x @ (y * z).T
            triple = triple + fnp.sum(x * y * z, axis=1)
        out = CP3(list(self.terms))
        out.add(eye, eye, ij.T, -1)
        out.add(eye, ik.T, eye, -1)
        out.add(jk, eye, eye, -1)
        out.add(eye * triple[:, None], eye, eye, 2)
        return out

    def block(self, start, stop):
        if not self.terms:
            raise ValueError("Empty CP3 has no width")
        n = self.terms[0][0].shape[0]
        out = fnp.zeros((stop - start, n, n), dtype=self.terms[0][0].dtype)
        for x, y, z in self.terms:
            for col in range(0, x.shape[1], 32):
                sl = slice(col, min(col + 32, x.shape[1]))
                out = out + fnp.einsum("ir,jr,kr->ijk", x[start:stop, sl],
                                      y[:, sl], z[:, sl])
        return out

    def canonicalize_small(self, width_limit=128):
        if not self.terms:
            return self
        n = self.terms[0][0].shape[0]
        rank = sum(x.shape[1] for x, _, _ in self.terms)
        if rank <= n * n:
            return self
        if n > width_limit:
            raise MemoryError(f"CP rank {rank} exceeds n² at n={n}")
        cube = self.block(0, n)
        eye = fnp.eye(n, dtype=cube.dtype)
        out = CP3()
        out.add(fnp.reshape(cube, (n, n * n)),
                fnp.repeat(eye, n, axis=1), fnp.tile(eye, (1, n)))
        return out

    def _column_scores(self):
        return fnp.concatenate([
            fnp.sqrt(fnp.sum(x * x, axis=0))
            * fnp.sqrt(fnp.sum(y * y, axis=0))
            * fnp.sqrt(fnp.sum(z * z, axis=0))
            for x, y, z in self.terms], axis=0)

    def top_columns(self, limit):
        """Keep the strongest existing CP columns without fitting new factors."""
        if limit < 1:
            raise ValueError("limit must be positive")
        total = sum(term[0].shape[1] for term in self.terms)
        if total <= limit:
            return self
        selected = fnp.argsort(-self._column_scores())[:limit]
        terms = []
        offset = 0
        for term in self.terms:
            width = term[0].shape[1]
            use = selected[(selected >= offset) & (selected < offset + width)] - offset
            if use.shape[0]:
                terms.append(tuple(leg[:, use] for leg in term))
            offset += width
        return CP3(terms)

    def compress_als(self, rank, iterations=4, preselect=None):
        if not self.terms:
            return self
        total = sum(term[0].shape[1] for term in self.terms)
        if total <= rank:
            return self
        r = min(rank, total)
        score = self._column_scores()
        ordering = fnp.argsort(-score)
        selected = ordering[:r]
        picked = [[], [], []]
        offset = 0
        for term in self.terms:
            width = term[0].shape[1]
            use = selected[(selected >= offset) & (selected < offset + width)] - offset
            if use.shape[0]:
                for axis in range(3):
                    picked[axis].append(term[axis][:, use])
            offset += width
        fit = [fnp.concatenate(legs, axis=1) for legs in picked]
        sources = self.terms
        if preselect is not None and preselect < total:
            keep = ordering[:max(r, int(preselect))]
            sources = []
            offset = 0
            for term in self.terms:
                width = term[0].shape[1]
                use = keep[(keep >= offset) & (keep < offset + width)] - offset
                if use.shape[0]:
                    sources.append(tuple(leg[:, use] for leg in term))
                offset += width
        for _ in range(iterations):
            for target in range(3):
                other = [i for i in range(3) if i != target]
                left, right = fit[other[0]], fit[other[1]]
                gram = (left.T @ left) * (right.T @ right)
                is_float32 = gram.dtype == fnp.float32
                solve_gram = gram.astype(fnp.float64) if is_float32 else gram
                ridge_scale = 1e-6 if is_float32 else 1e-10
                ridge = ridge_scale * fnp.maximum(
                    fnp.max(fnp.abs(fnp.diag(solve_gram))), 1e-30)
                solve_gram = solve_gram + ridge * fnp.eye(r, dtype=solve_gram.dtype)
                mttkrp = fnp.zeros_like(fit[target])
                for src in sources:
                    for col in range(0, src[target].shape[1], 1024):
                        sl = slice(col, min(col + 1024, src[target].shape[1]))
                        mttkrp = mttkrp + src[target][:, sl] @ (
                            (src[other[0]][:, sl].T @ left)
                            * (src[other[1]][:, sl].T @ right))
                rhs = mttkrp.T.astype(fnp.float64) if is_float32 else mttkrp.T
                solution = fnp.linalg.solve(solve_gram, rhs).T
                fit[target] = solution.astype(left.dtype) if is_float32 else solution
        out = CP3()
        out.add(*fit)
        return out


@dataclass
class Lazy211:
    terms: list = field(default_factory=list)

    def block(self, start, stop, *, zero_repeated=False):
        if not self.terms:
            raise ValueError("The empty slice has no width")
        out = sum((term.block(start, stop) for term in self.terms))
        if zero_repeated:
            n = out.shape[1]
            mask = fnp.ones(out.shape, dtype=out.dtype)
            for local, i in enumerate(range(start, stop)):
                mask[local, i, :] = 0
                mask[local, :, i] = 0
            for i in range(n):
                mask[:, i, i] = 0
            out = out * mask
        return out

    def to_cp3(self, limit=None):
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            specs, scores = [], []
            for term in self.terms:
                if isinstance(term, ThreeLeg):
                    for y, z in ((term.second, term.third),
                                 (term.third, term.second)):
                        coefficient = term.weight / 2
                        x = term.first
                        specs.append(("three", x, y, z, coefficient))
                        scores.append(fnp.sqrt(fnp.sum((coefficient * x) ** 2, axis=0))
                                      * fnp.sqrt(fnp.sum(y * y, axis=0))
                                      * fnp.sqrt(fnp.sum(z * z, axis=0)))
                else:
                    for first, second in term.pairs:
                        for x, y in ((first, second), (second, first)):
                            specs.append(("path", x, y, None, 1 / 3))
                            scores.append(fnp.sqrt(fnp.sum(x * x, axis=1))
                                          * fnp.sqrt(fnp.sum(y * y, axis=1)) / 3)
            total = sum(part.shape[0] for part in scores)
            if total > limit:
                selected = fnp.argsort(-fnp.concatenate(scores, axis=0))[:limit]
                out = CP3()
                offset = 0
                eye = None
                for (kind, x, y, z, coefficient), score in zip(specs, scores):
                    width = score.shape[0]
                    use = selected[(selected >= offset) & (selected < offset + width)] - offset
                    if use.shape[0]:
                        if kind == "three":
                            out.add(coefficient * x[:, use], y[:, use], z[:, use])
                        else:
                            if eye is None:
                                eye = fnp.eye(x.shape[0], dtype=x.dtype)
                            out.add(eye[:, use] / 3, x.T[:, use], y.T[:, use])
                    offset += width
                return out
        out = CP3()
        for term in self.terms:
            if isinstance(term, ThreeLeg):
                out.add(term.first, term.second, term.third, term.weight / 2)
                out.add(term.first, term.third, term.second, term.weight / 2)
            else:
                for first, second in term.pairs:
                    out.add_path(first, second, 0, 1 / 3)
                    out.add_path(second, first, 0, 1 / 3)
        return out


@dataclass
class Source:
    kind: str
    birth_layer: int
    payload: object
    crossed: bool
    cross_rank: int | None = None
    cross_f: object = None
    cross_g: object = None

    def cross_factors(self):
        if not isinstance(self.payload, CoreSource):
            raise TypeError("cross_factors requires CoreSource")
        if self.cross_f is None or self.cross_g is None:
            u, s, vh = fnp.linalg.svd(self.payload.core, full_matrices=False,
                                        k=self.cross_rank or min(self.payload.core.shape))
            r = min(self.cross_rank or s.shape[0], s.shape[0])
            root = fnp.sqrt(s[:r])
            self.cross_f = u[:, :r] * root
            self.cross_g = vh[:r].T * root
        return self.cross_f, self.cross_g


def _legs(payload):
    if isinstance(payload, StarSource):
        return (payload.a, payload.p)
    if isinstance(payload, DiagonalSource):
        return payload.legs
    return (payload.u, payload.v, payload.s, payload.t)


def _with_legs(payload, legs):
    if isinstance(payload, StarSource):
        return StarSource(*legs, payload.weight)
    if isinstance(payload, DiagonalSource):
        return DiagonalSource(tuple(legs), payload.weight, payload.tag)
    if isinstance(payload, CoreSource):
        return CoreSource(*legs, payload.core, payload.tag)
    return LowRankSource(*legs, payload.f, payload.g)


@dataclass
class _OldMember:
    kind: str
    factors: tuple
    weight: object = None
    f: object = None
    g: object = None
    core: object = None


class _SharedBasis:
    def __init__(self, width, rank, dtype):
        self.width, self.rank = width, min(rank, width)
        self.q = fnp.empty((width, 0), dtype=dtype)
        self.members = []
        self.gram = fnp.empty((0, 0), dtype=dtype)

    def transport(self, operator):
        if self.members:
            self.q = operator @ self.q

    def wick_scale(self, slope):
        if self.members:
            self.q = slope[:, None] * self.q

    def join(self, source, cross_f=None, cross_g=None):
        legs = _legs(source)
        sketch = fnp.array(legs[0][:, :self.rank], copy=True)
        for _ in range(2):
            image = sum((leg @ (leg.T @ sketch) for leg in legs),
                        fnp.zeros_like(sketch))
            if self.members:
                image = image + self.q @ (self.gram @ (self.q.T @ sketch))
            sketch, _ = fnp.linalg.qr(image, mode="reduced")
        q_new = sketch
        if self.members:
            rotation = q_new.T @ self.q
            for member in self.members:
                member.factors = tuple(rotation @ fac for fac in member.factors)
            self.gram = (rotation @ self.gram) @ rotation.T
        else:
            self.gram = fnp.zeros((self.rank, self.rank), dtype=q_new.dtype)
        factors = tuple(q_new.T @ leg for leg in legs)
        for fac in factors:
            self.gram = self.gram + fac @ fac.T
        if isinstance(source, StarSource):
            member = _OldMember("star", factors, weight=source.weight)
        elif isinstance(source, CoreSource):
            member = _OldMember("core_full", factors, f=cross_f,
                                g=cross_g, core=source.core)
        else:
            member = _OldMember("core", factors, f=source.f, g=source.g)
        self.members.append(member)
        self.q = q_new

    def slices(self):
        """Materialize all old legs once; contract each source family in a batch."""
        members = self.members
        q = self.q
        n, k = q.shape
        d4 = fnp.zeros((n,), dtype=q.dtype)
        d31_core = fnp.zeros((n, k), dtype=q.dtype)
        d31_dense = fnp.zeros((n, n), dtype=q.dtype)
        d22 = fnp.zeros((n, n), dtype=q.dtype)
        lazy = Lazy211()
        if not members:
            return d4, fnp.zeros((n, n), dtype=q.dtype), d22, lazy
        all_factors = fnp.concatenate([fac for m in members for fac in m.factors], axis=1)
        all_legs = q @ all_factors
        offset = 0
        star_pairs, low_left, low_right, crossed_cores = [], [], [], []
        full31_pairs, full22_pairs = [], []
        for member in members:
            legs = []
            for fac in member.factors:
                width = fac.shape[1]
                legs.append(all_legs[:, offset:offset + width])
                offset += width
            if member.kind == "star":
                a, p = legs
                fa, fp = member.factors
                w = member.weight
                aa, ap = a * a, a * p
                aaa = aa * a
                d4 = d4 + fnp.sum(aaa * p * w, axis=1)
                d31_core = d31_core + (3 * ((aa * p * w) @ fa.T)
                                         + (aaa * w) @ fp.T) / 4
                root = fnp.sqrt(fnp.abs(w))
                star_pairs.append((ap * root, aa * (root * fnp.sign(w))))
                lazy.terms.extend((ThreeLeg(ap * w, a, a, .5),
                                   ThreeLeg(aa * w, p, a, .5)))
                continue
            u, v, s, t = legs
            fu, fv, fs, ft = member.factors
            if member.kind == "core_full":
                da, db = u * v, s * t
                az, bz = mm(da, member.core), mm(db, member.core.T)
                d4 = d4 + fnp.sum(az * db, axis=1)
                full31_pairs.extend(((az * s / 4, t), (az * t / 4, s),
                                     (bz * u / 4, v), (bz * v / 4, u)))
                full22_pairs.append((az, db))
                lazy.terms.extend((ThreeLeg(az, s, t, 1 / 6),
                                   ThreeLeg(bz, u, v, 1 / 6)))
                f, g = member.f, member.g
                if f is None or g is None:
                    continue
            else:
                f, g = member.f, member.g
                left, right = (u * v) @ f, (s * t) @ g
                az, bz = left @ g.T, right @ f.T
                d4 = d4 + fnp.sum(left * right, axis=1)
                d31_core = d31_core + ((az * s) @ ft.T + (az * t) @ fs.T
                                         + (bz * u) @ fv.T + (bz * v) @ fu.T) / 4
                low_left.append(left)
                low_right.append(right)
                lazy.terms.extend((ThreeLeg(az, s, t, 1 / 6),
                                   ThreeLeg(bz, u, v, 1 / 6)))
            for j in range(f.shape[1]):
                cx = ((fu * f[:, j]) @ fv.T + (fv * f[:, j]) @ fu.T) / 2
                cy = ((fs * g[:, j]) @ ft.T + (ft * g[:, j]) @ fs.T) / 2
                crossed_cores.extend((cx, cy))
        if star_pairs:
            h = _hub(star_pairs)
            d22 = d22 + (h + h.T) / 2
        if low_left:
            left = fnp.concatenate(low_left, axis=1)
            right = fnp.concatenate(low_right, axis=1)
            h = left @ right.T
            d22 = d22 + (h + h.T) / 6
        if full31_pairs:
            d31_dense = d31_dense + _hub(full31_pairs)
            h = _hub(full22_pairs)
            d22 = d22 + (h + h.T) / 6
        if crossed_cores:
            cores = fnp.stack(crossed_cores, axis=0)
            lifted = (q[None] @ cores) @ q.T
            pairs = []
            for j in range(0, len(crossed_cores), 2):
                x, y = lifted[j], lifted[j + 1]
                d22 = d22 + (2 / 3) * x * y
                pairs.append((x, y))
            lazy.terms.append(Crossed211(pairs))
        return d4, d31_core @ q.T + d31_dense, d22, lazy


class K4Bank:
    def __init__(self, width, dtype=fnp.float64, m0=0, k=64,
                 cross_rank=1, triangle_rank=32):
        self.width = width
        self.dtype = dtype
        self.m0 = m0
        self.k = k
        self.cross_rank = cross_rank
        self.triangle_rank = triangle_rank
        self.sources = []
        self.shared = _SharedBasis(width, k, dtype)
        self.eval_layer = 0

    def add(self, kind, birth_layer, payload, crossed, cross_rank=None):
        self.sources.append(Source(kind, birth_layer, payload, crossed,
                                   cross_rank))

    def transport(self, operator):
        """Transport two sources per product and release their old legs at once."""
        _bank_memory("transport_start")
        for start in range(0, len(self.sources), 2):
            group = self.sources[start:start + 2]
            old_legs = [_legs(source.payload) for source in group]
            widths = [[leg.shape[1] for leg in legs] for legs in old_legs]
            whole = operator @ fnp.concatenate(
                [leg for legs in old_legs for leg in legs], axis=1)
            offset = 0
            for source, leg_widths in zip(group, widths):
                transported = []
                for width in leg_widths:
                    transported.append(whole[:, offset:offset + width])
                    offset += width
                source.payload = _with_legs(source.payload, transported)
            del old_legs
            _bank_memory(f"transport_group_{start // 2}")
        self.shared.transport(operator)
        _bank_memory("transport_done")

    def contract_W(self, operator):
        self.transport(operator)
        return self

    def wick_scale(self, slope):
        for source in self.sources:
            source.payload = _with_legs(source.payload,
                                        tuple(slope[:, None] * leg
                                              for leg in _legs(source.payload)))
        self.shared.wick_scale(slope)

    def join_old(self, layer, m0=None, rank=None):
        age = self.m0 if m0 is None else m0
        if rank is not None and rank != self.shared.rank:
            if self.shared.members:
                raise ValueError("Shared rank cannot change after the first join")
            self.shared = _SharedBasis(self.width, rank, self.dtype)
        young = []
        for source in self.sources:
            if (source.kind in ("star", "three2")
                    and layer - source.birth_layer > age
                    and isinstance(source.payload, (StarSource, LowRankSource, CoreSource))):
                f, g = source.cross_factors() if isinstance(source.payload, CoreSource) else (None, None)
                self.shared.join(source.payload, f, g)
            else:
                young.append(source)
        self.sources = young

    def add_star(self, layer, covariance_leg, center_leg, w3):
        self.add("star", layer, star_birth(covariance_leg, center_leg, w3), True)

    def add_three2(self, layer, covariance_leg, center_leg, k3_first,
                   k3_second, k3_third, w2, rank=None):
        born = three2_birth(covariance_leg, center_leg, k3_first,
                            k3_second, k3_third, w2)
        rank = self.cross_rank if rank is None else rank
        low, _ = LowRankSource.from_core(born, rank)
        self.add("three2", layer, low, True, rank)

    def birth_from_slices(self, layer, d4, d31, d22, d211_cp):
        birth_from_slices(self, layer, d4, d31, d22, d211_cp)

    def slices(self, cp_limit=None):
        _bank_memory("slices_start")
        n = self.width
        d4 = fnp.zeros((n,), dtype=self.dtype)
        d31 = fnp.zeros((n, n), dtype=self.dtype)
        d22 = fnp.zeros((n, n), dtype=self.dtype)
        lazy = Lazy211()
        stars = [s.payload for s in self.sources if isinstance(s.payload, StarSource)]
        lows = [s.payload for s in self.sources if isinstance(s.payload, LowRankSource) and s.crossed]
        if stars:
            a, b, c, terms = _young_stars(stars)
            d4, d31, d22 = d4 + a, d31 + b, d22 + c
            lazy.terms.extend(terms)
            _bank_memory("slices_stars")
        if lows:
            a, b, c, terms = _young_lows(lows)
            d4, d31, d22 = d4 + a, d31 + b, d22 + c
            lazy.terms.extend(terms)
            _bank_memory("slices_lows")
        for source in self.sources:
            p = source.payload
            if isinstance(p, StarSource) or (isinstance(p, LowRankSource) and source.crossed):
                continue
            a, b, c, cp = p.slices() if isinstance(p, DiagonalSource) else p.same_slices()
            d4, d31, d22 = d4 + a, d31 + b, d22 + c
            lazy.terms.extend(cp)
            if source.crossed and isinstance(p, CoreSource):
                x22, x211 = p.crossed_slices_with_factors(*source.cross_factors())
                d22 = d22 + x22
                lazy.terms.append(x211)
            elif source.crossed and isinstance(p, LowRankSource):
                x22, x211 = p.crossed_slices()
                d22 = d22 + x22
                lazy.terms.append(x211)
            _bank_memory(f"slices_source_{source.kind}")
        if self.shared.members:
            a, b, c, old = self.shared.slices()
            d4, d31, d22 = d4 + a, d31 + b, d22 + c
            lazy.terms.extend(old.terms)
            _bank_memory("slices_shared")
        _bank_memory("slices_before_cp")
        return d4, d31, d22, lazy.to_cp3(limit=cp_limit)

    def block211(self, lazy, start, stop, *, zero_repeated=False):
        if isinstance(lazy, CP3):
            return lazy.block(start, stop)
        if lazy.terms:
            return lazy.block(start, stop, zero_repeated=zero_repeated)
        return fnp.zeros((stop - start, self.width, self.width), dtype=self.dtype)


def birth_from_slices(bank, layer, d4, d31, d22, d211_cp):
    n = bank.width
    eye = fnp.eye(n, dtype=bank.dtype)
    one = fnp.ones((n,), dtype=bank.dtype)
    # Four coincident legs and the three-plus-one case are star tensors.
    # The star representation needs two transported legs, and the slice
    # contractions collapse duplicate permutations exactly.
    bank.add("slice4", layer, StarSource(eye, eye, d4), True)
    bank.add("slice31", layer,
             StarSource(eye, d31.T, 4 * one), True)
    merged_core = 3 * d22
    terms = d211_cp.terms if isinstance(d211_cp, CP3) else d211_cp
    if terms:
        columns = [[], [], []]
        for term in terms:
            x, y, z = (leg[:, None] if leg.ndim == 1 else leg for leg in term)
            if x.shape != y.shape or y.shape != z.shape or x.shape[0] != n:
                raise ValueError("(2,1,1) CP legs must have matching shapes")
            for bucket, leg in zip(columns, (x, y, z)):
                bucket.append(leg)
        x, y, z = (fnp.concatenate(bucket, axis=1) for bucket in columns)
        bank.add("slice211_cp", layer,
                 CoreSource(eye, eye, y, z, 6 * x, "slice211_cp"), False)
        yz = (x * y) @ z.T + (x * z) @ y.T
        bank.add("slice211_ij", layer,
                 CoreSource(eye, eye, eye, yz.T,
                            fnp.diag(-6 * one), "slice211_ij"), False)
        jk = x @ (y * z).T
        triple = fnp.sum(x * y * z, axis=1)
        merged_core = merged_core - 6 * jk + fnp.diag(12 * triple)
    # These three symmetric pair sources share all four transport legs; their
    # cores add exactly, so one source replaces three full-width contractions.
    bank.add("slice22", layer,
             CoreSource(eye, eye, eye, eye, merged_core, "slice22"), False)


"""Fourth-order ReLU cumulant step, using metered flopscope arrays.

The diagram table is the offline expansion of mlp_kprop.kprop_harmonic
get_all_terms_iso(4, 4), filtered by factored_keeps_term.
"""

import itertools
import math
import os

import flopscope as flops
import flopscope.numpy as fnp


_TERMS = {(1,): (((), 1.0), (((3,),), 0.16666666666666666), (((4,),), 0.041666666666666664)),
 (2,): (((), 1.0), (((3,),), 0.16666666666666666), (((4,),), 0.041666666666666664)),
 (3,): (((), 1.0), (((3,),), 0.16666666666666666), (((4,),), 0.041666666666666664)),
 (4,): (((), 1.0), (((3,),), 0.16666666666666666), (((4,),), 0.041666666666666664)),
 (1, 1): ((((1, 1),), 1.0),
          (((1, 2),), 1.0),
          (((1, 3),), 0.3333333333333333),
          (((1, 1), (1, 1)), 0.5),
          (((2, 2),), 0.25),
          (((0, 3), (1, 1)), 0.3333333333333333),
          (((1, 1), (1, 2)), 1.0),
          (((0, 4), (1, 1)), 0.08333333333333333),
          (((1, 1), (1, 1), (1, 1)), 0.16666666666666666),
          (((1, 1), (2, 2)), 0.25)),
 (2, 1): ((((1, 1),), 1.0),
          (((1, 2),), 0.5),
          (((2, 1),), 0.5),
          (((1, 3),), 0.16666666666666666),
          (((1, 1), (1, 1)), 0.5),
          (((2, 2),), 0.25),
          (((3, 1),), 0.16666666666666666),
          (((0, 3), (1, 1)), 0.16666666666666666),
          (((1, 1), (1, 2)), 0.5),
          (((1, 1), (2, 1)), 0.5),
          (((1, 1), (3, 0)), 0.16666666666666666),
          (((0, 4), (1, 1)), 0.041666666666666664),
          (((1, 1), (1, 1), (1, 1)), 0.16666666666666666),
          (((1, 1), (2, 2)), 0.25),
          (((1, 1), (4, 0)), 0.041666666666666664)),
 (3, 1): ((((1, 1),), 1.0),
          (((1, 2),), 0.5),
          (((2, 1),), 0.5),
          (((1, 3),), 0.16666666666666666),
          (((1, 1), (1, 1)), 0.5),
          (((2, 2),), 0.25),
          (((3, 1),), 0.16666666666666666),
          (((0, 3), (1, 1)), 0.16666666666666666),
          (((1, 1), (1, 2)), 0.5),
          (((1, 1), (2, 1)), 0.5),
          (((1, 1), (3, 0)), 0.16666666666666666),
          (((0, 4), (1, 1)), 0.041666666666666664),
          (((1, 1), (1, 1), (1, 1)), 0.16666666666666666),
          (((1, 1), (2, 2)), 0.25),
          (((1, 1), (4, 0)), 0.041666666666666664)),
 (2, 2): ((((1, 1),), 1.0),
          (((1, 2),), 1.0),
          (((1, 3),), 0.3333333333333333),
          (((1, 1), (1, 1)), 0.5),
          (((2, 2),), 0.25),
          (((0, 3), (1, 1)), 0.3333333333333333),
          (((1, 1), (1, 2)), 1.0),
          (((0, 4), (1, 1)), 0.08333333333333333),
          (((1, 1), (1, 1), (1, 1)), 0.16666666666666666),
          (((1, 1), (2, 2)), 0.25)),
 (1, 1, 1): ((((1, 1, 1),), 1.0),
             (((0, 1, 1), (1, 0, 1)), 3.0),
             (((1, 1, 2),), 1.5),
             (((0, 1, 1), (1, 0, 2)), 3.0),
             (((0, 1, 1), (1, 1, 1)), 3.0),
             (((0, 1, 1), (2, 0, 1)), 3.0),
             (((0, 1, 1), (0, 1, 1), (1, 0, 1)), 3.0),
             (((0, 1, 1), (2, 0, 2)), 1.5),
             (((0, 1, 1), (1, 0, 1), (1, 1, 0)), 1.0)),
 (2, 1, 1): ((((1, 1, 1),), 1.0),
             (((0, 1, 1), (1, 0, 1)), 2.0),
             (((1, 1, 2),), 1.0),
             (((1, 0, 1), (1, 1, 0)), 1.0),
             (((2, 1, 1),), 0.5),
             (((0, 1, 1), (1, 0, 2)), 1.0),
             (((0, 1, 2), (1, 0, 1)), 1.0),
             (((0, 1, 1), (1, 1, 1)), 1.0),
             (((0, 1, 2), (1, 1, 0)), 1.0),
             (((0, 1, 1), (2, 0, 1)), 1.0),
             (((1, 0, 1), (1, 1, 1)), 2.0),
             (((1, 0, 1), (1, 2, 0)), 1.0),
             (((1, 0, 1), (2, 1, 0)), 1.0),
             (((0, 1, 1), (0, 1, 1), (1, 0, 1)), 1.0),
             (((0, 2, 2), (1, 0, 1)), 0.5),
             (((0, 1, 1), (1, 0, 1), (1, 0, 1)), 1.0),
             (((0, 1, 1), (2, 0, 2)), 0.5),
             (((0, 1, 1), (1, 0, 1), (1, 1, 0)), 1.0),
             (((1, 0, 1), (1, 0, 1), (1, 1, 0)), 1.0),
             (((1, 0, 1), (2, 2, 0)), 0.5))}


class SymmetricK3(CP3):
    """Store one ordered factorization; symmetrize only when evaluating it."""

    def transport(self, weight):
        out = SymmetricK3()
        for x, y, z in self.terms:
            out.add(weight @ x, weight @ y, weight @ z)
        return out

    def repeated(self):
        if not self.terms:
            raise ValueError("Empty SymmetricK3 has no width")
        n = self.terms[0][0].shape[0]
        dtype = self.terms[0][0].dtype
        d3 = fnp.zeros(n, dtype=dtype)
        d21 = fnp.zeros((n, n), dtype=dtype)
        for x, y, z in self.terms:
            d3 = d3 + fnp.sum(x * y * z, axis=1)
            d21 = d21 + ((x * y) @ z.T + (x * z) @ y.T + (y * z) @ x.T) / 3
        return d3, d21

    def block(self, start, stop):
        return _sym_cp(self).block(start, stop)


def _option(config, name, default):
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _feedback_active(config, layer):
    selected = _option(config, "k211_feedback_layers", None)
    if selected is None:
        return True
    if isinstance(selected, int):
        return layer < selected
    return layer in selected


def _off(x):
    """Clear equal-index entries without a second unmetered array backend."""
    eye = fnp.eye(x.shape[0], dtype=x.dtype)
    return x * (1 - eye)


def _wick_factory(mean, variance):
    """E[d^k ReLU(Z)^p] from truncated Gaussian moments."""
    sigma = fnp.sqrt(fnp.maximum(variance, 1e-10))
    alpha = mean / sigma
    phi = fnp.exp(-0.5 * alpha * alpha) / math.sqrt(2 * math.pi)
    Phi = flops.stats.norm.cdf(alpha).astype(alpha.dtype)
    cache = {}

    def first(k):
        if k == 0:
            return sigma * phi + mean * Phi
        if k == 1:
            return Phi
        # He_m(alpha), with He_0=1 and He_1=alpha.
        h0 = fnp.ones_like(alpha)
        if k == 2:
            hermite = h0
        else:
            h1 = alpha
            for degree in range(2, k - 1):
                h0, h1 = h1, alpha * h1 - (degree - 1) * h0
            hermite = h1
        return ((-1) ** (k - 2)) * sigma ** (1 - k) * hermite * phi

    def wick(k, p):
        key = k, p
        if key in cache:
            return cache[key]
        if p == 1:
            out = first(k)
        elif k >= p:
            out = math.factorial(p) * first(k - p + 1)
        else:
            q = p - k
            # T_j = E[X^j 1_{X>-alpha}], X~N(0,1).
            moments = [Phi, phi]
            for j in range(2, q + 1):
                moments.append(((-alpha) ** (j - 1)) * phi + (j - 1) * moments[j - 2])
            result = fnp.zeros_like(alpha)
            for j in range(q + 1):
                result = result + math.comb(q, j) * alpha ** (q - j) * moments[j]
            out = (math.factorial(p) / math.factorial(q)) * (sigma ** q) * result
        cache[key] = out
        return out

    return wick


def _sym_cp(cp):
    result = CP3()
    for order in itertools.permutations(range(3)):
        result.extend(cp.permute(order).scaled(1 / 6))
    return result


def _repeated(cp, n, dtype):
    if cp.terms:
        return cp.repeated()
    return fnp.zeros(n, dtype=dtype), fnp.zeros((n, n), dtype=dtype)


def _slice_lookup(C, k3, bank_slices, has_k4=True):
    n = C.shape[0]
    d3, d21 = _repeated(k3, n, C.dtype)
    d4, d31, d22, d211 = bank_slices
    d21 = _off(d21)
    d31 = _off(d31)
    d22 = _off(d22)
    c_off = _off(C)
    cache = {}

    def get(vec):
        if vec in cache:
            return cache[vec]
        degree = sum(vec)
        occupied = [i for i, v in enumerate(vec) if v]
        part = tuple(vec[i] for i in occupied)
        if (degree == 3 and not k3.terms) or (degree == 4 and not has_k4):
            cache[vec] = None
            return None
        if degree == 2:
            value = fnp.diag(C) if len(part) == 1 else c_off
        elif degree == 3:
            if len(part) == 1:
                value = d3
            elif len(part) == 2:
                value = d21 if part == (2, 1) else d21.T
            else:
                value = _sym_cp(k3)
        elif degree == 4:
            if len(part) == 1:
                value = d4
            elif len(part) == 2:
                value = d31 if part == (3, 1) else (d31.T if part == (1, 3) else d22)
            elif len(part) == 3:
                repeated = part.index(2)
                singles = [i for i in range(3) if i != repeated]
                order = tuple(0 if i == repeated else 1 + singles.index(i) for i in range(3))
                value = d211.permute(order)
            else:
                raise ValueError(f"four-axis slice must remain factored: {part}")
        else:
            raise ValueError(f"unsupported cumulant degree {degree}")
        cache[vec] = value
        return value

    return get


def _low_power_slices(get_slice, wick):
    result = {}
    for part, terms in _TERMS.items():
        axes = len(part)
        if axes > 2:
            continue
        total = None
        for vec_part, coefficient in terms:
            degrees = tuple(sum(v[axis] for v in vec_part) for axis in range(axes))
            term = coefficient * wick(degrees[0], part[0])
            if axes == 2:
                term = term[:, None] * wick(degrees[1], part[1])[None, :]
            for vec in vec_part:
                active = [i for i, value in enumerate(vec) if value]
                value = get_slice(vec)
                if value is None:
                    term = None
                    break
                if axes == 1 or len(active) == 2:
                    term = term * value
                elif active[0] == 0:
                    term = term * value[:, None]
                else:
                    term = term * value[None, :]
            if term is not None:
                total = term if total is None else total + term
        if axes == 2:
            total = _off(total)
            if part[0] == part[1]:
                total = (total + total.T) / 2
        result[part] = total
    return result


def _edge(edges, a, b):
    value = edges[min(a, b), max(a, b)]
    return value if a < b else value.T


def _cycle_cp(edges, mode, rank, n, dtype):
    out = CP3()
    if mode == "delete":
        return out
    m01, m02, m12 = edges[0, 1], edges[0, 2], edges[1, 2]
    eye = fnp.eye(n, dtype=dtype)
    if mode == "exact":
        if n > 128:
            raise ValueError("exact covariance triangle needs n² CP columns")
        left = fnp.reshape(m01[:, :, None] * m02[:, None, :], (n, n * n))
        middle = fnp.reshape(fnp.broadcast_to(eye[:, :, None], (n, n, n)), (n, n * n))
        right = fnp.reshape(fnp.broadcast_to(eye[:, None, :], (n, n, n)), (n, n * n))
        out.add(left, middle, right * fnp.reshape(m12, (1, n * n)))
        return out
    if mode != "rank" or rank is None or rank < 1:
        raise ValueError(f"invalid triangle policy: {mode}, rank={rank}")
    r = min(rank, n)
    u, s, vh = fnp.linalg.svd(m12, full_matrices=False, k=r)
    out.add(
        fnp.reshape(fnp.broadcast_to(eye[:, :, None], (n, n, r)), (n, n * r)),
        fnp.reshape(m01.T[:, :, None] * (u[:, None, :r] * s[:r]), (n, n * r)),
        fnp.reshape(m02.T[:, :, None] * vh[:r].T[:, None, :], (n, n * r)),
    )
    return out

def _weighted_covariance_cp(edge, x, y, z, rank):
    n = edge.shape[0]
    out = CP3()
    for leg_a, leg_b, leg_c in zip(x.T, y.T, z.T):
        weighted = (leg_b[:, None] * edge) * leg_c[None, :]
        r = min(rank, n)
        u, s, vh = fnp.linalg.svd(weighted, full_matrices=False, k=r)
        out.add(fnp.broadcast_to(leg_a[:, None], (n, r)) * s[:r],
                u[:, :r], vh[:r].T)
    return out


def _scale_cube_top_columns(cube, vectors, limit):
    """Apply three Wick legs only to the strongest resulting CP columns."""
    if limit is None:
        for axis in range(3):
            cube = cube.scale_axis(axis, vectors[axis])
        return cube
    total = sum(term[0].shape[1] for term in cube.terms)
    if total <= limit:
        for axis in range(3):
            cube = cube.scale_axis(axis, vectors[axis])
        return cube
    scores = []
    for term in cube.terms:
        score = None
        for axis in range(3):
            leg = term[axis]
            weighted_norm2 = (leg * leg).T @ (vectors[axis] * vectors[axis])
            score = weighted_norm2 if score is None else score * weighted_norm2
        scores.append(fnp.sqrt(score))
    selected = fnp.argsort(-fnp.concatenate(scores))[:limit]
    result = CP3()
    offset = 0
    for term in cube.terms:
        width = term[0].shape[1]
        use = selected[(selected >= offset) & (selected < offset + width)] - offset
        if use.shape[0]:
            result.add(*(term[axis][:, use] * vectors[axis][:, None]
                         for axis in range(3)))
        offset += width
    return result


def _graph_cp(vec_part, get_slice, wicks, coefficient, triangle, triangle_rank,
              mixed_cycle, mixed_rank, column_limit=None):
    n = wicks[0].shape[0]
    vectors = [coefficient * wicks[0], wicks[1], wicks[2]]
    edges = {}
    cube = None
    for vec in vec_part:
        occupied = [i for i, power in enumerate(vec) if power]
        value = get_slice(vec)
        if value is None:
            return CP3()
        if len(occupied) == 1:
            vectors[occupied[0]] = vectors[occupied[0]] * value
        elif len(occupied) == 2:
            key = tuple(occupied)
            edges[key] = value if key not in edges else edges[key] * value
        elif len(occupied) == 3:
            if cube is not None:
                raise ValueError("one graph has multiple three-axis cumulants")
            cube = value
        else:
            raise ValueError(f"unsupported factor: {vec}")
    if cube is not None:
        if edges:
            if mixed_cycle == "delete":
                return CP3()
            if len(edges) != 1 or mixed_cycle not in ("rank", "exact"):
                raise ValueError("mixed cycle needs one edge and rank/exact policy")
            a, b = next(iter(edges))
            missing = 3 - a - b
            order = (missing, a, b)
            inverse = tuple(order.index(axis) for axis in range(3))
            if mixed_cycle == "exact" and n > 128:
                raise ValueError("exact mixed cycle needs n columns per CP column")
            limit = n if mixed_cycle == "exact" else mixed_rank
            if limit is None or limit < 1:
                raise ValueError("mixed-cycle rank must be positive")
            ranked = CP3()
            for term in cube.terms:
                x, y, z = (term[axis] for axis in order)
                ranked.extend(_weighted_covariance_cp(edges[a, b], x, y, z, limit))
            cube = ranked.permute(inverse)
        return _scale_cube_top_columns(cube, vectors, column_limit)
    out = CP3()
    if not edges:
        out.add(*vectors)
    elif len(edges) == 3:
        out = _cycle_cp(edges, triangle, triangle_rank, n, vectors[0].dtype)
    elif len(edges) == 1:
        (a, b), matrix = next(iter(edges.items()))
        other = 3 - a - b
        legs = [None, None, None]
        legs[a] = fnp.eye(n, dtype=matrix.dtype)
        legs[b] = matrix.T
        legs[other] = fnp.ones((n, n), dtype=matrix.dtype)
        out.add(*legs)
    elif len(edges) == 2:
        center = next(axis for axis in range(3)
                      if sum(axis in pair for pair in edges) == 2)
        legs = [None, None, None]
        legs[center] = fnp.eye(n, dtype=vectors[0].dtype)
        for axis in range(3):
            if axis != center:
                legs[axis] = _edge(edges, center, axis).T
        out.add(*legs)
    else:
        raise ValueError(f"too many graph edges: {len(edges)}")
    for axis in range(3):
        out = out.scale_axis(axis, vectors[axis])
    if column_limit is not None:
        out = out.top_columns(column_limit)
    return out


def _bound_cp(cp, width, rank_cap, preselect_factor=None, iterations=4):
    if rank_cap is None or rank_cap >= width * width:
        return cp.canonicalize_small()
    preselect = (None if preselect_factor is None else
                 max(rank_cap, int(preselect_factor * rank_cap)))
    return cp.compress_als(rank_cap, iterations=iterations, preselect=preselect)


def _three_axis_power(part, terms, get_slice, wick, width, config, rank_cap):
    raw = CP3()
    compress_once = _option(config, "cp_compress_once", False)
    als_iterations = _option(config, "p111_als_iterations", 4) if part == (1, 1, 1) else 4
    preselect_factor = _option(config, "cp_preselect_factor", None)
    preselect = (None if preselect_factor is None or rank_cap is None else
                 max(rank_cap, int(preselect_factor * rank_cap)))
    triangle = _option(config, "triangle", "rank")
    triangle_rank = _option(config, "triangle_rank", 32)
    mixed_cycle = _option(config, "mixed_cycle", "delete")
    mixed_rank = _option(config, "mixed_rank", None)
    for vec_part, coefficient in terms:
        degrees = [sum(v[axis] for v in vec_part) for axis in range(3)]
        wicks = [wick(degrees[axis], part[axis]) for axis in range(3)]
        if (part == (2, 1, 1) and vec_part == ((2, 1, 1),)
                and _option(config, "direct_k211_feedback", False)):
            slope = wick(1, 1)
            wicks[0] = wicks[0] - 2 * slope * slope
        contribution = _graph_cp(vec_part, get_slice, wicks, coefficient,
                                 triangle, triangle_rank, mixed_cycle, mixed_rank,
                                 preselect if compress_once else None)
        raw.extend(contribution)
        if os.environ.get("K4_DIAG") == "1":
            added_rank = sum(term[0].shape[1] for term in contribution.terms)
            total_rank = sum(term[0].shape[1] for term in raw.terms)
            print(f"k4 power={part} vec_part={vec_part} "
                  f"added_rank={added_rank} raw_rank={total_rank}",
                  flush=True)
        if not compress_once and rank_cap is not None and rank_cap < width * width:
            rank = sum(term[0].shape[1] for term in raw.terms)
            if rank > 2 * rank_cap:
                if os.environ.get("K4_DIAG") == "1":
                    print(f"k4 power={part} compress rank={rank} cap={rank_cap}",
                          flush=True)
                raw = raw.compress_als(rank_cap, iterations=als_iterations,
                                       preselect=preselect)
        elif compress_once and preselect is not None:
            rank = sum(term[0].shape[1] for term in raw.terms)
            if rank > 2 * preselect:
                raw = raw.top_columns(preselect)
    if part != (1, 1, 1):
        symmetric = CP3()
        symmetric.extend(raw)
        symmetric.extend(raw.permute((0, 2, 1)))
        raw = symmetric.scaled(0.5)
    if (part == (2, 1, 1) and rank_cap is not None
            and _option(config, "defer_p211_als", False)):
        return raw
    return _bound_cp(raw, width, rank_cap, preselect_factor, als_iterations)


def _k3_from_slices(p111, k3_diag, k3_21, width, dtype):
    k3 = SymmetricK3(list(p111.terms))
    p_diag, p_21 = _repeated(k3, width, dtype)
    delta_21 = _off(k3_21 - p_21)
    delta_diag = k3_diag - p_diag
    eye = fnp.eye(width, dtype=dtype)
    k3.add(eye * delta_diag[:, None] + 3 * delta_21.T, eye, eye)
    return k3


def _k211_from_power(p211, p111, p1, p11):
    out = CP3(list(p211.terms))
    out.extend(_sym_cp(p111).scale_axis(0, p1).scaled(-2))
    out.add_path(p11, p11, center=0, coefficient=-2)
    return out


def _add_two_leg_birth(bank, layer, C, k3, wick, cross_rank,
                       feedback_rank_cap, enable_three2_birth=True):
    n = bank.width
    c_off = _off(C)
    w1, w2, w3 = (wick(k, 1) for k in (1, 2, 3))
    eye = fnp.eye(n, dtype=C.dtype)
    a = w1[:, None] * c_off
    bank.add_star(layer, a, eye, w3)
    if not enable_three2_birth:
        return
    k3_distinct = _sym_cp(k3).zero_repeated()
    rank_before = sum(term[0].shape[1] for term in k3_distinct.terms)
    bill_before = (flops.current_budget().flops_used / (2**31)
                   if os.environ.get("K4_DIAG") == "1" else 0)
    if feedback_rank_cap is not None:
        k3_distinct = k3_distinct.compress_als(feedback_rank_cap)
    if os.environ.get("K4_DIAG") == "1":
        rank_after = sum(term[0].shape[1] for term in k3_distinct.terms)
        bill_after = flops.current_budget().flops_used / (2**31)
        print(f"k4 birth_feedback_als layer={layer} rank={rank_before}->{rank_after} "
              f"bill_units={bill_after-bill_before:.3f}", flush=True)
    if k3_distinct.terms:
        xs, ys, zs = [], [], []
        for x, y, z in k3_distinct.terms:
            for first, second, third in ((x, y, z), (y, x, z), (z, x, y)):
                xs.append(first / 3)
                ys.append(w1[:, None] * second)
                zs.append(w1[:, None] * third)
        x = fnp.concatenate(xs, axis=1)
        y = fnp.concatenate(ys, axis=1)
        z = fnp.concatenate(zs, axis=1)
        bank.add("three2", layer,
                 CoreSource(a, eye, y, z, 12 * w2[:, None] * x, "three2"),
                 True, cross_rank)
        if os.environ.get("K4_DIAG") == "1":
            print(f"k4 three2_core layer={layer} shape={(n, x.shape[1])}",
                  flush=True)


def nonlinear_step(mu, C, k3, bank, layer, config):
    """Advance preactivation cumulants through one ReLU.

    Input and output K3 use compact symmetric CP factors. `bank` is updated in
    place, preserving its factored fourth-order sources. `config` is a dict or
    an object with triangle, mixed_cycle, cross_rank, m0, old_rank and CP caps.
    """
    n = mu.shape[0]
    diagnostic = os.environ.get("K4_DIAG") == "1"
    def trace(phase):
        if diagnostic:
            print(f"k4 layer={layer} phase={phase} "
                  f"bill_units={flops.current_budget().flops_used / 2**31:.3f}",
                  flush=True)
    wick = _wick_factory(mu, fnp.diag(C))
    bank.eval_layer = layer
    feedback_rank_cap = _option(config, "feedback_rank_cap", None)
    slice_feedback_rank_cap = _option(config, "slice_feedback_rank_cap", None)
    if slice_feedback_rank_cap is None:
        slice_feedback_rank_cap = (feedback_rank_cap if feedback_rank_cap is not None
                                   else _option(config, "k211_rank_cap", None))
    k3_feedback_rank_cap = _option(config, "k3_feedback_rank_cap", None)
    if k3_feedback_rank_cap is None:
        k3_feedback_rank_cap = feedback_rank_cap
    source_limit = _option(config, "cp_source_limit", None)
    old_slices = bank.slices(cp_limit=source_limit)
    trace("old_slices")
    has_k4 = bool(bank.sources or bank.shared.members)
    get_slice = _slice_lookup(C, k3, old_slices, has_k4)
    p = _low_power_slices(get_slice, wick)
    trace("low_power")
    p111 = _three_axis_power((1, 1, 1), _TERMS[1, 1, 1], get_slice,
                             wick, n, config, _option(config, "k3_rank_cap", None))
    trace("p111")
    feedback_active = _feedback_active(config, layer)
    p211 = (_three_axis_power((2, 1, 1), _TERMS[2, 1, 1], get_slice,
                              wick, n, config, slice_feedback_rank_cap)
            if feedback_active else CP3())
    trace("p211" if feedback_active else "p211_skipped")
    p1, p2, p3, p4 = (p[part] for part in ((1,), (2,), (3,), (4,)))
    p11, p21, p31, p22 = (p[part] for part in ((1, 1), (2, 1), (3, 1), (2, 2)))
    k1 = p1
    k2_diag = p2 - p1 * p1
    k3_diag = p3 - 3 * p1 * p2 + 2 * p1 ** 3
    k3_21 = p21 - 2 * p1[:, None] * p11
    k4_diag = p4 - 4 * p1 * p3 - 3 * p2 * p2 + 12 * p1 * p1 * p2 - 6 * p1 ** 4
    k4_31 = (p31 + 6 * p1[:, None] ** 2 * p11 - 3 * p1[:, None] * p21
              - 3 * p11 * p2[:, None])
    k4_22 = (p22 + 4 * p1[:, None] * p1[None, :] * p11
              - 2 * p1[None, :] * p21 - 2 * p1[:, None] * p21.T - 2 * p11 * p11)
    k3_post = _k3_from_slices(p111, k3_diag, k3_21, n, C.dtype)
    trace("k3_post")
    k2 = _off(p11) + fnp.diag(k2_diag)

    # Old K4 is multiplied by the first Wick derivative on all four legs.
    slope = wick(1, 1)
    old_count = len(bank.sources)
    bank.wick_scale(slope)
    _add_two_leg_birth(bank, layer, C, k3, wick,
                       _option(config, "cross_rank", 1), k3_feedback_rank_cap,
                       _option(config, "enable_three2_birth", True))
    newborn = K4Bank(n, dtype=C.dtype)
    newborn.sources = bank.sources[old_count:]
    newborn.eval_layer = layer
    new_d4, new_d31, new_d22, new_211 = newborn.slices(cp_limit=source_limit)
    trace("newborn_slices")
    old_d4_pre, old_d31_pre, old_d22_pre, old_211_pre = old_slices
    slope2 = slope * slope
    old_d4 = slope2 * slope2 * old_d4_pre + new_d4
    old_d31 = slope2[:, None] * slope[:, None] * old_d31_pre * slope[None, :] + new_d31
    old_d22 = slope2[:, None] * old_d22_pre * slope2[None, :] + new_d22
    d4 = k4_diag - old_d4
    d31 = _off(k4_31 - old_d31)
    d22 = _off(k4_22 - old_d22)
    if feedback_active:
        k211 = _k211_from_power(p211, p111, p1, p11)
        if _option(config, "direct_k211_feedback", False):
            k211.extend(new_211.scaled(-1))
        else:
            old_211 = (old_211_pre.scale_axis(0, slope2)
                       .scale_axis(1, slope).scale_axis(2, slope))
            old_211.extend(new_211)
            k211.extend(old_211.scaled(-1))
        k211 = _bound_cp(k211, n, slice_feedback_rank_cap,
                         _option(config, "cp_preselect_factor", None),
                         _option(config, "k211_als_iterations", 4))
    else:
        k211 = CP3()
    trace("k211" if feedback_active else "k211_skipped")
    bank.birth_from_slices(layer, d4, d31, d22, k211)
    bank.join_old(layer, _option(config, "m0", 0), _option(config, "old_rank", 64))
    trace("join_old")
    return k1, k2, k3_post, bank


"""Fourth-order factored cumulant estimator for the ARC white-box task.

The prediction path keeps the third cumulant in CP legs and the fourth in
matrix and CP sources.  Configuration is fixed at module scope so that the
same parameters are used for local comparisons and official evaluation.
"""

from dataclasses import dataclass
import os
import resource
import time
import warnings

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP



# Full-width calibration read changed the first design point.
M0 = 2
OLD_RANK = 128
CROSS_RANK = 1
TRIANGLE_RANK = 32
TRIANGLE_MODE = "delete"
MIXED_CYCLE = "delete"
MIXED_RANK = None
K3_RANK_CAP = 256
K211_RANK_CAP = 512
FEEDBACK_RANK_CAP = 128
SLICE_FEEDBACK_RANK_CAP = 512
K3_FEEDBACK_RANK_CAP = 256
CP_COMPRESS_ONCE = True
CP_PRESELECT_FACTOR = 8
CP_SOURCE_LIMIT = 8192
ENABLE_THREE2_BIRTH = False
DIRECT_K211_FEEDBACK = True
DEFER_P211_ALS = True
K211_ALS_ITERATIONS = 1
# The current compressed feedback becomes unstable in the later layers at n=128.
# Keep the channel available for controlled tests; the bounded prototype omits it.
K211_FEEDBACK_LAYERS = 0
P111_ALS_ITERATIONS = 4


@dataclass(frozen=True)
class ChainConfig:
    triangle: str = TRIANGLE_MODE
    triangle_rank: int = TRIANGLE_RANK
    mixed_cycle: str = MIXED_CYCLE
    mixed_rank: int | None = MIXED_RANK
    cross_rank: int = CROSS_RANK
    m0: int = M0
    old_rank: int = OLD_RANK
    k3_rank_cap: int = K3_RANK_CAP
    k211_rank_cap: int = K211_RANK_CAP
    feedback_rank_cap: int | None = FEEDBACK_RANK_CAP
    slice_feedback_rank_cap: int | None = SLICE_FEEDBACK_RANK_CAP
    k3_feedback_rank_cap: int | None = K3_FEEDBACK_RANK_CAP
    cp_compress_once: bool = CP_COMPRESS_ONCE
    cp_preselect_factor: int | None = CP_PRESELECT_FACTOR
    cp_source_limit: int | None = CP_SOURCE_LIMIT
    enable_three2_birth: bool = ENABLE_THREE2_BIRTH
    direct_k211_feedback: bool = DIRECT_K211_FEEDBACK
    defer_p211_als: bool = DEFER_P211_ALS
    k211_als_iterations: int = K211_ALS_ITERATIONS
    k211_feedback_layers: int | tuple[int, ...] | None = K211_FEEDBACK_LAYERS
    p111_als_iterations: int = P111_ALS_ITERATIONS


def run_chain(weights, *, dtype=None, config: ChainConfig | None = None):
    """Run the same deterministic chain used by ``Estimator.predict``.

    ``weights`` follows the whestbench convention: each matrix is stored as
    input-by-output, so the contraction operator is its transpose.
    """
    if not weights:
        raise ValueError("weights must contain at least one layer")
    n = weights[0].shape[0]
    if dtype is None:
        dtype = weights[0].dtype
    cfg = config if config is not None else ChainConfig()
    warnings.filterwarnings("ignore", message="Symmetry lost along dims")
    mu = fnp.zeros(n, dtype=dtype)
    covariance = fnp.eye(n, dtype=dtype)
    k3 = SymmetricK3()
    bank = K4Bank(n, dtype=dtype, m0=cfg.m0, k=cfg.old_rank,
                  cross_rank=cfg.cross_rank,
                  triangle_rank=cfg.triangle_rank)
    predictions = []
    diagnostic = os.environ.get("K4_DIAG") == "1"
    for layer, raw_weight in enumerate(weights):
        started = time.monotonic() if diagnostic else 0.0
        weight = fnp.asarray(raw_weight, dtype=dtype).T
        mu = weight @ mu
        covariance = weight @ covariance @ weight.T
        k3 = k3.transport(weight)
        bank.transport(weight)
        mu, covariance, k3, bank = nonlinear_step(
            mu, covariance, k3, bank, layer, cfg)
        predictions.append(mu)
        if diagnostic:
            units = flops.current_budget().flops_used / (2**31)
            print(f"k4 layer={layer} wall={time.monotonic()-started:.2f}s "
                  f"bill_units={units:.3f} "
                  f"maxrss_gib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.3f}",
                  flush=True)
    return fnp.stack(predictions, axis=0)


class Estimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        # No network-dependent work is performed during setup.
        self.config = ChainConfig()

    def predict(self, mlp: MLP, budget: int):
        return run_chain(mlp.weights, dtype=fnp.float32, config=self.config)
