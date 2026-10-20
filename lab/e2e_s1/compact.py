"""Compact K=4 state for the late-start chain (end-to-end study).

Same reference nonlinearity as lab/k4full.py (factor_k4.factored_nonlin_kprop_k4,
augment=False, kappa3 dense); only the K4 groups change representation.

  young two-leg group (star+three2 in one kernel, crossed, age <= M0): exact,
      dense birth kernel Bb (n,n,n) and transport Q, read by k4full.group_dslices
      (age 0, Q = I, by an exact n^3 fast path).
  old two-leg groups (age > M0): projected on one shared sketch basis exactly as
      lab/cand_k4x/r4/accuracy.py does, and summed into ONE symmetric k^4 core
      S with basis L: kappa4_old = (L x L x L x L) S.
  slice groups: (4,), (3,1), (2,2) parts exact on their legs (Q, H = Q S31^T,
      D22 = 3 S22); the (2,1,1) kernel F[r,(a,b)] = 6 sym_ab s211[r,a,b] is
      compressed at birth along its repeated-index axis by subspace iteration on
      F itself, F ~ U V, and lives for life as s matrix pairs
      A_s = Q diag(U_s) Q^T, D_s = Q V_s Q^T.

Every group reads the four repeated-index slices of Sym_4(sum_r A_r x B_r):
  (4,)_i     = sum_r A_r[ii] B_r[ii]
  (3,1)_ab   = 1/2 sum_r (A_r[aa] B_r[ab] + B_r[aa] A_r[ab])
  (2,2)_ij   = 1/6 sum_r (A_r[ii] B_r[jj] + B_r[ii] A_r[jj])     + crossed 2/3 sum_r A_r[ij] B_r[ij]
  (2,1,1)_ijk= 1/6 sum_r (A_r[ii] B_r[jk] + B_r[ii] A_r[jk])     + crossed 1/3 sum_r (A_r[ij] B_r[ik] + A_r[ik] B_r[ij])
which is what k4full.group_dslices computes (checked in verify_small.py).
"""
import os, sys, inspect, textwrap, time
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("refs/mlp_cumulant_propagation/src", "tools", "lab"):
    p = os.path.join(ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)
import k4full as kf
from mlp_kprop.diagslice import DSTensor, zero_repeated
import mlp_kprop.factor_k4 as fk4

PARTS = kf.PARTS
CHUNK = int(os.environ.get("E2E_CHUNK", "64"))
CFG = dict(pair_rank=8, pair_iters=2, pair_over=4, core_rank=128, m0=2,
           slice_mode="pairs")   # "pairs": (2,1,1) kernels as matrix pairs; "dense": groups holding slice211 keep k4full's n^3 kernel
STATS = {}
METER = None          # optional callable returning metered units so far (set by the driver)


def _now():
    return (time.time(), METER() if METER is not None else 0.0)


def _acc(key, t0):
    t, u = _now()
    s, du = STATS.get(key, (0.0, 0.0))
    STATS[key] = (s + t - t0[0], du + u - t0[1])


def cp3(a, b, c, out=None, alpha=1.0):
    """out[i,j,k] += alpha * sum_r a[i,r] b[j,r] c[k,r], chunked over i (n^4 matmul)."""
    n, R = a.shape
    if out is None:
        out = torch.zeros((n, b.shape[0], c.shape[0]), dtype=a.dtype)
    cT = c.T.contiguous()
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        blk = a[i0:i1].unsqueeze(1) * b.unsqueeze(0)          # (m, n, R)
        out[i0:i1] += alpha * torch.matmul(blk, cT)
    return out


# ---------------------------------------------------------------- young two-leg

class Dense:
    """k4full.Group with birth layer; ident marks Q == I (age 0, before any transport)."""
    kind = "dense"
    __slots__ = ("types", "crossed", "xb", "yb", "Bb", "Q", "birth", "ident")

    def __init__(self, types, crossed, xb, yb, Bb, Q, birth, ident):
        self.types, self.crossed, self.xb, self.yb, self.Bb = types, crossed, xb, yb, Bb
        self.Q, self.birth, self.ident = Q, birth, ident

    def moved(self, Q, ident=False):
        return Dense(self.types, self.crossed, self.xb, self.yb, self.Bb, Q, self.birth, ident)

    def clone(self):
        return self.moved(self.Q.clone(), self.ident)

    def contract_W(self, W):
        return self.moved(W @ self.Q)

    def wick_(self, w):
        self.Q = w[:, None] * self.Q
        self.ident = False

    def add_dslices(self, out):
        t0 = _now()
        if self.ident is True and self.yb is None:
            _dense_age0(self, out)
            _acc("read_dense_age0", t0)
        else:
            kf.group_dslices(self, out)
            _acc("read_dense_two_leg" if self.types[0] in kf.TWO_LEG else "read_dense_slice", t0)


def _dense_age0(g, out):
    """group_dslices with Q = I, Y = I, X = xb, B = Bb: every term is n^3."""
    X, B = g.xb, g.Bb                                            # X (n,n) [i,r]; B [a,b,r]
    n = X.shape[0]
    ar = torch.arange(n)
    xd = torch.diagonal(X)                                       # dA_ir = X_ii delta_ir
    dB = torch.diagonal(B, dim1=0, dim2=1).T                     # dB[i,r] = B[i,i,r]
    out[(4,)] += xd * torch.diagonal(dB)
    # (3,1)_ab = 1/2 B[a,b,a] X_aa + 1/4 (X o dB)_ab + 1/4 dB_aa X_ba
    Baba = B[ar, :, ar]                                          # [a, b] = B[a,b,a]
    out[(3, 1)] += 0.5 * Baba * xd[:, None] + 0.25 * (X * dB) + 0.25 * torch.diagonal(dB)[:, None] * X.T
    # (2,2): (dA dB^T + dB dA^T)/6, dA dB^T [i,j] = X_ii dB[j,i]
    t = xd[:, None] * dB.T
    out[(2, 2)] += (t + t.T) / 6
    if g.crossed:
        # E[i,j] = sum_r X_ir delta_jr B[i,j,r] = X_ij B[i,j,j]
        E = X * B[:, ar, ar]
        out[(2, 2)] += (E + E.T) / 3
    o = out[(2, 1, 1)]
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        ii = ar[i0:i1]
        # S1[i,j,k] = X_ii B[j,k,i] ; U[i,j,k] = dB[i,k] X[j,k]
        S1 = xd[i0:i1, None, None] * B[:, :, i0:i1].permute(2, 0, 1)
        U = dB[i0:i1].unsqueeze(1) * X.unsqueeze(0)
        blk = S1 / 6 + (U + U.transpose(1, 2)) / 12
        if g.crossed:
            # T[i,j,k] = B[i,k,j] X_ij + X_ji B[i,k,i]
            T = B[i0:i1].permute(0, 2, 1) * X[i0:i1].unsqueeze(2) \
                + X.T[i0:i1].unsqueeze(2) * B[ii, :, ii].unsqueeze(1)
            blk += (T + T.transpose(1, 2)) / 6
        o[i0:i1] += blk


# ---------------------------------------------------------------- slice groups

SPECTRA = []          # filled when E2E_SPECTRUM is set: full eigenvalues of F F^T per birth


def compress_kernel(F, s, start, iters, over):
    """F (n, m) ~ U V by subspace iteration on F F^T from `start` (n, l): U (n,s), V (s,m)."""
    if os.environ.get("E2E_SPECTRUM"):
        G = F @ F.T
        SPECTRA.append(torch.linalg.eigvalsh((G + G.T) * 0.5).flip(0).double().cpu().numpy())
        import numpy as _np
        _np.save(os.environ["E2E_SPECTRUM"], _np.stack(SPECTRA))
    Y = start[:, : s + over]
    Y, _ = torch.linalg.qr(Y)
    for _ in range(iters):
        Z = F.T @ Y
        Y, _ = torch.linalg.qr(F @ Z)
    Bm = Y.T @ F                                                 # (l, m)
    G = Bm @ Bm.T
    evals, evecs = torch.linalg.eigh((G + G.T) * 0.5)
    E = evecs[:, -s:]
    U = Y @ E
    V = E.T @ Bm
    kept = float(evals[-s:].sum() / torch.clamp((F * F).sum(), min=1e-300))
    return U, V, kept


class Slice:
    """A slice group: legs Q (x = y = Q), exact s4/s31/s22 parts, (2,1,1) as matrix pairs."""
    kind = "slice"
    __slots__ = ("types", "crossed", "Q", "s4", "H", "D22", "PA", "PD", "birth", "kept")

    def __init__(self, types, crossed, Q, s4, H, D22, PA, PD, birth, kept=None):
        self.types, self.crossed, self.Q, self.s4, self.H, self.D22 = types, crossed, Q, s4, H, D22
        self.PA, self.PD, self.birth, self.kept = PA, PD, birth, kept

    def _new(self, Q, H, PA, PD):
        return Slice(self.types, self.crossed, Q, self.s4, H, self.D22, PA, PD, self.birth, self.kept)

    def clone(self):
        c = lambda t: None if t is None else t.clone()
        return self._new(self.Q.clone(), c(self.H), c(self.PA), c(self.PD))

    def contract_W(self, W):
        H = None if self.H is None else W @ self.H
        PA = PD = None
        if self.PA is not None:
            PA = torch.matmul(torch.matmul(W, self.PA), W.T)
            PD = torch.matmul(torch.matmul(W, self.PD), W.T)
        return self._new(W @ self.Q, H, PA, PD)

    def wick_(self, w):
        self.Q = w[:, None] * self.Q
        if self.H is not None:
            self.H = w[:, None] * self.H
        if self.PA is not None:
            ww = w[:, None] * w[None, :]
            self.PA = self.PA * ww
            self.PD = self.PD * ww

    def add_dslices(self, out):
        t0 = _now()
        if self.s4 is not None or self.H is not None or self.D22 is not None:
            self._struct(out)
        _acc("read_slice_legs", t0)
        t0 = _now()
        if self.PA is not None:
            self._pairs(out)
        _acc("read_slice_pairs", t0)

    def _struct(self, out):
        Q, H, s4, D22, cr = self.Q, self.H, self.s4, self.D22, self.crossed
        n = Q.shape[0]
        Q2 = Q * Q
        z = torch.zeros_like(Q)
        dB = z.clone()
        if s4 is not None:
            dB += Q2 * s4[None, :]
        if H is not None:
            dB += 4 * H * Q
        QD = None
        if D22 is not None:
            dB += Q2 @ D22.T
            QD = Q2 @ D22                                        # (Q2 D22)_ic
        out[(4,)] += (Q2 * dB).sum(1)
        # (3,1)
        L = dB * Q
        if s4 is not None:
            L = L + Q2 * Q * s4[None, :]
        if H is not None:
            L = L + 2 * Q2 * H
        if QD is not None:
            L = L + QD * Q
        t31 = L @ Q.T
        if H is not None:
            t31 = t31 + 2 * (Q2 * Q) @ H.T
        out[(3, 1)] += 0.5 * t31
        # (2,2)
        t = Q2 @ dB.T
        out[(2, 2)] += (t + t.T) / 6
        if cr:
            c = z.clone()
            if s4 is not None:
                c += Q2 * s4[None, :]
            e = c @ Q2.T
            if H is not None:
                QH = Q * H
                e = e + 2 * (QH @ Q2.T + Q2 @ QH.T)
            if D22 is not None:
                for r0 in range(0, n, CHUNK):
                    r1 = min(r0 + CHUNK, n)
                    # sum_rc D_rc (Q_ir Q_ic)(Q_jr Q_jc): per r, (Q_r o Q) D_r (Q_r o Q)^T
                    K = Q[:, r0:r1].T.unsqueeze(1) * Q.T.unsqueeze(0)          # [r, c, i] = Q_ir Q_ic
                    KD = K * D22[r0:r1].unsqueeze(2)
                    e = e + KD.reshape(-1, n).T @ K.reshape(-1, n)
            out[(2, 2)] += (2.0 / 3.0) * e
        # (2,1,1) = cp3(a1, Q, Q) + E + E^T_jk,  E = cp3(cQ2, Q, H)
        a1 = dB / 6
        if s4 is not None:
            a1 = a1 + Q2 * s4[None, :] * (1 / 6 + (2 / 3 if cr else 0))
        if QD is not None:
            a1 = a1 + QD / 6
        if cr and H is not None:
            a1 = a1 + (4 / 3) * Q * H
        o = out[(2, 1, 1)]
        cp3(a1, Q, Q, out=o)
        if H is not None:
            c = 1 / 3 + (2 / 3 if cr else 0)
            E = cp3(c * Q2, Q, H)
            o += E
            o += E.transpose(1, 2)
            del E
        if cr and D22 is not None:
            # 1/3 (X + X^T_jk), X_i = Q M_i Q^T, M_i = diag(Q_i) D22 diag(Q_i)
            for i0 in range(0, n, CHUNK):
                i1 = min(i0 + CHUNK, n)
                M = Q[i0:i1].unsqueeze(2) * D22.unsqueeze(0) * Q[i0:i1].unsqueeze(1)
                X = torch.matmul(torch.matmul(Q, M), Q.T)
                o[i0:i1] += (X + X.transpose(1, 2)) / 3

    def _pairs(self, out):
        A, D, cr = self.PA, self.PD, self.crossed
        s, n, _ = A.shape
        dA = torch.diagonal(A, dim1=1, dim2=2)                  # (s, n)
        dD = torch.diagonal(D, dim1=1, dim2=2)
        out[(4,)] += (dA * dD).sum(0)
        out[(3, 1)] += 0.5 * ((dA.unsqueeze(2) * D).sum(0) + (dD.unsqueeze(2) * A).sum(0))
        t = dA.T @ dD
        out[(2, 2)] += (t + t.T) / 6
        if cr:
            out[(2, 2)] += (2.0 / 3.0) * (A * D).sum(0)
        o = out[(2, 1, 1)]
        o += ((dA.T @ D.reshape(s, n * n) + dD.T @ A.reshape(s, n * n)) / 6).reshape(n, n, n)
        if cr:
            At, Dt = A.permute(1, 2, 0), D.permute(1, 0, 2)    # At[i,j,s] = A_s[i,j], Dt[i,s,k] = D_s[i,k]
            T = torch.matmul(At.contiguous(), Dt.contiguous())  # [i,j,k] = sum_s A_s[ij] D_s[ik]
            o += (T + T.transpose(1, 2)) / 3


def slice_from_dstensor(ds, types, crossed, birth, start):
    """Compact birth of one slice group of SPEC from the K4 repeated slices."""
    n = ds.n
    dt = ds.dtype
    Q = torch.eye(n, dtype=dt)
    s4 = ds.get_slice((4,)).clone() if "slice4" in types else None
    # slice31: B[j,i,i] += 4 s31[i,j], sym over (a,b) -> F_r = 2 s31[r,:] e_r^T + 2 e_r s31[r,:]^T,
    # so B_r = 2 (h_r q_r^T + q_r h_r^T) with h_r = Q s31[r,:]^T: H = Q S31^T, = S31^T at birth
    H = ds.get_slice((3, 1)).T.clone() if "slice31" in types else None
    D22 = 3 * ds.get_slice((2, 2)).clone() if "slice22" in types else None
    PA = PD = None
    kept = None
    if "slice211" in types:
        t0 = _now()
        s211 = ds.get_slice((2, 1, 1))
        F = (3 * (s211 + s211.transpose(1, 2))).reshape(n, n * n)
        U, V, kept = compress_kernel(F, CFG["pair_rank"], start, CFG["pair_iters"], CFG["pair_over"])
        del F
        s = U.shape[1]
        PA = torch.zeros((s, n, n), dtype=dt)
        PA[:, torch.arange(n), torch.arange(n)] = U.T
        PD = V.reshape(s, n, n).clone()
        PD = 0.5 * (PD + PD.transpose(1, 2))
        _acc("birth_pairs_compress", t0)
    return Slice(types, crossed, Q, s4, H, D22, PA, PD, birth, kept)


def dense_slice(ds, types, crossed, birth):
    """k4full.SK4.from_dstensor for one slice group: dense (n,n,n) kernel, Q = I."""
    t0 = _now()
    n = ds.n
    kw = dict(dtype=ds.dtype)
    B = torch.zeros((n, n, n), **kw)
    if "slice4" in types:
        B += kf._einsum_delta(ds.get_slice((4,)), 'i -> i i i')
    if "slice31" in types:
        B += 4 * kf._einsum_delta(ds.get_slice((3, 1)), 'i j -> j i i')
    if "slice22" in types:
        B += 3 * kf._einsum_delta(ds.get_slice((2, 2)), 'i j -> j j i')
    if "slice211" in types:
        B += 6 * ds.get_slice((2, 1, 1)).permute(1, 2, 0)
    B = (B + B.transpose(0, 1)) / 2
    _acc("birth_dense_slice", t0)
    return Dense(types, crossed, None, None, B, torch.eye(n, **kw), birth, False)


def mode_product(tensor, basis, axis):
    """lab/k4n3_tucker_age.py: contract one tensor axis with basis.T, leaving that axis in place."""
    return torch.tensordot(basis.T, tensor, dims=([1], [axis])).movedim(0, axis)


def reconstruct(core, basis, axis):
    return torch.tensordot(basis, core, dims=([1], [axis])).movedim(0, axis)


def shared_basis(tensor, old, rank):
    """lab/k4n3_tucker_age.py shared_basis, verbatim."""
    n = tensor.shape[0]
    gram = torch.zeros((n, n), device=tensor.device, dtype=tensor.dtype)
    for axis in range(3):
        unfold = tensor.movedim(axis, 0).contiguous().reshape(n, -1)
        gram.addmm_(unfold, unfold.T)
        del unfold
    if old is not None and old.numel():
        old, _ = torch.linalg.qr(old, mode="reduced")
        weight = gram.trace() / (3 * old.shape[1])
        gram.addmm_(old, old.T, beta=1, alpha=float(weight))
    gram = (gram + gram.T) * 0.5
    _, vectors = torch.linalg.eigh(gram)
    return vectors[:, -rank:].contiguous()


class FeedbackProjection:
    """lab/cand_k4x/r4/accuracy.py aged_dslices: old slice211 groups' (2,1,1) read on a k3 Tucker basis."""

    def __init__(self, cls, rank, m0f, birth_of):
        self.mp, self.rc, self.sb = mode_product, reconstruct, shared_basis
        self.cls, self.rank, self.m0f, self.birth_of = cls, rank, m0f, birth_of
        self.original = cls._dslices
        self.layer = -1
        self.prior_q = self.transported_q = self.projected_q = None
        self.projected_layer = -1
        self.reads = []

    def start_layer(self, layer, weight, K_after_linear, wick):
        """Call after linear_kprop of this layer, before the nonlinear step."""
        self.layer = layer
        self.transported_q = weight @ self.prior_q if self.prior_q is not None else None
        if self.transported_q is not None:
            w1 = wick(mean=K_after_linear[1].core, var=K_after_linear[2].core.diag(), k=1, p=1)
            self.transported_q = w1[:, None] * self.transported_q

    def install(self):
        fp = self

        def aged(sk4, groups):
            old = [g for g in groups if "slice211" in g.types and fp.layer - fp.birth_of(g) > fp.m0f]
            if not old or fp.rank >= sk4.n:
                return fp.original(sk4, groups)
            t0 = _now()
            young = [g for g in groups if g not in old]
            fresh = fp.original(sk4, young)
            aged_ = fp.original(sk4, old)
            t1 = _now()
            tensor = aged_[(2, 1, 1)]
            if fp.projected_layer != fp.layer:
                fp.projected_q = fp.sb(tensor, fp.transported_q, fp.rank)
                fp.projected_layer = fp.layer
            core = tensor
            for axis in range(3):
                core = fp.mp(core, fp.projected_q, axis)
            projected = core
            for axis in range(3):
                projected = fp.rc(projected, fp.projected_q, axis)
            projected = zero_repeated(projected)
            aged_[(2, 1, 1)] = projected
            fp.prior_q = fp.projected_q
            fp.reads.append(dict(layer=fp.layer, old_groups=len(old)))
            _acc("feedback_projection", t1)
            return {part: fresh[part] + aged_[part] for part in fresh}

        self.cls._dslices = aged

    def uninstall(self):
        self.cls._dslices = self.original


# ---------------------------------------------------------------- old core

class Core:
    """kappa4_old = (L x L x L x L) S, S symmetric (k,k,k,k); G = sum over births of C C^T."""
    kind = "core"
    types = ("core",)
    __slots__ = ("L", "S", "G", "members")

    def __init__(self, L, S, G, members):
        self.L, self.S, self.G, self.members = L, S, G, members

    def clone(self):
        return Core(self.L.clone(), self.S, self.G, list(self.members))

    def contract_W(self, W):
        return Core(W @ self.L, self.S, self.G, list(self.members))

    def wick_(self, w):
        self.L = w[:, None] * self.L

    def add_dslices(self, out):
        t0 = _now()
        L, S = self.L, self.S
        n, k = L.shape
        Lam = (L.unsqueeze(2) * L.unsqueeze(1)).reshape(n, k * k)          # Lam[i,(p,q)] = L_ip L_iq
        V = Lam @ S.reshape(k * k, k * k)                                   # V[i,(s,t)]
        out[(2, 2)] += V @ Lam.T
        V3 = V.reshape(n, k, k)
        Wv = (V3 * L.unsqueeze(2)).sum(1)                                   # Wv[i,t] = sum_s V[i,s,t] L_is
        out[(3, 1)] += Wv @ L.T
        out[(4,)] += (Wv * L).sum(1)
        o = out[(2, 1, 1)]
        LT = L.T.contiguous()
        for i0 in range(0, n, CHUNK):
            i1 = min(i0 + CHUNK, n)
            o[i0:i1] += torch.matmul(torch.matmul(L, V3[i0:i1]), LT)
        _acc("read_core", t0)


def sym4(T):
    out = torch.zeros_like(T)
    import itertools
    for p in itertools.permutations(range(4)):
        out += T.permute(*p)
    return out / 24


def core_from_dense(g, C):
    """Sym_4(sum_r A_r x B_r) of dense group g with Q replaced by S C, in the basis S (k x n C)."""
    xh = C @ (g.xb if g.xb is not None else torch.eye(C.shape[1], dtype=C.dtype))   # (k, R)
    yh = C @ (g.yb if g.yb is not None else torch.eye(C.shape[1], dtype=C.dtype))
    k, R = xh.shape
    n = C.shape[1]
    Bh = (C @ g.Bb.reshape(n, n * R)).reshape(k, n, R)
    Bh = torch.matmul(C, Bh)                                               # (k, k, R) = C Bb_r C^T
    Ah = 0.5 * (xh.unsqueeze(1) * yh.unsqueeze(0) + yh.unsqueeze(1) * xh.unsqueeze(0))   # (k,k,R)
    T = Ah.reshape(k * k, R) @ Bh.reshape(k * k, R).T
    return sym4(T.reshape(k, k, k, k))


def project_old(K4, layer, weight, births_of):
    """accuracy.py old-source projection, in the merged-core form.

    old = two-leg groups with layer - birth > m0 (dense ones enter, the core holds the rest).
    sketch = weight[:, :k]; twice: image = sum over births of Q Q^T sketch; sketch = qr(image).
    """
    k, m0 = CFG["core_rank"], CFG["m0"]
    enter = [g for g in K4.groups if g.kind == "dense" and g.types[0] in kf.TWO_LEG and layer - g.birth > m0]
    core = [g for g in K4.groups if g.kind == "core"]
    assert len(core) <= 1
    core = core[0] if core else None
    if not enter and core is None:
        return None
    t0 = _now()
    unique = {}
    for g in enter:
        unique.setdefault(g.birth, g.Q)
    sketch = weight[:, :k].contiguous()
    for _ in range(2):
        image = torch.zeros_like(sketch)
        for q in unique.values():
            image = image + q @ (q.T @ sketch)
        if core is not None:
            image = image + core.L @ (core.G @ (core.L.T @ sketch))
        sketch, _ = torch.linalg.qr(image, mode="reduced")
    S_new = None
    G_new = torch.zeros((k, k), dtype=sketch.dtype)
    members = []
    if core is not None:
        P = sketch.T @ core.L                                               # (k, k)
        S = core.S
        for ax in range(4):
            S = torch.tensordot(P, S, dims=([1], [ax])).movedim(0, ax)
        S_new = S
        G_new = G_new + P @ core.G @ P.T
        members += core.members
    for b, q in unique.items():
        C = sketch.T @ q
        G_new = G_new + C @ C.T
    for g in enter:
        C = sketch.T @ g.Q
        Sg = core_from_dense(g, C)
        S_new = Sg if S_new is None else S_new + Sg
        members.append(g.birth)
    groups = [g for g in K4.groups if not (g in enter or g is core)]
    groups.append(Core(sketch, S_new, G_new, members))
    K4.groups = groups
    K4.clear_repeated()
    _acc("project_old", t0)
    return dict(entered=len(enter), births=sorted(set(members)))


# ---------------------------------------------------------------- the K4 tensor

class CSK4:
    """Drop-in for FactoredTensor4 inside the reference nonlinear step (augment=False)."""

    def __init__(self, n, groups=None, repeated=None, device=None, dtype=None):
        self.n, self.d = n, 4
        self.groups = list(groups or [])
        self.device, self.dtype = device, dtype
        self.repeated = repeated if repeated is not None else DSTensor(d=4, n=n, slices=dict(), device=device, dtype=dtype)

    @property
    def ndim(self):
        return 4

    def clear_repeated(self):
        self.repeated = DSTensor(d=4, n=self.n, slices=dict(), device=self.device, dtype=self.dtype)

    def clone(self):
        return CSK4(self.n, [g.clone() for g in self.groups], self.repeated.clone(), self.device, self.dtype)

    def contract_W(self, W):
        return CSK4(self.n, [g.contract_W(W) for g in self.groups], None, self.device, self.dtype)

    def contract_wick_(self, wick):
        for part in list(self.repeated.slices):
            sl = self.repeated.slices[part]
            for ax, p in enumerate(part):
                shape = [1] * len(part); shape[ax] = -1
                sl = sl * (wick ** p).reshape(shape)
            self.repeated.slices[part] = sl
        for g in self.groups:
            g.wick_(wick)

    def _dslices(self, groups):
        n, kw = self.n, dict(device=self.device, dtype=self.dtype)
        out = {p: torch.zeros((n,) * len(p), **kw) for p in PARTS}
        for g in groups:
            g.add_dslices(out)
        return {p: zero_repeated(t) for p, t in out.items()}

    def add_groups_(self, groups):
        if self.repeated.slices:
            new = self._dslices(groups)
            for part in self.repeated.slices:
                self.repeated.slices[part] = self.repeated.slices[part] + new[part]
        self.groups.extend(groups)

    def get_dslice(self, part):
        sp = tuple(sorted(part, reverse=True))
        if sp not in self.repeated.slices:
            self.repeated.slices.update(self._dslices(self.groups))
        return self.repeated.get_slice(part)

    def __add__(self, other):
        return CSK4(self.n, self.groups + other.groups, None, self.device, self.dtype)

    @staticmethod
    def from_dstensor(ds):
        groups = []
        for types, crossed in kf.SPEC:
            if types[0] not in kf.SLICE:
                continue
            if CFG["slice_mode"] == "dense" and "slice211" in types:
                groups.append(dense_slice(ds, types, crossed, BIRTH["layer"]))
            else:
                groups.append(slice_from_dstensor(ds, types, crossed, BIRTH["layer"], BIRTH["start"]))
        return CSK4(ds.n, groups, None, ds.device, ds.dtype)


BIRTH = {"layer": 0, "start": None}   # set by the driver before each nonlinear step


def two_leg_buffers(n, like):
    return kf.two_leg_buffers(n, like)


def add_two_leg(pK, X, groups):
    t0 = _now()
    n = pK.n
    new = []
    for types, crossed, buf in groups:
        B = (buf + buf.transpose(0, 1)) / 2
        I = torch.eye(n, dtype=X.dtype)
        new.append(Dense(types, crossed, X.clone(), None, B, I, BIRTH["layer"], True))
        del B
    groups.clear()
    _acc("birth_two_leg_store", t0)
    pK.add_groups_(new)


src = inspect.getsource(fk4.factored_nonlin_kprop_k4)
for old, new in kf.reps:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
src = textwrap.dedent(src)
ns = dict(fk4.__dict__)
ns.update(FactoredTensor4=CSK4, two_leg_buffers=two_leg_buffers, add_two_leg=add_two_leg)
exec(compile(src, "compact_nonlin", "exec"), ns)
nonlin = ns["factored_nonlin_kprop_k4"]
