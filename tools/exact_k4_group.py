"""Exact NumPy reader for one factored fourth-cumulant source.

This ports the two-leg, birth-kernel and cumulative-Q storage, and the four
repeated-index reads, from ``lab/k4full.py``. A group holds
``Sym(sum_r A_r (x) B_r)`` with ``A_r=Sym(x_r (x) y_r)``. No rank reduction is
performed. This module is one component of the full nonlinear chain; its
standalone check does not score a network.
"""

from __future__ import annotations

from dataclasses import dataclass

import flopscope.numpy as fnp
import numpy as np


@dataclass
class Group:
    xb: np.ndarray | None  # None denotes the identity birth leg
    yb: np.ndarray | None
    birth: np.ndarray       # (n, n, R), dense only for B
    q: np.ndarray           # cumulative linear and Wick transport
    crossed: bool = True
    identity: bool = False   # true only at birth, before any transport

    def linear(self, weight: np.ndarray) -> None:
        self.q = fnp.matmul(weight, self.q)
        self.identity = False

    def wick(self, slope: np.ndarray) -> None:
        self.q = slope[:, None] * self.q
        self.identity = False


def two_leg_birth(covariance: np.ndarray, k3_distinct: np.ndarray,
                  w1: np.ndarray, w2: np.ndarray, w3: np.ndarray) -> Group:
    """Build the full simple-K4 path, star and three2 source at one layer.

    ``k3_distinct[j,k,l]`` is the symmetric preactivation third cumulant with
    repeated indices zeroed. The output kernel is laid out ``B[k,l,j]``.
    """
    n = covariance.shape[0]
    c = covariance * (1 - np.eye(n, dtype=covariance.dtype))
    x = w1[:, None] * c
    jkl_path = (12 * c[:, :, None] * c[None, :, :]
                * w2[:, None, None] * w2[None, :, None] * w1[None, None, :])
    jkl_star = (4 * c[:, :, None] * c[:, None, :]
                * w3[:, None, None] * w1[None, :, None] * w1[None, None, :])
    jkl_three2 = (12 * k3_distinct * w2[:, None, None]
                   * w1[None, :, None] * w1[None, None, :])
    birth = (jkl_path + jkl_star + jkl_three2).transpose(1, 2, 0)
    birth = (birth + birth.transpose(1, 0, 2)) / 2
    return Group(x, None, birth, fnp.eye(n, dtype=covariance.dtype), identity=True)


def slice_birth(s4: np.ndarray, s31: np.ndarray, s22: np.ndarray,
                s211: np.ndarray) -> Group:
    """Embed the repeated fourth-cumulant slices in the second exact group."""
    n = s4.shape[0]
    # B[k,l,j] = slice4_j delta_jk delta_jl + 4 slice31_jk delta_jl
    #          + 3 slice22_jk delta_kl + 6 slice211_jkl.
    # Construct the dense kernel without unmetered in-place writes.
    eye = fnp.eye(n, dtype=s4.dtype)
    birth = (s4[None, None, :] * eye[:, None, :] * eye[None, :, :]
             + 4 * s31.T[:, None, :] * eye[None, :, :]
             + 3 * s22.T[:, None, :] * eye[:, :, None]
             + 6 * s211.transpose(1, 2, 0))
    birth = (birth + birth.transpose(1, 0, 2)) / 2
    return Group(None, None, birth, fnp.eye(n, dtype=s4.dtype), identity=True)


def slices(group: Group, chunk: int = 128) -> dict[tuple[int, ...], np.ndarray]:
    """Read the (4), (3,1), (2,2), (2,1,1) slices of one exact group.

    Repeated coordinates of the last three arrays are cleared, as in
    ``SK4._dslices``. ``chunk`` only bounds scratch space; it changes no source.
    """
    q = group.q
    n = q.shape[0]
    r = group.birth.shape[2]
    if group.identity:
        x = q if group.xb is None else group.xb
        y = q if group.yb is None else group.yb
        b = group.birth
    else:
        x = q if group.xb is None else fnp.matmul(q, group.xb)
        y = q if group.yb is None else fnp.matmul(q, group.yb)
        m = fnp.matmul(q, group.birth.reshape(n, n * r)).reshape(n, n, r)
        b = fnp.matmul(q, m)
    da = x * y
    db = fnp.diagonal(b, axis1=0, axis2=1).T
    s4 = fnp.sum(da * db, axis=1)
    same_legs = group.xb is None and group.yb is None
    s31 = fnp.matmul(b, da[:, :, None]).squeeze(2) * 0.5
    if same_legs:
        s31 = s31 + fnp.matmul(x * db, x.T) * 0.5
    else:
        s31 = s31 + (fnp.matmul(x * db, y.T) + fnp.matmul(y * db, x.T)) * 0.25
    s22 = (fnp.matmul(da, db.T) + fnp.matmul(db, da.T)) / 6
    s211_blocks = []
    e_blocks = []
    flat = b.reshape(n * n, r)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        bi = b[lo:hi]
        same_b = fnp.matmul(flat, da[lo:hi].T).reshape(n, n, hi - lo).transpose(2, 0, 1)
        u = fnp.matmul(x[None, :, :] * db[lo:hi, None, :], y.T)
        block = same_b / 6 + (u / 6 if same_legs else (u + u.transpose(0, 2, 1)) / 12)
        if group.crossed:
            t = fnp.matmul(y, (bi * x[lo:hi, None, :]).transpose(0, 2, 1))
            if same_legs:
                block = block + (t + t.transpose(0, 2, 1)) / 3
            else:
                t = t + fnp.matmul(x, (bi * y[lo:hi, None, :]).transpose(0, 2, 1))
                block = block + (t + t.transpose(0, 2, 1)) / 6
            e = fnp.matmul(bi * y[None, :, :], x[lo:hi, :, None]).squeeze(2)
            e_blocks.append(e)
        s211_blocks.append(block)

    if group.crossed:
        e = fnp.concatenate(e_blocks, axis=0)
        s22 = s22 + (e + e.T) / 3
    s211 = fnp.concatenate(s211_blocks, axis=0)
    mask2 = 1 - np.eye(n, dtype=q.dtype)
    s31 = s31 * mask2
    s22 = s22 * mask2
    mask3 = (mask2[:, :, None] * mask2[:, None, :]
             * mask2[None, :, :])
    s211 = s211 * mask3
    return {(4,): s4, (3, 1): s31, (2, 2): s22, (2, 1, 1): s211}


def _dense_check() -> None:
    """Independently materialize Sym_4 at width 5 and check every slice."""
    rng = np.random.default_rng(47)
    n, r = 5, 3
    q = rng.standard_normal((n, n))
    xb = rng.standard_normal((n, r))
    yb = rng.standard_normal((n, r))
    birth = rng.standard_normal((n, n, r))
    birth = (birth + birth.transpose(1, 0, 2)) / 2
    group = Group(xb, yb, birth, q)
    got = slices(group, chunk=2)

    x, y = q @ xb, q @ yb
    a = (x[:, None, :] * y[None, :, :] + y[:, None, :] * x[None, :, :]) / 2
    b = np.einsum("ia,jb,abr->ijr", q, q, birth)
    dense = np.zeros((n, n, n, n))
    # Each of the six ordered placements of A and B has weight 1/6.
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    dense[i, j, k, l] = (
                        np.dot(a[i, j], b[k, l]) + np.dot(a[i, k], b[j, l])
                        + np.dot(a[i, l], b[j, k]) + np.dot(a[j, k], b[i, l])
                        + np.dot(a[j, l], b[i, k]) + np.dot(a[k, l], b[i, j])
                    ) / 6
    expected = {
        (4,): np.array([dense[i, i, i, i] for i in range(n)]),
        (3, 1): np.array([[dense[i, i, i, j] if i != j else 0
                           for j in range(n)] for i in range(n)]),
        (2, 2): np.array([[dense[i, i, j, j] if i != j else 0
                           for j in range(n)] for i in range(n)]),
        (2, 1, 1): np.array([[[dense[i, i, j, k] if len({i, j, k}) == 3 else 0
                              for k in range(n)] for j in range(n)] for i in range(n)]),
    }
    for part in expected:
        np.testing.assert_allclose(got[part], expected[part], rtol=2e-13, atol=2e-13)
    print("exact K=4 group: four repeated-index slices match dense Sym_4 at n=5")


if __name__ == "__main__":
    _dense_check()
