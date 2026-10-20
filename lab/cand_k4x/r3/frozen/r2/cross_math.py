"""NumPy reference: C*kappa3 repeated-leaf (211) birth and contractions.

All arrays are O(n^2).  D[i,j] means kappa[i,i,j].  The returned
Q31[i,j] means T[i,i,i,j] (ROW tripled), matching the r2 probe convention.
Keep matrix diagonals until checking them against q4, then zero the two
matrix diagonals before adding to B's off-diagonal slice inputs.

This is the repeated-leaf 3+2 sector, not the full distinct kappa3 sector.
Its quartic polynomial is
  F(x)=12 sum_j b_j*x_j * ((C@(p*x))_j * (D.T@(p*x)^2)_j
                         - ((C*D.T)@(p*x)^3)_j).
Both C and D have zero diagonals; the final subtraction removes the
coincidence of the covariance leaf and the repeated kappa3 leaf.  Thus
only the index partition 211 remains.  Projection removes its complete
radial trace, leaving B's existing radial channel in charge of that part.

rank selects D's SVD approximation BEFORE every contraction.  Its diagonal
is removed analytically, including in the crossed Q22 contraction, so all
three returned slices describe exactly the same approximate tensor.
"""
import numpy as np


def birth(C, D21, p, b, rank=4):
    """Build a source; p=w(1,1), b=w(2,1) at the birth preactivation."""
    C = np.array(C, dtype=np.float64, copy=True)
    D = np.array(D21, dtype=np.float64, copy=True)
    p = np.array(p, dtype=np.float64, copy=True)
    b = np.array(b, dtype=np.float64, copy=True)
    np.fill_diagonal(C, 0)
    np.fill_diagonal(D, 0)
    n = len(p)
    # rank=None retains the full source, useful only for tiny validation.
    u, s, vh = np.linalg.svd(D, full_matrices=False)
    r = n if rank is None else min(int(rank), n)
    left = u[:, :r] * s[:r]
    right = vh[:r].T
    diag_lr = np.sum(left * right, axis=1)
    D = left @ right.T
    np.fill_diagonal(D, 0)
    E = C * D.T
    t = (p * p) @ D
    trace = C * (
        b[:, None] * p[None, :] * (t[:, None] - (p * p)[None, :] * D.T)
        + b[None, :] * p[:, None] * (t[None, :] - (p * p)[:, None] * D)
    )
    np.fill_diagonal(trace, 0)
    # trace(trace)=0 for a pure 211 tensor, so the scalar trace term is zero.
    G = 6.0 * trace / (n + 4.0)
    return dict(C=C, D=D, E=E, p=p, b=b, left=left, right=right,
                diag_lr=diag_lr, G=G)


def contract(source, P, project=True):
    """Return (q4, Q31_rowtrip, Q22), with true diagonals retained.

P maps birth postactivation coordinates to the current preactivation.
Carry P through intervening linear weights AND Wick w1 factors.
Full crossed Q22 costs 2*rank GEMMs plus one diagonal correction.
The remaining source contractions cost 11 GEMMs; projection costs 3.
"""
    C, D, E, p, b = (source[k] for k in ('C', 'D', 'E', 'p', 'b'))
    A = P * b
    V = P * p
    V2 = V * V
    U = V @ C.T
    T = V2 @ D
    H = (V2 * V) @ E.T
    B = U * T - H
    q4 = 12.0 * np.sum(A * B, axis=1)

    J = (A * U) @ D.T
    K = A @ E
    # Row single, column tripled internally; transpose exactly once below.
    Q31 = 3.0 * (A @ B.T + U @ (A * T).T
                  + 2.0 * V @ (V * J).T
                  - 3.0 * V @ (V2 * K).T)

    Z = (A * U) @ T.T
    Q22 = 2.0 * (Z + Z.T)
    Z = (K * V) @ V2.T
    Q22 -= 6.0 * (Z + Z.T)
    left, right = source['left'], source['right']
    for r in range(left.shape[1]):
        X = (A * right[:, r]) @ U.T
        Y = (V * left[:, r]) @ V.T
        Q22 += 4.0 * (X + X.T) * Y
    # Remove the diagonal introduced by low-rank reconstruction of D.
    Z = (A * V * source['diag_lr']) @ (U * V).T
    Q22 -= 4.0 * (Z + Z.T)

    if project:
        metric = P @ P.T
        core = (P @ source['G']) @ P.T
        md = np.diag(metric)
        gd = np.diag(core)
        q4 -= md * gd
        Q31 -= 0.5 * (metric * gd[None, :] + core * md[None, :])
        Q22 -= (md[:, None] * gd[None, :] + gd[:, None] * md[None, :]
                + 4.0 * metric * core) / 6.0
    return q4, Q31.T, Q22
