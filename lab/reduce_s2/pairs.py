"""Kernel-directed matrix pairs for both expensive birth families.

B[a,b,r] ~= sum_s V[s,a,b] U[r,s].  The two legs become
A[s] = sym(X diag(U[:,s]) Y.T), so the represented tensor is
Sym_4 sum_s A[s] x V[s].  No truth is used to choose directions.
The ledger records conventional arithmetic for this replacement only;
the remaining rows retain the previous study/Table7a convention.
"""
import math
import torch

EVENTS = []
LAYER = 0


class Ops:
    def __init__(self, n):
        self.n, self.flops = n, 0

    def mm(self, a, b):
        batch = math.prod(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        self.flops += batch * a.shape[-2] * b.shape[-1] * (2*a.shape[-1]-1)
        return a @ b

    def qr(self, a):
        m, k = a.shape
        self.flops += 2*m*k*k - 2*k**3/3
        return torch.linalg.qr(a, mode='reduced')[0]

    def record(self, family, birth, phase):
        EVENTS.append(dict(family=family, birth=birth, layer=LAYER, phase=phase,
                           flops=self.flops, units=self.flops/(2*self.n**3)))


class PairGroup:
    def __init__(self, g, U, V, A, family, birth, kept):
        self.types, self.crossed = g.types, g.crossed
        self.Q = g.Q
        self.U, self.V, self.A = U, V, A
        self.family, self.birth, self.kept = family, birth, kept
        self.Bb = U  # stable identity across transport; never a dense kernel

    def moved(self, Q):
        out = PairGroup(self, self.U, self.V, self.A, self.family, self.birth, self.kept)
        out.Q = Q
        return out


def compress(g, rank, family, birth, iters=2, over=4):
    n, dev, dt = g.Q.shape[0], g.Q.device, g.Q.dtype
    if rank <= 0:
        return g
    k = min(rank, n)
    B = g.Bb.to(dev)
    F = B.permute(2, 0, 1).reshape(n, n*n)
    op = Ops(n)
    if k == n:
        U = torch.eye(n, device=dev, dtype=dt)
        V = F
    else:
        ell = min(k+over, n)
        # Seeded independently of labels, identical in every experiment.
        gen = torch.Generator(device=dev).manual_seed(731 + birth)
        Y = op.qr(torch.randn(n, ell, generator=gen, device=dev, dtype=dt))
        for _ in range(iters):
            Y = op.qr(op.mm(F, op.mm(F.T, Y)))
        H = op.mm(Y.T, F)
        G = op.mm(H, H.T)
        op.flops += 9*ell**3  # symmetric eigensolver convention, documented proxy
        _, E = torch.linalg.eigh((G+G.T)*0.5)
        E = E[:, -k:]
        U, V = op.mm(Y, E), op.mm(E.T, H)
    kept = float(V.square().sum()/F.square().sum().clamp_min(1e-30))
    V = V.reshape(k, n, n)
    V = (V+V.transpose(1, 2))*0.5
    A = None
    if g.xb is not None or g.yb is not None:
        X = torch.eye(n, device=dev, dtype=dt) if g.xb is None else g.xb.to(dev)
        if g.yb is None:
            A = X[None] * U.T[:, None, :]
        else:
            A = op.mm(X[None] * U.T[:, None, :], g.yb.to(dev).T)
        A = (A+A.transpose(1, 2))*0.5
    op.record(family, birth, 'compress')
    return PairGroup(g, U, V, A, family, birth, kept)


def read(g, out):
    Q, n, s = g.Q, g.Q.shape[0], g.U.shape[1]
    op = Ops(n)
    if g.A is None:
        A = op.mm(Q[None] * g.U.T[:, None, :], Q.T)
    else:
        A = op.mm(op.mm(Q, g.A), Q.T)
    D = op.mm(op.mm(Q, g.V), Q.T)
    da, dd = A.diagonal(dim1=1, dim2=2), D.diagonal(dim1=1, dim2=2)
    out[(4,)] += (da*dd).sum(0)
    out[(3,1)] += 0.5*((da[:,:,None]*D).sum(0)+(dd[:,:,None]*A).sum(0))
    t = op.mm(da.T, dd)
    out[(2,2)] += (t+t.T)/6
    if g.crossed:
        out[(2,2)] += (2/3)*(A*D).sum(0)
    o = out[(2,1,1)]
    o += (op.mm(da.T, D.reshape(s,n*n))+op.mm(dd.T,A.reshape(s,n*n))).reshape(n,n,n)/6
    if g.crossed:
        t = op.mm(A.permute(1,2,0).contiguous(),D.permute(1,0,2).contiguous())
        o += (t+t.transpose(1,2))/3
    op.record(g.family, g.birth, 'read')


def ledger(life, start, ranks):
    from lab.reduce_s1.ledger import units
    base, rows = units(life, start)
    for fam, row in [('nontree','non-tree first reads'),('ck3','C.k3 births and crossing')]:
        if ranks[fam] <= 0:
            continue
        rows[row] = 0.0
        # Birth formation remains in K=4 sources. Charge compression once and
        # ALL actual compressed reads in the counted birth/read window.
        selected = [e for e in EVENTS if e['family']==fam and start<=e['birth']<=13
                    and e['layer']<=14]
        rows[f'{fam} kernel compression + matrix-pair reads'] = sum(e['units'] for e in selected)
    return sum(rows.values()), rows
