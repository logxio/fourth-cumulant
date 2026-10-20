"""Compress only the repeated birth index of a non-tree K4 source.

The small-width reference still forms dense births/K3. Pair transport/read
itself is O(r*n**3) and holds O(r*n**2) state. No truth enters the basis.
"""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
for folder in ('tools', 'refs/mlp_cumulant_propagation/src', 'lab', 'lab/cand_k4x'):
    sys.path.insert(0, str(ROOT / folder))
os.environ.setdefault('AW_SPEC', 'star,three2,slice4+slice31,!slice22+slice211')
import delayed_reference as d
from tree_birth import TreeBirth
from k4n3_young_direct_chain import tree_tensor
torch, np, f = d.torch, d.np, d.f


class PairGroup(f.Group):
    __slots__ = ('u', 'v', 'born')
    def __init__(self, u, v, Q, born):
        super().__init__(('pair211',), False, None, None, v, Q)
        self.u, self.v, self.born = u, v, born
    def moved(self, Q):
        return PairGroup(self.u, self.v, Q, self.born)


def pair_slices(g, out):
    """Same-side Sym(A_s tensor B_s), exactly the reference coefficients."""
    q = g.Q
    # A_s = Q diag(u_s) Q.T; B_s = Q v_s Q.T.
    aa = (q[None] * g.u.T[:, None, :]) @ q.T
    bb = (q[None] @ g.v) @ q.T
    da, db = aa.diagonal(dim1=1, dim2=2), bb.diagonal(dim1=1, dim2=2)
    out[(4,)].add_((da * db).sum(0))
    out[(3, 1)].add_((da[:, :, None]*bb + db[:, :, None]*aa).sum(0)/2)
    out[(2, 2)].add_((da.T @ db + db.T @ da)/6)
    # Materialization belongs to the reference-chain probe. A production
    # K3 consumer must use the matrix-pair factors directly.
    for a, b, av, bv in zip(aa, bb, da, db):
        out[(2, 1, 1)].add_(av[:, None, None]*b[None]/6)
        out[(2, 1, 1)].add_(bv[:, None, None]*a[None]/6)


class PairProbe(TreeBirth):
    def __init__(self, mode, rank, seed=111, capture=False):
        super().__init__()
        self.mode, self.rank, self.seed = mode, rank, seed
        self.layer = -1
        self.pending = None
        self.rows, self.reads = [], []
        self.capture, self.captured = capture, []

    def split_birth(self, ds):
        if not any('slice211' in types for types, _ in f.SPEC):
            return
        total = ds.slices[(2, 1, 1)]
        tree = tree_tensor(self.current.trees, ds.n, total.device, total.dtype)
        residual = total-tree
        ds.slices[(2, 1, 1)] = tree
        if self.mode == 'tree':
            return
        # B_ab,r = 6 sym_ab T_r,a,b, symmetric in the last two axes.
        b = 3*(residual + residual.transpose(1, 2))
        if self.capture:
            self.captured.append(b.clone())
        flat = b.reshape(ds.n, -1)
        rank = min(self.rank, ds.n)
        start = time.perf_counter()
        if self.mode == 'exact':
            u = torch.eye(ds.n, device=b.device, dtype=b.dtype)
            v = b
        elif self.mode == 'svd':
            # Diagnostic optimal rank only; its n^4 construction is charged
            # as such, never advertised as a scalable candidate.
            u = torch.linalg.svd(flat, full_matrices=False)[0][:, :rank]
            v = (u.T @ flat).reshape(rank, ds.n, ds.n)
        else:
            rng = torch.Generator(device=b.device).manual_seed(self.seed+1009*self.layer)
            if self.mode == 'random':
                u = (torch.randint(2, (ds.n, rank), generator=rng, device=b.device)*2-1).to(b.dtype)/rank**.5
                v = (u.T @ flat).reshape(rank, ds.n, ds.n)
            elif self.mode == 'rsvd':
                ell = min(ds.n, rank+4)
                omega = torch.randn((ds.n*ds.n, ell), generator=rng, device=b.device, dtype=b.dtype)
                basis = torch.linalg.qr(flat @ omega, mode='reduced')[0]
                core = basis.T @ flat
                small_u = torch.linalg.svd(core, full_matrices=False)[0][:, :rank]
                u = basis @ small_u
                v = (small_u.T @ core).reshape(rank, ds.n, ds.n)
            else:
                raise ValueError(self.mode)
        secs = time.perf_counter()-start
        err = float((flat-u@v.reshape(len(v), -1)).square().sum())
        norm = float(flat.square().sum())
        self.rows.append(dict(layer=self.layer, rank=len(v), seconds=secs,
                              energy=norm, relative_error2=err/max(norm, 1e-300)))
        self.pending = (u, v)

    def install(self):
        super().install()
        self.original_from = self.saved_from
        def make(ds):
            out = self.original_from(ds)
            if self.pending is not None:
                u, v = self.pending
                out.groups.append(PairGroup(u, v, torch.eye(ds.n, device=ds.device, dtype=ds.dtype), self.layer))
                self.pending = None
            return out
        self.saved_from = make
        self.original_read = f.group_dslices
        def read(g, out):
            if isinstance(g, PairGroup):
                start = time.perf_counter()
                pair_slices(g, out)
                self.reads.append(dict(layer=self.layer, born=g.born, rank=len(g.v),
                                       seconds=time.perf_counter()-start))
            else:
                self.original_read(g, out)
        f.group_dslices = read

    def uninstall(self):
        self.saved_from = self.original_from
        super().uninstall()
        f.group_dslices = self.original_read
