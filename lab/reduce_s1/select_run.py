"""reduce_s1: chain B to layer s0, then the research K=4 suffix carrying a chosen set of states.

A plan names, per family and birth layer b, how many later layers read the state (its life A:
reads at b+1..b+A; A >= 14-b lives to the end, as in Table 7a). Families:
  star     star births (two-leg group)
  ck3      C.k3 births (two-leg group, 'three2')
  tree     slice births (4,), (3,1), (2,2) and the tree-type graph terms of (2,1,1)
  nontree  the non-tree remainder of the (2,1,1) birth, held as its own group
Splitting the (2,1,1) birth into tree and non-tree groups is exact (the group is linear in its
birth kernel), so a plan with every state at full life is the Table 7a s0 run.
The B seed (chain B's (4,), (3,1), (2,2) slices and radial feed at the switch) lives in every plan.

--instrument records, for every live state at every read, its first-order effect on the final
mean: the (4,) slice through the mean, the (3,1), (2,2) slices through the next layer's
variance and the (2,1,1) slice through the next layer's kappa3 diagonal, each carried to the output by the chain's own slopes. Its norm is the read weight.
No truth enters the recurrence, the read weights or the plan; truth is read to score.

usage: select_run.py --plan plan.json --snapshots DIR --output out.json [--instrument]
       select_run.py --plan plan.json --synthetic 48 --output out.json   (CPU float64 check)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('AW_SPEC', 'star,three2,slice4+slice31,!slice22+slice211')
ROOT = Path(__file__).resolve().parents[2]
for sub in ('tools', 'refs/mlp_cumulant_propagation/src', 'lab', 'lab/cand_k4x'):
    sys.path.insert(0, str(ROOT / sub))
import numpy as np
import torch
import b_suffix as bs
from k4n3_young_direct_chain import Recorder, tree_tensor
from mlp_kprop.diagslice import _zero_repeated
f = bs.f
d = bs.d
P211 = (2, 1, 1)
FAMILIES = ('star', 'ck3', 'tree', 'nontree')


class Plan:
    def __init__(self, spec):
        self.spec = spec
        self.start = int(spec['start'])
        self.life = {fam: {int(b): int(a) for b, a in spec.get('life', {}).get(fam, {}).items()}
                     for fam in FAMILIES}
        self.last_birth = int(spec.get('last_birth', 13))

    def born(self, fam, b):
        return b <= self.last_birth and b in self.life[fam]

    def expired(self, fam, b, layer):
        if fam == 'seed':
            return False
        life = self.life[fam][b]
        return life < 14 - b and layer - b > life

    def layer_spec(self, layer):
        if layer >= 15:
            return []
        parts = [p for fam, p in (('star', 'star'), ('ck3', 'three2')) if self.born(fam, layer)]
        if self.born('tree', layer) or self.born('nontree', layer):
            parts += ['slice4+slice31', '!slice22+slice211']
        return f.parse_spec(','.join(parts)) if parts else []


class Split(Recorder):
    """Tree / non-tree split of every newborn (2,1,1) slice; keeps what the plan keeps."""

    def __init__(self, plan, registry):
        super().__init__(light=True, birth_mode='split')
        self.plan, self.registry = plan, registry
        self.layer, self.pending, self.residual = None, None, set()
        self.norms = {}

    def split_birth(self, ds):
        self.pending = None
        if not (self.plan.born('tree', self.layer) or self.plan.born('nontree', self.layer)):
            return
        total = ds.slices[P211]
        tree = tree_tensor(self.current.trees, ds.n, total.device, total.dtype)
        self.pending = total - tree
        self.norms[self.layer] = dict(total=float(total.norm()), tree=float(tree.norm()),
                                      non_tree=float(self.pending.norm()))
        ds.slices[P211] = tree

    def install(self):
        super().install()
        inner = f.SK4.from_dstensor
        rec = self

        def from_dstensor(ds):
            out = inner(ds)
            layer, n = rec.layer, ds.n
            groups = list(out.groups) if rec.plan.born('tree', layer) else []
            if rec.pending is not None and rec.plan.born('nontree', layer):
                B = 6 * rec.pending.permute(1, 2, 0)
                B = (B + B.transpose(0, 1)) / 2
                g = f.Group(('slice211',), False, None, None, f._store(B),
                            torch.eye(n, device=ds.device, dtype=ds.dtype))
                rec.residual.add(id(g.Bb))
                groups.append(g)
                del B
            rec.pending = None
            out.groups = groups
            return out

        f.SK4.from_dstensor = staticmethod(from_dstensor)


class Reads:
    """Per state and read layer: (4,) mean shift and next-layer variance shift, before transport."""

    def __init__(self, registry, weights):
        self.registry, self.weights = registry, weights
        self.layer, self.mean, self.var = None, None, None
        self.seen, self.rows = set(), []
        self.original = f.group_dslices
        self.with211 = True

    def coefs(self):
        s = self.var.clamp_min(1e-30).sqrt()
        a = self.mean / s
        phi = torch.exp(-a * a / 2) / math.sqrt(2 * math.pi)
        return dict(w1=torch.special.ndtr(a), w2=phi / s, w3=-a * phi / s ** 2,
                    c4=(a * a - 1) * phi / (24 * s ** 3))

    def wrap(self, g, out):
        key = self.registry.get(id(g.Bb))
        tag = (id(g.Bb), self.layer)
        if key is None or tag in self.seen:
            return self.original(g, out)
        self.seen.add(tag)
        before = {p: out[p].clone() for p in ((4,), (3, 1), (2, 2))}
        b211 = out[P211].clone() if self.with211 and self.layer < 15 else None
        self.original(g, out)
        c4 = out[(4,)] - before[(4,)]
        c31 = out[(3, 1)] - before[(3, 1)]
        c22 = out[(2, 2)] - before[(2, 2)]
        c31.fill_diagonal_(0)
        c22.fill_diagonal_(0)
        k = self.coefs()
        dC = k['w3'][:, None] * c31 * k['w1'][None, :]
        dC = (dC + dC.T) / 6 + k['w2'][:, None] * c22 * k['w2'][None, :] / 4
        dC.fill_diagonal_(0)
        row = dict(family=key[0], birth=key[1], layer=self.layer,
                   m4=(k['c4'] * c4).double().cpu().numpy(),
                   e4=float(c4.norm()), e31=float(c31.norm()), e22=float(c22.norm()))
        if self.layer < 15:
            w = torch.as_tensor(self.weights[self.layer + 1].T.copy(), device=dC.device, dtype=dC.dtype)
            row['dvar'] = ((w @ dC) * w).sum(1).double().cpu().numpy()
            if b211 is not None:
                # (2,1,1) -> next-layer kappa3 diagonal: (3/2) sum_ijk (W_mi w2_i)(W_mj w1_j)(W_mk w1_k) kappa_iijk
                c211 = b211.neg_().add_(out[P211])
                _zero_repeated(c211)
                A, V = w * k['w2'][None, :], w * k['w1'][None, :]
                Z = torch.empty((n := c211.shape[0], n), device=w.device, dtype=w.dtype)
                for i0 in range(0, n, 128):
                    Y = (c211[i0:i0 + 128].reshape(-1, n) @ V.T).reshape(-1, n, n)   # [i, j, m]
                    Z[i0:i0 + 128] = (Y * V.T[None]).sum(1)                          # [i, m]
                    del Y
                row['d3'] = (1.5 * (A * Z.T).sum(1)).double().cpu().numpy()
                row['e211'] = float(c211.norm())
                del c211, Z, A, V
        self.rows.append(row)
        del before, c4, c31, c22, dC

    def finish(self, moments):
        """moments: {layer: (mean, var)} preactivation of the run. Returns rows with read weights."""
        W = {k: np.asarray(self.weights[k], dtype=np.float64).T for k in moments}
        moments = {k: (m, np.maximum(v, 1e-30)) for k, (m, v) in moments.items()}
        slope = {k: 0.5 * (1 + np.vectorize(math.erf)(m / np.sqrt(2 * v))) for k, (m, v) in moments.items()}
        vcoef = {k: np.exp(-m * m / (2 * v)) / np.sqrt(2 * np.pi) / (2 * np.sqrt(v)) for k, (m, v) in moments.items()}
        # kappa3 diagonal -> mean: E f'''/3! = -alpha phi(alpha) / (6 sigma^2)
        k3coef = {k: -(m / np.sqrt(v)) * np.exp(-m * m / (2 * v)) / np.sqrt(2 * np.pi) / (6 * v) for k, (m, v) in moments.items()}

        def carry(x, frm):
            for k in range(frm + 1, 16):
                x = slope[k] * (W[k] @ x)
            return x

        U = []
        for r in self.rows:
            u4 = carry(r.pop('m4'), r['layer'])
            uc = carry(vcoef[r['layer'] + 1] * r.pop('dvar'), r['layer'] + 1) if 'dvar' in r else np.zeros_like(u4)
            u3 = carry(k3coef[r['layer'] + 1] * r.pop('d3'), r['layer'] + 1) if 'd3' in r else np.zeros_like(u4)
            r.update(w=float(np.linalg.norm(u4 + uc + u3)), w4=float(np.linalg.norm(u4)), wC=float(np.linalg.norm(uc)),
                     w3=float(np.linalg.norm(u3)), w_meanvar=float(np.linalg.norm(u4 + uc)))
            U.append(u4 + uc + u3)
        return self.rows, np.array(U, dtype=np.float32)


def family_of(g, residual):
    if id(g.Bb) in residual:
        return 'nontree'
    return {'star': 'star', 'three2': 'ck3'}.get(g.types[0], 'tree')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--snapshots', type=Path)
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--instrument', action='store_true')
    p.add_argument('--synthetic', type=int, default=0, help='CPU float64 width for the mechanics check')
    p.add_argument('--unsplit', action='store_true', help='synthetic only: plain recurrence, no split (parity)')
    p.add_argument('--tree-only-reference', action='store_true', help='synthetic only: TreeBirth reference')
    a = p.parse_args()
    spec = json.loads(a.plan.read_text())
    plan = Plan(spec)
    s0 = plan.start
    torch.set_num_threads(4)
    t0 = time.monotonic()
    registry, moments = {}, {}
    if a.synthetic:
        n, dev, dtype = a.synthetic, 'cpu', torch.float64
        rng = np.random.default_rng(7)
        weights = [(rng.standard_normal((n, n)) * (2 / n) ** .5) for _ in range(16)]
        truth, target = None, None
        K = f.coerce_input({1: torch.zeros(n, dtype=dtype), 2: torch.eye(n, dtype=dtype)}, k_max=4, kind=f.Kind.SIMPLE)
        means = []
        f.SPEC = []
        for layer in range(s0):
            K = f.linear_kprop(K, torch.as_tensor(weights[layer].T.copy(), dtype=dtype), k_max=4, set_metric=2.)
            K = f.nonlin(K, nonlin_wick_coef=f.WICK_COEF_D['relu'])
            means.append(K[1].to_tensor().double().numpy())
        diag = {}
    else:
        n, dev, dtype = 1024, torch.device('cuda', torch.cuda.current_device()), torch.float32
        truth = d.load_truth('mini')
        weights = d.weights_from_seed(truth['mlp_seed'][a.index])
        target = truth['all_layer_means'][a.index].astype(float)
        snapshot = np.load(a.snapshots / f'B_mini{a.index}_pre{s0}.npz')
        mu, C, T, diag = bs.expand(snapshot)
        assert float(C.diag().min()) > 0
        K = f.coerce_input({1: mu, 2: C, 3: T}, k_max=4, kind=f.Kind.SIMPLE)
        del mu, C, T
        inv = torch.as_tensor(snapshot['inv_perm'].astype('int64'), device=dev)

        def unsort(name):
            v = torch.as_tensor(snapshot[name], device=dev, dtype=dtype)
            for axis in range(v.ndim):
                v = v.index_select(axis, inv)
            return v

        Cs = unsort('C')
        Cs.fill_diagonal_(0)
        G = Cs * float(snapshot['lam_feed'])
        d211 = G.unsqueeze(0).expand(n, -1, -1).clone() / 3
        ds = f.DSTensor(d=4, n=n, slices={(4,): unsort('g4row'), (3, 1): unsort('wk431').T,
                                          (2, 2): unsort('wk4m'), (2, 1, 1): d211},
                        autozero=True, device=dev, dtype=dtype)
        f.SPEC = f.parse_spec('slice4+slice31+slice22+slice211')
        K[4] = f.SK4.from_dstensor(ds)
        for g in K[4].groups:
            registry[id(g.Bb)] = ('seed', s0)
        del ds, d211, G, Cs
        means = list(snapshot['prefix'])
    split = None
    if a.tree_only_reference:
        from tree_birth import TreeBirth
        split = TreeBirth(0)
        split.install()
    elif not a.unsplit:
        split = Split(plan, registry)
        split.install()
    reads = None
    if a.instrument:
        reads = Reads(registry, weights)
        f.group_dslices = reads.wrap
    layers = []
    for layer in range(s0, 16):
        f.LAST_LAYER = layer == 15
        kept = [g for g in K[4].groups if not plan.expired(*registry.get(id(g.Bb), ('seed', s0)), layer)]
        dropped = len(K[4].groups) - len(kept)
        if dropped:
            K[4].groups = kept
            K[4].clear_repeated()
        if layer > s0 or a.synthetic:
            w = torch.as_tensor(weights[layer].T.copy(), device=dev, dtype=dtype)
            K = f.linear_kprop(K, w, k_max=4, set_metric=2.)
        f.SPEC = plan.layer_spec(layer)
        mean, var = K[1].core, K[2].core.diag()
        moments[layer] = (mean.double().cpu().numpy(), var.double().cpu().numpy())
        if reads is not None:
            reads.layer, reads.mean, reads.var = layer, mean, var
        if split is not None:
            split.layer = layer
            split.start_layer()
        t1 = time.monotonic()
        K = f.nonlin(K, nonlin_wick_coef=f.WICK_COEF_D['relu'])
        residual = split.residual if isinstance(split, Split) else set()
        for g in K[4].groups:
            if id(g.Bb) not in registry:
                registry[id(g.Bb)] = (family_of(g, residual), layer)
        live = {}
        for g in K[4].groups:
            fam = registry[id(g.Bb)][0]
            live[fam] = live.get(fam, 0) + 1
        means.append(K[1].to_tensor().double().cpu().numpy())
        row = dict(layer=layer, seconds=time.monotonic() - t1, memory=f._mem(), dropped=dropped, live=live,
                   split=split.norms.get(layer) if isinstance(split, Split) else None)
        if target is not None:
            row['mse'] = float(np.mean((means[-1] - target[layer]) ** 2))
        layers.append(row)
        print(json.dumps(row), flush=True)
    pred = np.stack(means)
    result = dict(plan=spec, index=a.index, start=s0, synthetic=a.synthetic, bridge=diag,
                  seconds=time.monotonic() - t0, memory=f._mem(), layers=layers)
    if target is not None:
        result['mse'] = float(np.mean((pred[-1] - target[-1]) ** 2))
        result['per_layer_mse'] = np.mean((pred - target) ** 2, axis=1).tolist()
    import ledger
    result['units'], result['unit_rows'] = ledger.units(spec.get('life', {}), s0)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output.with_suffix('.npy'), pred)
    if reads is not None:
        rows, U = reads.finish(moments)
        np.save(a.output.with_suffix('.u.npy'), U)
        result['reads'] = rows
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT', json.dumps({k: v for k, v in result.items() if k not in ('layers', 'reads')}), flush=True)


if __name__ == '__main__':
    main()
