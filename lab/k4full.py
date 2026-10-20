"""Full factored K=4 (ARC reference factor_k4, augment=False) at n=1024 on one GPU.

kappa4 = sum over groups of Sym(sum_r A_r x B_r) with A_r = Sym(x_r (x) y_r) and B_r dense. A group keeps its birth
legs (xb, yb; None = identity), its birth B (Bb, n x n x R, on GPU or pinned host memory) and one cumulative
transport Q (n x n): every linear step is Q <- W Q, every wick step Q <- diag(w) Q, so x = Q xb, y = Q yb and
B_r = Q Bb_r Q^T. The reference keeps A dense too (n x n x R per factor, ~256 GB at n=1024 by layer 16);
here A is two n x R legs and only B costs n^3, so the full chain is one n^3 birth per group plus GPU scratch.
Birth groups per layer: the two-leg group (x = w1 * C_off, y = I; B = 12 path + 4 star + 12 three2) and the
slice group of from_dstensor (x = y = I; B = slice4 + 4 slice31 + 3 slice22 + 6 slice211).

AW_SPEC sets the groups: ',' separates groups, '+' joins types in one group, a leading '!' drops that group's crossed
pairings ((2,2) 2/3 term, (2,1,1) 1/3 + 1/3 terms); types left out are dropped whole.
  full  = path+star+three2,slice4+slice31+slice22+slice211   (== reference)
  cheap = star,slice4+slice31,!slice22+slice211              (== lab/tk4.py, CHEAP=slice22,slice211)
usage: k4full.py public <idx list> [device] [f32|f64]      official public MLPs vs mini_truth.npz
       k4full.py kscan <n> <L> <seed> <N> [device] [f32|f64] seeded MLPs vs data/kscan_*.npz MC truth
env: AW_SPEC (default full), AW_STORE host|gpu|gpu:<GB> (births on GPU while allocated memory stays under GB, then
     pinned host; default host), AW_CHUNK (rows per (2,1,1) chunk, default 128),
     AW_MAX_LAYERS (prefix run)
"""
import sys, os, time, math, inspect, textwrap, resource
import numpy as np, torch
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import linear_kprop, coerce_input, Kind
from mlp_kprop.wick import WICK_COEF_D
from mlp_kprop.diagslice import DSTensor, zero_repeated, _einsum_delta
import mlp_kprop.factor_k4 as fk4

TWO_LEG = ("path", "star", "three2")
SLICE = ("slice4", "slice31", "slice22", "slice211")
PRESETS = {"full": "path+star+three2,slice4+slice31+slice22+slice211",
           "cheap": "star,slice4+slice31,!slice22+slice211"}
PARTS = [(4,), (3, 1), (2, 2), (2, 1, 1)]


def parse_spec(spec):
    spec = PRESETS.get(spec, spec)
    groups = []
    for g in spec.split(","):
        crossed = not g.startswith("!")
        types = tuple(g.lstrip("!").split("+"))
        assert all(t in TWO_LEG + SLICE for t in types), g
        assert all(t in TWO_LEG for t in types) or all(t in SLICE for t in types), f"mixed group {g}"
        groups.append((types, crossed))
    return groups


SPEC = parse_spec(os.environ.get("AW_SPEC", "full"))
STORE = os.environ.get("AW_STORE", "host")
CHUNK = int(os.environ.get("AW_CHUNK", "128"))
STATS = {"groups": 0, "birth_bytes": 0, "dslice_s": 0.0}


class Group:
    __slots__ = ("types", "crossed", "xb", "yb", "Bb", "Q")

    def __init__(self, types, crossed, xb, yb, Bb, Q):
        self.types, self.crossed, self.xb, self.yb, self.Bb, self.Q = types, crossed, xb, yb, Bb, Q

    def moved(self, Q):
        return Group(self.types, self.crossed, self.xb, self.yb, self.Bb, Q)


LAST_LAYER = False   # births of the last layer never propagate: leave them on the GPU, freed with the layer


def _store(B):
    if LAST_LAYER or B.device.type == "cpu":
        return B
    STATS["groups"] += 1
    STATS["birth_bytes"] += B.numel() * B.element_size()
    if STORE == "gpu" or (STORE.startswith("gpu:") and torch.cuda.memory_allocated() + 3 * B.numel() * B.element_size()
                           < float(STORE[4:]) * 2**30):
        return B
    out = torch.empty(B.shape, dtype=B.dtype, pin_memory=True)
    out.copy_(B)
    return out


def group_dslices(g, out):
    """Add this group's four dslices (no zero_repeated yet) into out[part]."""
    t0 = time.time()
    Q = g.Q
    n = Q.shape[0]
    X = Q if g.xb is None else Q @ g.xb
    Y = Q if g.yb is None else Q @ g.yb
    Bb = g.Bb.to(Q.device, non_blocking=True)
    R = Bb.shape[2]
    M = (Q @ Bb.reshape(n, n * R)).reshape(n, n, R)        # [a, l, r]
    del Bb
    B = torch.matmul(Q, M)                                  # [a, b, r] = sum_kl Q_ak Q_bl Bb_klr
    del M
    dA = X * Y                                              # A_r[i, i]
    dB = torch.diagonal(B, dim1=0, dim2=1).T.contiguous()  # B_r[i, i]
    out[(4,)] += (dA * dB).sum(1)
    out[(3, 1)] += 0.5 * torch.matmul(B, dA.unsqueeze(2)).squeeze(2) + 0.25 * ((X * dB) @ Y.T + (Y * dB) @ X.T)
    out[(2, 2)] += (dA @ dB.T + dB @ dA.T) / 6
    Bflat = B.reshape(n * n, R)
    o211 = out[(2, 1, 1)]
    for i0 in range(0, n, CHUNK):
        i1 = min(i0 + CHUNK, n)
        Bi = B[i0:i1]                                       # (c, n, R)
        # same-side: 1/6 sum_r A_r[i,i] B_r[j,k] + 1/6 sum_r A_r[j,k] B_r[i,i]
        S1 = (Bflat @ dA[i0:i1].T).reshape(n, n, i1 - i0).permute(2, 0, 1)
        U = torch.matmul(X.unsqueeze(0) * dB[i0:i1].unsqueeze(1), Y.T)   # sum_r dB_ir x_jr y_kr
        blk = S1 / 6 + (U + U.transpose(1, 2)) / 12
        del S1, U
        if g.crossed:
            # 1/3 sum_r A_r[i,j] B_r[i,k] + (j<->k), A_r[i,j] = (x_ir y_jr + y_ir x_jr) / 2
            T = torch.matmul(Y, (Bi * X[i0:i1].unsqueeze(1)).transpose(1, 2))
            T += torch.matmul(X, (Bi * Y[i0:i1].unsqueeze(1)).transpose(1, 2))
            blk += (T + T.transpose(1, 2)) / 6
            # (2,2) crossed: 2/3 sum_r A_r[i,j] B_r[i,j] = 1/3 (E + E^T), E[i,j] = sum_r x_ir y_jr B_r[i,j]
            E = torch.matmul(Bi * Y.unsqueeze(0), X[i0:i1].unsqueeze(2)).squeeze(2)   # (c, n)
            out[(2, 2)][i0:i1] += E / 3
            out[(2, 2)][:, i0:i1] += E.T / 3
            del T, E
        o211[i0:i1] += blk
        del blk
    STATS["dslice_s"] += time.time() - t0


class SK4:
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
        return SK4(self.n, [g.moved(g.Q.clone()) for g in self.groups], self.repeated.clone(), self.device, self.dtype)

    def contract_W(self, W):
        return SK4(self.n, [g.moved(W @ g.Q) for g in self.groups], None, self.device, self.dtype)

    def contract_wick_(self, wick):
        for part in list(self.repeated.slices):
            sl = self.repeated.slices[part]
            for ax, p in enumerate(part):
                shape = [1] * len(part); shape[ax] = -1
                sl = sl * (wick ** p).reshape(shape)
            self.repeated.slices[part] = sl
        for g in self.groups:
            g.Q = wick[:, None] * g.Q

    def _dslices(self, groups):
        n, kw = self.n, dict(device=self.device, dtype=self.dtype)
        out = {p: torch.zeros((n,) * len(p), **kw) for p in PARTS}
        for g in groups:
            group_dslices(g, out)
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
        return SK4(self.n, self.groups + other.groups, None, self.device, self.dtype)

    @staticmethod
    def from_dstensor(ds):
        n, kw = ds.n, dict(device=ds.device, dtype=ds.dtype)
        Q = torch.eye(n, **kw)
        groups = []
        for types, crossed in SPEC:
            if types[0] not in SLICE:
                continue
            B = torch.zeros((n, n, n), **kw)
            if "slice4" in types:
                B += _einsum_delta(ds.get_slice((4,)), 'i -> i i i')
            if "slice31" in types:
                B += 4 * _einsum_delta(ds.get_slice((3, 1)), 'i j -> j i i')
            if "slice22" in types:
                B += 3 * _einsum_delta(ds.get_slice((2, 2)), 'i j -> j j i')
            if "slice211" in types:
                B += 6 * ds.get_slice((2, 1, 1)).permute(1, 2, 0)
            B = (B + B.transpose(0, 1)) / 2
            groups.append(Group(types, crossed, None, None, _store(B), Q.clone()))
            del B
        return SK4(n, groups, None, ds.device, ds.dtype)


def two_leg_buffers(n, like):
    """One n^3 buffer per two-leg group of SPEC; returns ({type: buffer or None}, [(types, crossed, buffer)])."""
    by_type, groups = {t: None for t in TWO_LEG}, []
    for types, crossed in SPEC:
        if types[0] not in TWO_LEG:
            continue
        buf = torch.zeros((n, n, n), device=like.device, dtype=like.dtype)
        for t in types:
            by_type[t] = buf.permute(2, 0, 1)
        groups.append((types, crossed, buf))
    return by_type, groups


def add_two_leg(pK, X, groups):
    n = pK.n
    Q = torch.eye(n, device=X.device, dtype=X.dtype)
    new = []
    for types, crossed, buf in groups:
        B = (buf + buf.transpose(0, 1)) / 2
        new.append(Group(types, crossed, X.clone(), None, _store(B), Q.clone()))
        del B
    groups.clear()
    pK.add_groups_(new)


# --- the reference nonlinear step, with kappa4 held as SK4 groups
src = inspect.getsource(fk4.factored_nonlin_kprop_k4)
reps = [
    ("            A = _einsum_delta(A_s * w(A_i_inc)[:, None], 'i j -> i j j')\n"
     "            B = torch.zeros((n, n, n), device=mean.device, dtype=mean.dtype)\n"
     "            B_jkl = B.permute(2, 0, 1)\n",
     "            A = A_s * w(A_i_inc)[:, None]\n"
     "            _TB, _TG = two_leg_buffers(n, mean)\n"),
    ("                    B_jkl.add_(tmp, alpha=12)  # number of path graphs",
     "                    if _TB['path'] is not None: _TB['path'].add_(tmp, alpha=12)"),
    ("                    B_jkl.add_(tmp, alpha=4)  # number of star graphs",
     "                    if _TB['star'] is not None: _TB['star'].add_(tmp, alpha=4)"),
    ("                B_jkl.add_(tmp, alpha=12)  # number of 3+2 graphs",
     "                if _TB['three2'] is not None: _TB['three2'].add_(tmp, alpha=12)"),
    ("            pK_1111.add_factors_((A, B))\n\n        # WK2111",
     "            add_two_leg(pK_1111, A, _TG)\n            del _TB\n\n        # WK2111"),
]
for old, new in reps:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
src = textwrap.dedent(src)
ns = dict(fk4.__dict__)
ns.update(FactoredTensor4=SK4, two_leg_buffers=two_leg_buffers, add_two_leg=add_two_leg)
exec(compile(src, "k4full_nonlin", "exec"), ns)
nonlin = ns["factored_nonlin_kprop_k4"]


def run_chain(W, dev, dt, layers=None):
    """W: list of (n, n) float32 arrays in x @ W convention. Returns per-layer means (L, n) float64."""
    n, L = W[0].shape[0], len(W)
    mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=dt)
    for i in range(L):
        mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(device=dev, dtype=dt).contiguous()
    K = coerce_input({1: torch.zeros(n, device=dev, dtype=dt), 2: torch.eye(n, device=dev, dtype=dt)}, k_max=4, kind=Kind.SIMPLE)
    wick = WICK_COEF_D["relu"]
    means = []
    global LAST_LAYER
    for l in range(layers or L):
        LAST_LAYER = l == (layers or L) - 1
        K = linear_kprop(K, mlp.Ws[l].weight, k_max=4, set_metric=mlp.init_scale[l])
        K = nonlin(K, nonlin_wick_coef=wick)
        means.append(K[1].to_tensor().double().cpu().numpy())
    return np.stack(means)


def _mem():
    gpu = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss / 2**30 if sys.platform == "darwin" else rss / 2**20
    return f"gpu_peak={gpu:.1f}G host_maxrss={rss:.1f}G births={STATS['birth_bytes'] / 2**30:.0f}G"


def main():
    mode = sys.argv[1]
    spec_name = os.environ.get("AW_SPEC", "full")
    if mode == "public":
        from mlpdata import weights_from_seed, load_truth
        idx = [int(x) for x in sys.argv[2].split(",")]
        dev = sys.argv[3] if len(sys.argv) > 3 else "cuda"
        dt = torch.float64 if (len(sys.argv) > 4 and sys.argv[4] == "f64") else torch.float32
        t = load_truth("mini")
        for i in idx:
            W = weights_from_seed(t["mlp_seed"][i])
            gt = t["all_layer_means"][i].astype(np.float64)
            STATS.update(groups=0, birth_bytes=0, dslice_s=0.0)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            pred = run_chain(W, dev, dt, int(os.environ.get("AW_MAX_LAYERS", len(W))))
            per = np.mean((pred - gt[:len(pred)]) ** 2, axis=1)
            status = "final" if len(pred) == len(W) else f"prefix{len(pred)}"
            print(f"k4full spec={spec_name} mlp {i} {dev} {dt}: {status} {per[-1]:.4e} all {per.mean():.3e} "
                  f"t={time.time() - t0:.0f}s dslice={STATS['dslice_s']:.0f}s {_mem()} | " +
                  " ".join(f"{x:.2e}" for x in per), flush=True)
            if os.environ.get("AW_SAVE"):
                np.save(f"{os.environ['AW_SAVE']}_mlp{i}.npy", pred)
    elif mode == "kscan":
        n, L, seed, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(float(sys.argv[5]))
        dev = sys.argv[6] if len(sys.argv) > 6 else "cpu"
        dt = torch.float32 if (len(sys.argv) > 7 and sys.argv[7] == "f32") else torch.float64
        rng = np.random.default_rng(seed)
        W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
        z = np.load(os.environ.get("AW_DATA", "data") + f"/kscan_{n}_{L}_{seed}_{N}.npz")
        t0 = time.time()
        pred = run_chain(W, dev, dt, int(os.environ.get("AW_MAX_LAYERS", L)))
        per = np.mean((pred - z["gt"][:len(pred)]) ** 2, axis=1)
        print(f"k4full spec={spec_name} n={n} L={len(pred)}/{L} seed={seed} {dev} {dt}: final {per[-1]:.4e} "
              f"t={time.time() - t0:.0f}s {_mem()} | " + " ".join(f"{x:.1e}" for x in per[::3]), flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
