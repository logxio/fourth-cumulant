"""Small-width checks of compact.py against the torch research chain (k4full + r4 projection).

usage: verify_small.py [n] [layers] [s0]
  1. every group read == slices of the dense Sym_4 tensor == k4full.group_dslices
  2. age-0 fast path == group_dslices at Q = I
  3. slice groups (struct legs + full-rank pairs) == k4full slice groups after transport and wick
  4. merged spec star+three2 == separate star,three2 (reference engine)
  5. whole late-start suffix, compact at full ranks == reference without projection
  6. whole suffix with the r4 projection (core rank k < n): compact merged core == reference projected Q
"""
import sys, os, math, copy, itertools
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compact as cm
kf = cm.kf
from mlp_kprop.diagslice import DSTensor, zero_repeated

torch.set_grad_enabled(False)
DT = torch.float64
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
L = int(sys.argv[2]) if len(sys.argv) > 2 else 8
S0 = int(sys.argv[3]) if len(sys.argv) > 3 else 2
g = torch.Generator().manual_seed(0)
rnd = lambda *s: torch.randn(*s, generator=g, dtype=DT)
fails = []


def check(name, a, b, tol=1e-10):
    if isinstance(a, dict):
        errs = {p: float((a[p] - b[p]).abs().max() / max(float(b[p].abs().max()), 1e-300)) for p in a}
        err = max(errs.values())
    else:
        err = float((a - b).abs().max() / max(float(b.abs().max()), 1e-300))
        errs = err
    ok = err < tol
    print(("PASS" if ok else "FAIL"), name, f"{err:.2e}", "" if ok else errs, flush=True)
    if not ok:
        fails.append(name)


def zr(out):
    return {p: zero_repeated(t.clone()) for p, t in out.items()}


def empty():
    return {p: torch.zeros((n,) * len(p), dtype=DT) for p in kf.PARTS}


def dense_sym_slices(Ar, Br):
    T = torch.einsum('rab,rcd->abcd', Ar, Br)
    S = sum(T.permute(*p) for p in itertools.permutations(range(4))) / 24
    i = torch.arange(n)
    a, b = i[:, None], i[None, :]
    return zr({(4,): S[i, i, i, i], (3, 1): S[i, i, i, :], (2, 2): S[a, a, b, b], (2, 1, 1): S[i, i]})


# ---- 1. group reads
for crossed in (True, False):
    xb, Q = rnd(n, n), rnd(n, n)
    Bb = rnd(n, n, n); Bb = (Bb + Bb.transpose(0, 1)) / 2
    grp = kf.Group(("star", "three2"), crossed, xb, None, Bb, Q)
    o1 = empty(); kf.group_dslices(grp, o1); o1 = zr(o1)
    X, Y = Q @ xb, Q
    Ar = 0.5 * (torch.einsum('ar,br->rab', X, Y) + torch.einsum('ar,br->rab', Y, X))
    Br = torch.einsum('ak,klr,bl->rab', Q, Bb, Q)
    if crossed:
        o2 = dense_sym_slices(Ar, Br)
        check("1 group_dslices crossed == Sym4 slices", o1, o2)
        core = cm.Core(torch.eye(n, dtype=DT), cm.core_from_dense(cm.Dense(grp.types, True, xb, None, Bb, Q, 0, False), Q), None, [])
        o3 = empty(); core.add_dslices(o3); o3 = zr(o3)
        check("1 merged core read == group_dslices", o3, o1)
    else:
        print("info 1 non-crossed group (same-side only) not a Sym4 slice; checked via engines below")

# ---- 2. age-0 fast path
for diag0 in (True, False):
    xb = rnd(n, n)
    if diag0:
        xb.fill_diagonal_(0)
    Bb = rnd(n, n, n); Bb = (Bb + Bb.transpose(0, 1)) / 2
    for crossed in (True, False):
        d = cm.Dense(("star",), crossed, xb, None, Bb, torch.eye(n, dtype=DT), 0, True)
        o1 = empty(); d.add_dslices(o1)
        o2 = empty(); kf.group_dslices(kf.Group(("star",), crossed, xb, None, Bb, torch.eye(n, dtype=DT)), o2)
        check(f"2 age0 fast path crossed={crossed} zero_diag={diag0}", zr(o1), zr(o2))


# ---- 3. slice groups
def rand_ds(scale=1.0):
    s4 = rnd(n) * scale
    s31 = rnd(n, n) * scale
    s22 = rnd(n, n) * scale; s22 = (s22 + s22.T) / 2
    s211 = rnd(n, n, n) * scale; s211 = (s211 + s211.transpose(1, 2)) / 2
    return DSTensor(d=4, n=n, slices={(4,): s4, (3, 1): s31, (2, 2): s22, (2, 1, 1): s211}, autozero=True, dtype=DT)


cm.CFG.update(pair_rank=n, pair_over=0, pair_iters=1)
for spec in ("slice4+slice31+slice22+slice211", "slice4+slice31,!slice22+slice211"):
    ds = rand_ds()
    kf.SPEC = kf.parse_spec(spec)
    ref = kf.SK4.from_dstensor(ds)
    cm.BIRTH.update(layer=0, start=rnd(n, n))
    cmp = cm.CSK4.from_dstensor(ds)
    for step in range(2):
        W = rnd(n, n) / math.sqrt(n)
        w = rnd(n).abs() + 0.5
        ref = ref.contract_W(W); ref.contract_wick_(w)
        cmp = cmp.contract_W(W); cmp.contract_wick_(w)
    a = cmp._dslices(cmp.groups); b = ref._dslices(ref.groups)
    check(f"3 slice groups spec={spec}", a, b)


# ---- 4-6. engines over a whole late-start suffix
def weights():
    gg = torch.Generator().manual_seed(7)
    return [(torch.randn(n, n, generator=gg, dtype=DT) * math.sqrt(2.0 / n)) for _ in range(L)]


WS = weights()


def prefix_state():
    kf.SPEC = []
    K = kf.coerce_input({1: torch.zeros(n, dtype=DT), 2: torch.eye(n, dtype=DT)}, k_max=4, kind=kf.Kind.SIMPLE)
    for l in range(S0):
        K = kf.linear_kprop(K, WS[l], k_max=4, set_metric=2.0)
        K = kf.nonlin(K, nonlin_wick_coef=kf.WICK_COEF_D['relu'])
    K = kf.linear_kprop(K, WS[S0], k_max=4, set_metric=2.0)
    return K


SEED = rand_ds(float(os.environ.get("SEED_SCALE", "1e-2")))


def run(engine, spec, rank=None, m0=2, fb_rank=None, m0f=1):
    fp = None
    K = prefix_state()
    K = {d: K[d] for d in K if d != 4}
    kf.SPEC = kf.parse_spec("slice4+slice31+slice22+slice211")
    if engine == "ref":
        K[4] = kf.SK4.from_dstensor(SEED)
        births = {id(gr.Bb): S0 for gr in K[4].groups}
    else:
        cm.BIRTH.update(layer=S0, start=start_vectors(K))
        K[4] = cm.CSK4.from_dstensor(SEED)
    means = []
    if fb_rank is not None:
        if engine == "ref":
            fp = cm.FeedbackProjection(kf.SK4, fb_rank, m0f, lambda gr: births.get(id(gr.Bb), fp.layer))
        else:
            fp = cm.FeedbackProjection(cm.CSK4, fb_rank, m0f, lambda gr: gr.birth)
        fp.install()
    try:
        means = _loop(engine, spec, rank, m0, fp, K, means, births if engine == "ref" else None)
    finally:
        if fp is not None:
            fp.uninstall()
    return torch.stack(means)


def _loop(engine, spec, rank, m0, fp, K, means, births):
    for layer in range(S0, L):
        weight = WS[layer]
        if layer > S0:
            K = kf.linear_kprop(K, weight, k_max=4, set_metric=2.0)
        if fp is not None:
            fp.start_layer(layer, weight, K, kf.WICK_COEF_D['relu'])
        kf.SPEC = [] if layer == L - 1 else kf.parse_spec(spec)
        if engine == "ref":
            K = kf.nonlin(K, nonlin_wick_coef=kf.WICK_COEF_D['relu'])
            births.update({id(gr.Bb): births.get(id(gr.Bb), layer) for gr in K[4].groups})
            if rank is not None:
                old = [gr for gr in K[4].groups if gr.types[0] in ('star', 'three2') and layer - births[id(gr.Bb)] > m0]
                if old:
                    unique = {births[id(gr.Bb)]: gr.Q for gr in old}
                    sk = weight[:, :rank].contiguous()
                    for _ in range(2):
                        im = sum((q @ (q.T @ sk) for q in unique.values()), torch.zeros_like(sk))
                        sk, _ = torch.linalg.qr(im, mode='reduced')
                    for gr in old:
                        gr.Q = sk @ (sk.T @ gr.Q)
                    K[4].clear_repeated()
        else:
            cm.BIRTH.update(layer=layer, start=start_vectors(K))
            K = cm.nonlin(K, nonlin_wick_coef=kf.WICK_COEF_D['relu'])
            if rank is not None:
                cm.CFG.update(core_rank=rank, m0=m0)
                cm.project_old(K[4], layer, weight, None)
        means.append(K[1].to_tensor().clone())
    return means


def start_vectors(K):
    mu = K[1].core
    v = mu / mu.norm()
    return torch.cat([v[:, None], WS[0][:, : n - 1]], 1)


kf.CHUNK = 4
cm.CHUNK = 5
a = run("ref", "star,three2,slice4+slice31,!slice22+slice211")
b = run("ref", "star+three2,slice4+slice31,!slice22+slice211")
check("4 reference: star+three2 merged == separate", b, a)
c = run("cmp", "star+three2,slice4+slice31,!slice22+slice211")
check("5 compact full rank == reference (no projection)", c, a, tol=1e-8)
k = max(2, n // 3)
d = run("ref", "star+three2,slice4+slice31,!slice22+slice211", rank=k)
e = run("cmp", "star+three2,slice4+slice31,!slice22+slice211", rank=k)
check(f"6 merged core k={k} == reference r4 projection", e, d, tol=1e-8)
print("projection effect on means (ref k vs ref exact):", float((d - a).abs().max()))
cm.CFG.update(pair_rank=2, pair_over=2, pair_iters=2)
f = run("cmp", "star+three2,slice4+slice31,!slice22+slice211", rank=k)
print("pair rank 2 vs exact pairs, max |dmean|:", float((f - e).abs().max()), " scale", float(e.abs().max()))
cm.CFG.update(slice_mode="dense")
kr = max(2, n // 2)
h = run("ref", "star,three2,slice4+slice31,!slice22+slice211", rank=k, fb_rank=kr)
i_ = run("cmp", "star+three2,slice4+slice31,!slice22+slice211", rank=k, fb_rank=kr)
check(f"7 dense slices + merged core k={k} + feedback k3={kr} == r4 config (reference)", i_, h, tol=1e-8)
print("feedback projection effect (ref):", float((h - d).abs().max()))
print("FAILS:", fails if fails else "none")
