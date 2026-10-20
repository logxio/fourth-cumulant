"""Where the kappa4 per-neuron residual pays at the last layer (lean_k3_aug, k4_aug, 1024 wide, mini 0-3).
Cloud only; reuses var_subspace.py dumps in <out>: chain<i>.npz (init) and mc<i>.npz (mc, 2e7 samples, two halves).
r4 = MC kappa4_iiii minus the per-layer affine fit of the chain's wk4_row (layer 0 exact) -- the 0.634 oracle.
Stages:
  rec   one plain chain run per net: per-neuron kappa4 predictors from the chain's own post slices of layer l-1
        vec  ARC per-neuron diagonal state (504aldo k4_vec, k4_prop='ww'): n P2 (c4 K4v + c22 rowsum K22), carried by W*W
        s31  exact one-layer transport of the (4,), (2,2), (3,1) post slices
        gnorm chain wk4_row * |w_i|^2 / metric_c (row-metric instead of the average metric)
  lin   first-order model: a kappa4 change x at layer l moves the post mean by c4 x (c4 = (a^2-1) phi(a) / (24 s^3)),
        optionally the post variance by v4 x (feeds the next pre variance through W*W); mean carried by Phi (u W).
        Adjoint lambda_l of e = P_15 - T_15 gives the per-neuron sensitivity of the last-layer error.
        Prints layer windows, neuron-group masks and predictor scores (raw / sensitivity-weighted / final ratio).
  win   full reruns: r4 only on layer windows / neuron groups / predictors, to check the first-order model.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from var_subspace import lean_module, load_truth, weights_from_seed, prepare, residual, KS

WINDOWS = [(l, l) for l in range(1, 16)] + [(1, 4), (5, 8), (9, 12), (13, 15), (1, 8), (9, 15), (1, 15), (1, 14), (12, 15)]


def phi(x):
    return torch.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)


def rec(a):
    m = lean_module(); torch.set_default_device('cpu'); torch.set_default_dtype(torch.float64)
    st = m.build_static_coefs(1024); torch.set_default_dtype(torch.float32); torch.set_default_device('cuda')
    tr = load_truth('mini'); n = 1024; mc = 2.0
    for idx in a.idx:
        out = a.out / f'pred{idx}.npz'
        if out.exists():
            continue
        W = np.stack(weights_from_seed(tr['mlp_seed'][idx]))
        Wl = [torch.tensor(w.T, device='cuda').contiguous() for w in W]   # lean orientation: pre = Wl @ h
        P = {k: np.zeros((16, n)) for k in ('vec', 's31', 's22', 's4')}
        def gh(l, G, s):
            if l == 15:
                return G
            Wn = Wl[l + 1]; W2 = Wn * Wn
            K4v = s['K4v']; K22 = s['K22'].clone(); K31 = s['K31'].clone()
            K22.fill_diagonal_(0); K31.fill_diagonal_(0)
            gv = n * st['P2'] * (st['k4_c4'] * K4v + st['k4_c22'] * K22.sum(0))
            P['vec'][l + 1] = (st['wk4_c4'] * mc * (W2 @ gv)).cpu().numpy()
            s4 = (W2 * W2) @ K4v
            s22 = s4 + 3.0 * (W2 * (W2 @ K22)).sum(1)
            s31 = s22 + 4.0 * ((Wn ** 3) * (Wn @ K31.T)).sum(1)
            P['s4'][l + 1] = s4.cpu().numpy(); P['s22'][l + 1] = s22.cpu().numpy(); P['s31'][l + 1] = s31.cpu().numpy()
            return G
        m.HOOK = None; t0 = time.time()
        m.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True, g_hook=gh)
        np.savez(out, **P); print('REC', idx, round(time.time() - t0, 1), flush=True)


def net(a, idx, tr):
    ch, v, k3, k4, r3, r4, q3, q4 = prepare(a, idx)
    W = torch.tensor(np.stack(weights_from_seed(tr['mlp_seed'][idx])), device='cuda', dtype=torch.float64)  # x @ W[l]
    mu = torch.tensor(ch['mu'], device='cuda', dtype=torch.float64); var = torch.tensor(ch['var'], device='cuda', dtype=torch.float64)
    s = var.sqrt(); al = mu / s; p = phi(al) / s
    c4 = (al * al - 1) * phi(al) / (24 * s ** 3)
    Pm = torch.tensor(ch['P'], device='cuda', dtype=torch.float64)
    v4 = (-2 * (mu / var) * p) / 24 - 2 * Pm * c4          # d Var(relu) / d kappa4
    Phi = torch.special.ndtr(al); kap = phi(al) / (2 * s)
    e = Pm[-1] - torch.tensor(tr['all_layer_means'][idx][-1], device='cuda', dtype=torch.float64)
    pr = np.load(a.out / f'pred{idx}.npz')
    rownorm = (W * W).sum(1)                                 # |w_i|^2 per output neuron, (16, n)
    preds = {k: pr[k] for k in ('vec', 's31', 's22', 's4')}
    preds['gnorm'] = ch['k4'] * rownorm.cpu().numpy() / 2.0
    vr = ch['var'] / ch['var'].mean(1, keepdims=True)
    preds['gvar'] = ch['k4'] * vr; preds['gvar2'] = ch['k4'] * vr ** 2
    for ki, k in enumerate(KS):
        if k in (64, 128):
            preds[f'sub{k}'] = q4[:, ki]                      # (2, 16, n), MC-based, already chain-residualized
    return dict(ch=ch, k4=k4, r4=r4, W=W, mu=mu, var=var, al=al, c4=c4, v4=v4, Phi=Phi, kap=kap, e=e, rownorm=rownorm, preds=preds)


def carry(d, x, var_hop):
    """x: (16, n) torch per-layer kappa4 change -> last-layer mean change"""
    u = torch.zeros(1024, device='cuda', dtype=torch.float64); dv = None
    for l in range(16):
        if l:
            u = d['Phi'][l] * (u @ d['W'][l])
            if var_hop and dv is not None:
                u = u + d['kap'][l] * (dv @ (d['W'][l] ** 2))
        u = u + d['c4'][l] * x[l]
        dv = d['v4'][l] * x[l]
    return u


def adjoint(d, var_hop):
    lam = torch.zeros(16, 1024, device='cuda', dtype=torch.float64); lam[15] = d['e']
    for l in range(15, 0, -1):
        lam[l - 1] = d['W'][l] @ (d['Phi'][l] * lam[l])
    sens = d['c4'] * lam
    if var_hop:
        nu = torch.zeros_like(lam)
        for l in range(15):
            nu[l] = (d['W'][l + 1] ** 2) @ (d['kap'][l + 1] * lam[l + 1])
        sens = sens + d['v4'] * nu
    return lam, sens


def T(x):
    return torch.tensor(np.asarray(x), device='cuda', dtype=torch.float64)


def groups(d, lam, sens):
    """per-layer quartile masks (16, n) -> name -> list of 4 masks"""
    feats = dict(alpha=d['al'], sigma=d['var'].sqrt(), rownorm=d['rownorm'], c4abs=d['c4'].abs(),
                 lamabs=lam.abs(), sensabs=sens.abs(), k4chain=T(d['ch']['k4']), r4abs=T(np.abs(d['r4']).mean(0)))
    out = {}
    for name, f in feats.items():
        q = torch.quantile(f[1:].float(), torch.tensor([.25, .5, .75], device='cuda'), dim=1).double()   # (3, 15)
        q = torch.cat([torch.zeros(3, 1, device='cuda', dtype=torch.float64), q], 1)
        ms = []
        for g in range(4):
            lo = q[g - 1][:, None] if g else -torch.inf
            hi = q[g][:, None] if g < 3 else torch.inf
            ms.append(((f >= lo) & (f < hi)).double())
        out[name] = ms
    return out


def lin(a):
    tr = load_truth('mini'); acc = {}; base = 0.0; rows = []
    def add(key, val):
        acc[key] = acc.get(key, 0.0) + val
    for idx in a.idx:
        d = net(a, idx, tr); e = d['e']; b = float((e * e).mean()); base += b
        rA, rB = T(d['r4'][0]), T(d['r4'][1])
        for vh in (0, 1):
            tag = 'mv' if vh else 'm'
            def ratio(xA, xB):
                uA, uB = carry(d, xA, vh), carry(d, xB, vh)
                return float(((e + uA) * (e + uB)).mean())
            add(f'{tag}:all', ratio(rA, rB))
            for (lo, hi) in WINDOWS:
                msk = torch.zeros(16, 1, device='cuda', dtype=torch.float64); msk[lo:hi + 1] = 1
                add(f'{tag}:L{lo}-{hi}', ratio(rA * msk, rB * msk))
            lam, sens = adjoint(d, vh)
            # first-order gain share per layer: 2 sens . rbar / |e|^2 (in units of the final ratio)
            g1 = (2 * sens * (rA + rB) / 2).sum(1) / (e * e).sum()
            rows.append(dict(idx=idx, tag=tag, lin_term=g1.tolist()))
            G = groups(d, lam, sens)
            for name, ms in G.items():
                for gi, mk in enumerate(ms):
                    add(f'{tag}:grp:{name}:q{gi}', ratio(rA * mk, rB * mk))
            # predictors: raw share, sensitivity-weighted share, final ratio (direct and per-layer best scale)
            for pn, pv in d['preds'].items():
                if pv.ndim == 2:
                    pres = residual(d['ch']['k4'], pv); pres[0] = 0; pA = pB = T(pres)
                else:
                    pA, pB = T(pv[0]), T(pv[1])
                for wname, w in (('raw', torch.ones_like(sens)), ('sens', sens)):
                    tA, tB, xA, xB = w * rA, w * rB, w * pA, w * pB
                    den = (tA * tB).sum(1); eps = ((tA - xA) * (tB - xB)).sum(1)
                    corr = 0.5 * (tA * xB + tB * xA).sum(1) / torch.sqrt(den.clamp_min(1e-300) * (xA * xB).sum(1).abs().clamp_min(1e-300))
                    rows.append(dict(idx=idx, tag=tag, pred=pn, w=wname, share=(1 - eps / den.clamp_min(1e-300)).tolist(), corr=corr.tolist()))
                add(f'{tag}:pred:{pn}:direct', ratio(pA, pB))
                # per-layer LS scale of the predictor onto the residual (cross-half), this net only (optimistic)
                beta = (0.5 * (rA * pB + rB * pA)).sum(1) / (pA * pB).sum(1).clamp_min(1e-300)
                add(f'{tag}:pred:{pn}:bestscale', ratio(pA * beta[:, None], pB * beta[:, None]))
        print('LIN', idx, flush=True)
    res = dict(base=base, ratios={k: v / base for k, v in acc.items()}, rows=rows)
    (a.out / 'sens_lin.json').write_text(json.dumps(res))
    for k, v in res['ratios'].items():
        print(f'{k:40s} {v:.4f}')
    # per-layer tables averaged over nets
    for tag in ('m', 'mv'):
        lt = np.mean([r['lin_term'] for r in rows if r.get('tag') == tag and 'lin_term' in r], 0)
        print(tag, 'lin_term per layer', ' '.join(f'{x:+.3f}' for x in lt))
        for pn in d['preds']:
            for wname in ('raw', 'sens'):
                sh = np.mean([r['share'] for r in rows if r.get('pred') == pn and r['w'] == wname and r['tag'] == tag], 0)
                cr = np.mean([r['corr'] for r in rows if r.get('pred') == pn and r['w'] == wname and r['tag'] == tag], 0)
                print(f'{tag} {pn:7s} {wname:4s} share ' + ' '.join(f'{x:+.2f}' for x in sh[1:]) + ' | corr ' + ' '.join(f'{x:+.2f}' for x in cr[1:]))


def win(a):
    m = lean_module(); torch.set_default_device('cpu'); torch.set_default_dtype(torch.float64)
    st = m.build_static_coefs(1024); torch.set_default_dtype(torch.float32); torch.set_default_device('cuda')
    tr = load_truth('mini'); acc = {}; base = 0.0
    modes = a.modes.split(',')
    for idx in a.idx:
        d = net(a, idx, tr); W = np.stack(weights_from_seed(tr['mlp_seed'][idx])); gt = tr['all_layer_means'][idx].astype(float)
        b = float(np.mean((d['ch']['P'][-1] - gt[-1]) ** 2)); base += b
        lam, sens = adjoint(d, 1); G = groups(d, lam, sens)
        for mode in modes:
            ps = []
            for h in range(2):
                tok = mode.split(':')
                if tok[0] == 'L':
                    lo, hi = map(int, tok[1].split('-')); x = d['r4'][h].copy(); x[:lo] = 0; x[hi + 1:] = 0
                elif tok[0] == 'grp':
                    x = d['r4'][h] * G[tok[1]][int(tok[2])].cpu().numpy()
                elif tok[0] == 'pred':
                    pv = d['preds'][tok[1]]
                    if pv.ndim == 2:
                        x = residual(d['ch']['k4'], pv); x[0] = 0
                    else:
                        x = pv[h].copy()
                    if len(tok) > 2 and tok[2] == 'bestscale':
                        rA, rB = d['r4']; pA = pB = x if pv.ndim == 2 else None
                        if pv.ndim == 3:
                            pA, pB = pv[0], pv[1]
                        beta = (0.5 * (rA * pB + rB * pA)).sum(1) / np.maximum((pA * pB).sum(1), 1e-300)
                        x = x * beta[:, None]
                elif tok[0] == 'all':
                    x = d['r4'][h]
                else:
                    raise ValueError(mode)
                xt = torch.tensor(x, device='cuda', dtype=torch.float32)
                def hook(l, name, y):
                    if l and name == 'k4':
                        return y + xt[l]
                    return y
                m.HOOK = hook
                pred = m.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True)
                if isinstance(pred, tuple):
                    pred = pred[0]
                ps.append(pred.cpu().numpy().astype(float) - gt)
            val = float(np.mean(ps[0][-1] * ps[1][-1])); acc[mode] = acc.get(mode, 0.0) + val
            print('WIN', idx, mode, round(val / b, 4), flush=True)
    res = {k: v / base for k, v in acc.items()}
    (a.out / f'sens_win_{a.tag}.json').write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f'WTOTAL {k:30s} {v:.4f}', flush=True)


def img(a):
    """Last-layer image of the residual: M_l = J_{15<-l} diag(c4_l) (mean channel). Which directions of r_l reach the output."""
    tr = load_truth('mini'); acc = {}; base = 0.0; spec = []; lf = {}
    KK = (1, 4, 16, 64, 256)
    def add(k, v):
        acc[k] = acc.get(k, 0.0) + v
    for idx in a.idx:
        d = net(a, idx, tr); e = d['e']; b = float((e * e).mean()); base += b
        rA, rB = T(d['r4'][0]), T(d['r4'][1]); n = 1024
        J = torch.eye(n, device='cuda', dtype=torch.float64); Ms = [None] * 16
        for l in range(15, 0, -1):
            Ms[l] = J * d['c4'][l][None, :]
            J = J @ (d['Phi'][l][:, None] * d['W'][l].T)          # J_{15<-l-1} = J_{15<-l} diag(Phi_l) W_l^T
        uA = sum(Ms[l] @ rA[l] for l in range(1, 16)); uB = sum(Ms[l] @ rB[l] for l in range(1, 16))
        add('all', float(((e + uA) * (e + uB)).mean()))
        add('u_power', float((uA * uB).mean())); add('e_dot_u', float((e * (uA + uB) / 2).mean()))
        Vs = []
        for l in range(1, 16):
            U, S, Vh = torch.linalg.svd(Ms[l]); Vs.append(Vh)
            sp = (S * S).cumsum(0) / (S * S).sum()
            spec.append(dict(idx=idx, l=l, top=[float(sp[k - 1]) for k in KK]))
        for K in KK:
            xA = torch.stack([torch.zeros(n, device='cuda', dtype=torch.float64)] + [Vs[l - 1][:K].T @ (Vs[l - 1][:K] @ rA[l]) for l in range(1, 16)])
            xB = torch.stack([torch.zeros(n, device='cuda', dtype=torch.float64)] + [Vs[l - 1][:K].T @ (Vs[l - 1][:K] @ rB[l]) for l in range(1, 16)])
            vA = sum(Ms[l] @ xA[l] for l in range(1, 16)); vB = sum(Ms[l] @ xB[l] for l in range(1, 16))
            add(f'svdK{K}', float(((e + vA) * (e + vB)).mean()))
            # residual power that lives in that K-dim row space (per layer averaged)
            add(f'svdK{K}_powshare', float(sum((xA[l] * xB[l]).sum() / (rA[l] * rB[l]).sum() for l in range(1, 16)) / 15) * b)
        # column-norm selector (computable without e): top quartile of ||M_l[:, i]||
        cn = torch.stack([torch.zeros(n, device='cuda', dtype=torch.float64)] + [Ms[l].norm(dim=0) for l in range(1, 16)])
        for qi, q in enumerate((0.75, 0.5)):
            thr = torch.quantile(cn[1:].float(), q, dim=1).double(); mk = torch.cat([torch.zeros(1, device='cuda', dtype=torch.float64), thr])[:, None]
            mk = (cn >= mk).double()
            vA = sum(Ms[l] @ (rA[l] * mk[l]) for l in range(1, 16)); vB = sum(Ms[l] @ (rB[l] * mk[l]) for l in range(1, 16))
            add(f'colnorm_top{int((1-q)*100)}', float(((e + vA) * (e + vB)).mean()))
        # predictors: last-layer image match and linear-term capture
        for pn, pv in d['preds'].items():
            if pv.ndim == 2:
                pres = residual(d['ch']['k4'], pv); pres[0] = 0; pA = pB = T(pres)
            else:
                pA, pB = T(pv[0]), T(pv[1])
            wA = sum(Ms[l] @ pA[l] for l in range(1, 16)); wB = sum(Ms[l] @ pB[l] for l in range(1, 16))
            lf.setdefault(pn, [0.0, 0.0, 0.0, 0.0])
            lf[pn][0] += float((e * (wA + wB) / 2).sum()); lf[pn][1] += float((e * (uA + uB) / 2).sum())
            lf[pn][2] += float(((uA - wA) * (uB - wB)).sum()); lf[pn][3] += float((uA * uB).sum())
            # best global scale of the image (one number per net)
            beta = float((0.5 * (uA * wB + uB * wA)).sum() / (wA * wB).sum().clamp_min(1e-300))
            add(f'img:{pn}:direct', float(((e + wA) * (e + wB)).mean())); add(f'img:{pn}:beta', float(((e + beta * wA) * (e + beta * wB)).mean()))
        print('IMG', idx, flush=True)
    res = dict(ratios={k: v / base for k, v in acc.items()}, spec=spec,
               pred={k: dict(linfrac=v[0] / v[1], imgshare=1 - v[2] / v[3]) for k, v in lf.items()})
    (a.out / 'sens_img.json').write_text(json.dumps(res, indent=1))
    for k, v in res['ratios'].items():
        print(f'{k:28s} {v:.4f}')
    for k, v in res['pred'].items():
        print(f'pred {k:7s} linear-term fraction {v["linfrac"]:+.3f}  image share {v["imgshare"]:+.3f}')
    sp = np.array([[s['l']] + s['top'] for s in spec])
    for l in range(1, 16):
        print(f'layer {l:2d} singular power in top {KK}: ' + ' '.join(f'{x:.3f}' for x in sp[sp[:, 0] == l, 1:].mean(0)))


def slicemc(a):
    """MC truth of the post-activation kappa4 slices (4,), (2,2), (3,1) of layers 0-14, transported one layer exactly
    (T4, T22, T31) and the pre-activation kappa4 of layers 1-15 from the same samples; two independent halves."""
    tr = load_truth('mini'); chunk = a.chunk; nch = int(np.ceil(a.n / chunk / 2)) * 2; n = 1024
    for idx in a.idx:
        out = a.out / f'slice{idx}.npz'
        if out.exists():
            continue
        W = torch.tensor(np.stack(weights_from_seed(tr['mlp_seed'][idx])), device='cuda')
        Tm = torch.tensor(tr['all_layer_means'][idx], device='cuda', dtype=torch.float32)
        Cz = torch.zeros(16, n, device='cuda'); Cz[1:] = torch.stack([Tm[l] @ W[l + 1] for l in range(15)])
        V = torch.zeros(2, 15, 3, n, device='cuda', dtype=torch.float64)          # sums hc^2, hc^3, hc^4 (post, layers 0-14)
        Q = torch.zeros(2, 15, 3, n, n, device='cuda', dtype=torch.float64)      # hc^T hc, (hc^2)^T hc^2, (hc^3)^T hc
        S = torch.zeros(2, 16, 4, n, device='cuda', dtype=torch.float64)         # pre d .. d^4
        A = torch.zeros(2, 15, n, n, device='cuda', dtype=torch.float64)         # (hc_l^2)^T d_{l+1}^2 for the (2,1,1) pattern
        g = torch.Generator(device='cuda').manual_seed(7000003 + int(tr['mlp_seed'][idx]) % 1000000007)
        x = torch.empty(chunk, n, device='cuda'); t0 = time.time()
        for ci in range(nch):
            torch.randn(x.shape, generator=g, out=x); h = x; hf = ci % 2
            for l in range(16):
                z = h @ W[l]; d = z - Cz[l]; d2 = d * d
                if l:
                    A[hf, l - 1] += (h2.T @ d2).double()
                S[hf, l] += torch.stack([d.sum(0), d2.sum(0), (d2 * d).sum(0), (d2 * d2).sum(0)]).double()
                h = z.clamp_min(0)
                if l < 15:
                    hc = h - Tm[l]; h2 = hc * hc; h3 = h2 * hc
                    V[hf, l] += torch.stack([h2.sum(0), h3.sum(0), (h2 * h2).sum(0)]).double()
                    Q[hf, l, 0] += (hc.T @ hc).double(); Q[hf, l, 1] += (h2.T @ h2).double(); Q[hf, l, 2] += (h3.T @ hc).double()
            if (ci + 1) % 100 == 0:
                print('SLICEMC', idx, ci + 1, nch, round(time.time() - t0, 1), flush=True)
        nh = nch // 2 * chunk; V /= nh; Q /= nh; S /= nh; A /= nh
        out_d = dict(T4=np.zeros((2, 16, n)), T22=np.zeros((2, 16, n)), T31=np.zeros((2, 16, n)), T211=np.zeros((2, 16, n)), k4=np.zeros((2, 16, n)), n=nh)
        for hf in range(2):
            a1, a2, a3, a4 = S[hf].unbind(1)
            out_d['k4'][hf] = (a4 - 4 * a1 * a3 - 3 * a2 * a2 + 12 * a2 * a1 * a1 - 6 * a1 ** 4).cpu().numpy()
            for l in range(15):
                m2, m3, m4 = V[hf, l]; C2, A22, A31 = Q[hf, l]
                K4v = m4 - 3 * m2 * m2
                K22 = A22 - m2[:, None] * m2[None, :] - 2 * C2 * C2; K22.fill_diagonal_(0)
                K31 = A31 - 3 * m2[:, None] * C2; K31.fill_diagonal_(0)                    # triple index first
                Wn = W[l + 1].double(); W2 = Wn * Wn
                out_d['T4'][hf, l + 1] = ((W2 * W2).T @ K4v).cpu().numpy()
                out_d['T22'][hf, l + 1] = (3 * ((K22 @ W2) * W2).sum(0)).cpu().numpy()
                out_d['T31'][hf, l + 1] = (4 * ((K31 @ Wn) * Wn ** 3).sum(0)).cpu().numpy()
                # S_i = sum_a W_ai^2 kappa(h_a, h_a, y_i, y_i), y = hc @ W; it contains (4,) once, (3,1)/2, (2,2)/3 and (2,1,1)/6
                QW = C2 @ Wn; vy = (Wn * QW).sum(0)
                Kay = A[hf, l] - m2[:, None] * vy[None, :] - 2 * QW * QW
                Ssum = (W2 * Kay).sum(0)
                out_d['T211'][hf, l + 1] = (6 * (Ssum - torch.tensor(out_d['T4'][hf, l + 1], device='cuda') - torch.tensor(out_d['T31'][hf, l + 1], device='cuda') / 2 - torch.tensor(out_d['T22'][hf, l + 1], device='cuda') / 3)).cpu().numpy()
        np.savez(out, **out_d); print('SLICEMC_DONE', idx, round(time.time() - t0, 1), flush=True)


def tree(d, W):
    """Leading-order (tree-diagram) birth of the post-activation (2,1,1) and (1,1,1,1) kappa4 of layer l by a ReLU on
    correlated Gaussian pre-activations, contracted exactly with w_i^4 of layer l+1. Chain's own pre mean / var /
    covariance (C_pre(l) = W_l^T C_post(l-1) W_l). Three n^3 products per layer. Returns (T211, T1111) as (16, n)."""
    ch = d['ch']; n = 1024; t211 = np.zeros((16, n)); t1111 = np.zeros((16, n))
    for l in range(15):
        Wl = W[l]; Wn = W[l + 1]
        Cpost_prev = torch.tensor(ch[f'C{l - 1}'], device='cuda', dtype=torch.float64) if l else torch.eye(n, device='cuda', dtype=torch.float64)
        Cpre = Wl.T @ Cpost_prev @ Wl
        mu, v = d['mu'][l], d['var'][l]; s = v.sqrt(); al = mu / s; ph = phi(al); Ph = torch.special.ndtr(al); g = s * ph + mu * Ph
        rho = Cpre / (s[:, None] * s[None, :]); rho.fill_diagonal_(0)
        c1, c2, c3 = s * Ph, s * ph, -s * al * ph
        lam = 2 * s * s * Ph * (1 - Ph) - 2 * s * ph * g; eta = 2 * s * g * (1 - Ph)
        P = rho @ (c1[:, None] * Wn)
        Q2 = (rho * rho) @ ((c1 * c1)[:, None] * Wn * Wn)
        X = c2[:, None] * Wn * P; Y = rho @ X
        t211[l + 1] = (6 * (Wn * Wn * (lam[:, None] * (P * P - Q2) + 2 * eta[:, None] * Y)).sum(0)).cpu().numpy()
        t1111[l + 1] = (4 * (Wn * c3[:, None] * P ** 3).sum(0) + 12 * (X * Y).sum(0)).cpu().numpy()
    return t211, t1111


def slice_(a):
    """Split the kappa4 residual by index pattern: truth = T4 + T22 + T31 + rest (rest = (2,1,1) + (1,1,1,1));
    chain = s4 + (s22 - s4) + (s31 - s22) + (G - s31) from its own post slices. Each error piece is residualized
    against the chain's wk4_row like r4 and scored by its last-layer image."""
    tr = load_truth('mini'); acc = {}; base = 0.0; lf = {}
    for idx in a.idx:
        d = net(a, idx, tr); e = d['e']; b = float((e * e).mean()); base += b
        sl = np.load(a.out / f'slice{idx}.npz'); pr = np.load(a.out / f'pred{idx}.npz'); G = d['ch']['k4']
        J = torch.eye(1024, device='cuda', dtype=torch.float64); Ms = [None] * 16
        for l in range(15, 0, -1):
            Ms[l] = J * d['c4'][l][None, :]; J = J @ (d['Phi'][l][:, None] * d['W'][l].T)
        def image(x):
            return sum(Ms[l] @ T(x[l]) for l in range(1, 16))
        c4p, c22, c31 = pr['s4'], pr['s22'] - pr['s4'], pr['s31'] - pr['s22']; crest = G - pr['s31']
        t211, t1111 = tree(d, d['W'])
        for nm, tv, key in (('211', t211, 'T211'), ('1111', t1111, None)):
            tA = sl[key][0] if key else sl['k4'][0] - sl['T4'][0] - sl['T22'][0] - sl['T31'][0] - sl['T211'][0]
            tB = sl[key][1] if key else sl['k4'][1] - sl['T4'][1] - sl['T22'][1] - sl['T31'][1] - sl['T211'][1]
            cr = [float(0.5 * np.mean(tv[l] * (tA[l] + tB[l])) / np.sqrt(max(np.mean(tA[l] * tB[l]), 1e-300) * np.mean(tv[l] ** 2))) for l in range(1, 16)]
            sc = [float(0.5 * np.mean(tv[l] * (tA[l] + tB[l])) / np.mean(tv[l] ** 2)) for l in range(1, 16)]
            print('TREE', idx, nm, 'corr', ' '.join(f'{x:+.2f}' for x in cr), '| LS scale', ' '.join(f'{x:.2f}' for x in sc), flush=True)
        pieces = {}
        for hf in range(2):
            k4 = sl['k4'][hf]; T4, T22, T31 = sl['T4'][hf], sl['T22'][hf], sl['T31'][hf]; rest = k4 - T4 - T22 - T31
            T211 = sl['T211'][hf]; T1111 = rest - T211
            raw = dict(total=k4 - G, e4=T4 - c4p, e22=T22 - c22, e31=T31 - c31, erest=rest - crest,
                       true31=T4 + T22 + T31 - G, truerest=rest, true211=T211, true1111=T1111,
                       s31_plus_t211=pr['s31'] + T211 - G, s31_plus_t1111=pr['s31'] + T1111 - G,
                       tree211=t211, tree1111=t1111, tree=t211 + t1111, s31_tree=pr['s31'] + t211 + t1111 - G)
            for k, v in raw.items():
                x = residual(G, G + v); x[0] = 0; pieces.setdefault(k, []).append(x)
        uA, uB = image(pieces['total'][0]), image(pieces['total'][1])
        add = lambda k, v: acc.__setitem__(k, acc.get(k, 0.0) + v)
        for k, (xA, xB) in pieces.items():
            iA, iB = image(xA), image(xB)
            add(f'{k}:final', float(((e + iA) * (e + iB)).mean()))
            lf.setdefault(k, [0.0] * 6)
            lf[k][0] += float((e * (iA + iB) / 2).sum()); lf[k][1] += float((e * (uA + uB) / 2).sum())
            lf[k][2] += float((iA * iB).sum()); lf[k][3] += float((uA * uB).sum())
            lf[k][4] += float(np.mean([np.mean(xA[l] * xB[l]) for l in range(9, 16)]))
            lf[k][5] += float(np.mean([np.mean(pieces['total'][0][l] * pieces['total'][1][l]) for l in range(9, 16)]))
            print('SLICE', idx, k, round(float(((e + iA) * (e + iB)).mean()) / b, 4), flush=True)
    res = dict(final={k: v / base for k, v in acc.items()},
               piece={k: dict(linfrac=v[0] / v[1], imgpow=v[2] / v[3], rawpow_l9_15=v[4] / v[5]) for k, v in lf.items()})
    (a.out / 'sens_slice.json').write_text(json.dumps(res, indent=1))
    for k, v in res['piece'].items():
        print(f'PIECE {k:9s} final {res["final"][k + ":final"]:.4f}  linear-term frac {v["linfrac"]:+.3f}  image power {v["imgpow"]:.3f}  raw power (l9-15) {v["rawpow_l9_15"]:.3f}', flush=True)


def lean_c():
    """var_subspace.lean_module plus a full pre-activation covariance hook HOOK(l, 'C', C)."""
    import types
    from var_subspace import ROOT
    path = ROOT / 'refs/504aldo/lean/lean_k3_aug.py'; src = path.read_text()
    edits = [('Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(dtype) for w in weights]',
              'Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(device=torch.get_default_device(), dtype=dtype) for w in weights]'),
             ('torch.set_num_threads(22)', 'pass'),
             ('                wk4_row = metric_c * dG\n', '                wk4_row = metric_c * dG\n                if HOOK is not None: wk4_row = HOOK(l, "k4", wk4_row)\n'),
             ('        C = 0.5 * (C + C.T)\n        if K3 is not None:', '        C = 0.5 * (C + C.T)\n        if HOOK is not None:\n            C = HOOK(l, "C", C)\n        if K3 is not None:')]
    for old, new in edits:
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    m = types.ModuleType('leanc'); m.__file__ = str(path); exec(compile(src, str(path), 'exec'), m.__dict__); m.HOOK = None
    return m


def cfix(a):
    """Is the kappa4 residual born from the chain's covariance error? Replace the pre-activation covariance by the
    MC truth W_l^T Cov(h_{l-1}) W_l (C: full, Coff: off-diagonal only, Cdiag: diagonal only) and read (1) the final ratio,
    (2) what is left of the kappa4 residual in that run: its own wk4_row vs MC kappa4 after the per-layer affine fit,
    scored by its last-layer image (first-order map of the plain chain)."""
    m = lean_c(); torch.set_default_device('cpu'); torch.set_default_dtype(torch.float64)
    st = m.build_static_coefs(1024); torch.set_default_dtype(torch.float32); torch.set_default_device('cuda')
    tr = load_truth('mini'); acc = {}; base = 0.0; img = {}
    modes = a.modes.split(',')
    for idx in a.idx:
        d = net(a, idx, tr); W = np.stack(weights_from_seed(tr['mlp_seed'][idx])); gt = tr['all_layer_means'][idx].astype(float)
        e = d['e']; b = float((e * e).mean()); base += b
        mcz = np.load(a.out / f'mc{idx}.npz'); Q = mcz['Q']                              # (2, 16, n, n) post covariance (truth-mean centred)
        Wt = torch.tensor(W, device='cuda', dtype=torch.float32)
        k4t = d['k4']                                                                    # (2, 16, n) MC kappa4
        J = torch.eye(1024, device='cuda', dtype=torch.float64); Ms = [None] * 16
        for l in range(15, 0, -1):
            Ms[l] = J * d['c4'][l][None, :]; J = J @ (d['Phi'][l][:, None] * d['W'][l].T)
        def image(x):
            return sum(Ms[l] @ T(x[l]) for l in range(1, 16))
        r0 = [image(d['r4'][h]) for h in range(2)]
        for mode in modes:
            tok = set(mode.split('+')); ps = []; rows = []
            for h in range(2):
                Qh = torch.tensor(Q[h], device='cuda', dtype=torch.float32); rec = {}
                r4h = torch.tensor(d['r4'][h], device='cuda', dtype=torch.float32); k4h = torch.tensor(k4t[h], device='cuda', dtype=torch.float32)
                def hook(l, name, x):
                    if l == 0:
                        return x
                    if name == 'C' and tok & {'C', 'Coff', 'Cdiag'}:
                        Ct = Wt[l].T @ Qh[l - 1] @ Wt[l]; Ct = 0.5 * (Ct + Ct.T)
                        if 'Coff' in tok:
                            Ct = Ct - torch.diag(torch.diagonal(Ct)) + torch.diag(torch.diagonal(x))
                        if 'Cdiag' in tok:
                            Ct = x - torch.diag(torch.diagonal(x)) + torch.diag(torch.diagonal(Ct))
                        return Ct
                    if name == 'k4':
                        rec[l] = x.detach().double().cpu().numpy()
                        if 'k4true' in tok:
                            return k4h[l]
                        if 'k4res' in tok:
                            return x + r4h[l]
                    return x
                m.HOOK = hook
                pred = m.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True)
                if isinstance(pred, tuple):
                    pred = pred[0]
                ps.append(pred.cpu().numpy().astype(float) - gt)
                k4run = np.zeros((16, 1024))
                for l, x in rec.items():
                    k4run[l] = x
                rr = residual(k4run, k4t[h]); rr[0] = 0; rows.append(rr)
            val = float(np.mean(ps[0][-1] * ps[1][-1])); acc[mode] = acc.get(mode, 0.0) + val
            # what is left of the kappa4 residual in this run, through the plain chain's first-order image
            iA, iB = image(rows[0]), image(rows[1])
            img.setdefault(mode, [0.0, 0.0, 0.0, 0.0])
            img[mode][0] += float((iA * iB).sum()); img[mode][1] += float((r0[0] * r0[1]).sum())
            img[mode][2] += float((e * (iA + iB) / 2).sum()); img[mode][3] += float((e * (r0[0] + r0[1]) / 2).sum())
            pw = np.mean([np.mean(rows[0][l] * rows[1][l]) / np.mean(d['r4'][0][l] * d['r4'][1][l]) for l in (6, 9, 12, 14, 15)])
            print('CFIX', idx, mode, round(val / b, 4), 'k4-residual power vs plain (l6,9,12,14,15 avg)', round(float(pw), 3), flush=True)
    res = dict(ratios={k: v / base for k, v in acc.items()}, k4left={k: dict(image_power=v[0] / v[1], linear_term=v[2] / v[3]) for k, v in img.items()})
    (a.out / f'sens_cfix_{a.tag}.json').write_text(json.dumps(res, indent=1))
    for k in res['ratios']:
        print(f'CTOTAL {k:22s} ratio {res["ratios"][k]:.4f}  k4 residual left: image power {res["k4left"][k]["image_power"]:.3f} linear term {res["k4left"][k]["linear_term"]:.3f}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['rec', 'lin', 'win', 'img', 'cfix', 'slicemc', 'slice_']); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--idx', default='0,1,2,3'); p.add_argument('--modes', default='all'); p.add_argument('--tag', default='a'); p.add_argument('--n', type=float, default=2e7); p.add_argument('--chunk', type=int, default=16384)
    a = p.parse_args(); a.idx = [int(i) for i in a.idx.split(',')]
    torch.set_grad_enabled(False); torch.set_num_threads(6); torch.backends.cuda.matmul.allow_tf32 = False; torch.set_float32_matmul_precision('highest')
    globals()[a.stage](a)
