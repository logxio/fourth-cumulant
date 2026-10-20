"""LT2 corrector trainer (GPU, torch) on lt2_prep.py memmaps. Variants on top of lt_train.py (v1):
  --ext       8 extra own columns (coff_abs, cn2, delta, k22abs, k22sq, k21sq, k11sq, k11sum)
  --xall      the previous layer's standardized own columns carried through W and W*W
  --emb D     a learned D-dim per-neuron state of the previous layer carried through W and W*W (recurrent)
  --uin       the previous layer's carried correction as inputs: a = (u_{l-1} @ W_l)/es_l, Phi_l*a,
              ((u_{l-1}/es_{l-1})^2 @ W_l^2)  (recurrent)
Stages: local (row-sampled; non-recurrent only) -> seq (per-MLP batches through the recurrence, local-eps
loss, optional + beta * final loss) -> e2e (final-layer loss after transport, as v1).
  --aux-w G   auxiliary supervision (lt4_auxprep.py <prep>/<split>_AY.npy, MC variance truth): extra linear outputs
              predict the chain's variance errors (--aux-k ds: kappa*ds, the first-order mean error carried by the
              pre-activation variance error; dv: post-activation variance error; both) at every layer with MC
              moments; loss += G * their normalised MSE. Training only: the export drops the aux outputs.
Held-out readout = sum_mini MSE(e15 - u15) / sum_mini MSE(e15).
usage: lt2_train.py <prep_dir> <out_dir> [flags]
"""
import argparse, json, os, queue, threading, time
import numpy as np
import torch

L, N = 16, 1024
OWN1 = list(range(27))
V1T = list(range(27, 43))
EXT = list(range(43, 51))


def tf(x):
    return torch.sign(x) * torch.log1p(torch.abs(x) * 1e3)


class Net(torch.nn.Module):
    def __init__(self, fin, H, depth, out, act):
        super().__init__()
        A = torch.nn.GELU if act == "gelu" else torch.nn.ReLU
        mods = [torch.nn.Linear(fin, H), A()]
        for _ in range(depth - 1):
            mods += [torch.nn.Linear(H, H), A()]
        mods += [torch.nn.Linear(H, out)]
        self.f = torch.nn.Sequential(*mods)

    def forward(self, x):
        return self.f(x)


class GFeed:
    """Weight batches from a GPU-resident float16 copy (--wgpu)."""

    def __init__(self, Wg, batches):
        self.Wg = Wg; self.batches = batches

    def __iter__(self):
        for b in self.batches:
            yield b, self.Wg[torch.from_numpy(np.asarray(b)).to(self.Wg.device)]


class WFeed:
    """Background prefetch of float16 weight batches (memmap -> pinned -> cuda)."""

    def __init__(self, W, batches, dev):
        self.q = queue.Queue(maxsize=3)
        self.dev = dev
        self.t = threading.Thread(target=self.run, args=(W, batches), daemon=True)
        self.t.start()

    def run(self, W, batches):
        for b in batches:
            arr = torch.from_numpy(np.ascontiguousarray(W[b]))
            self.q.put((b, arr.pin_memory() if self.dev.type == "cuda" else arr))
        self.q.put(None)

    def __iter__(self):
        while True:
            it = self.q.get()
            if it is None:
                return
            b, arr = it
            yield b, arr.to(self.dev, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prep"); ap.add_argument("out")
    ap.add_argument("--ext", action="store_true"); ap.add_argument("--xall", action="store_true")
    ap.add_argument("--emb", type=int, default=0); ap.add_argument("--uin", action="store_true")
    ap.add_argument("--embw", action="store_true")
    ap.add_argument("--hidden", type=int, default=128); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--act", default="gelu")
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seq-epochs", type=int, default=0); ap.add_argument("--seq-lr", type=float, default=1e-3)
    ap.add_argument("--seq-beta", type=float, default=0.0)
    ap.add_argument("--e2e-epochs", type=int, default=4); ap.add_argument("--e2e-lr", type=float, default=1e-4)
    ap.add_argument("--e2e-sched", default="const")
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--init", default="")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--wgpu", action="store_true")
    ap.add_argument("--fm", action="store_true")   # train on full + mini (no held-out set): keeps the last epoch
    ap.add_argument("--aux-w", type=float, default=0.0); ap.add_argument("--aux-k", default="ds")
    ap.add_argument("--aux-in", action="store_true")   # analysis only: the MC targets as inputs (oracle bound)
    ap.add_argument("--aux-add", action="store_true")  # g += (asc/es) * the kappa*ds aux output (structured use)
    a = ap.parse_args()
    assert not a.fm or (a.wgpu and a.epochs == 0), "--fm needs --wgpu and the recurrent (seq) stage"
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    dev = torch.device(a.device)
    t0 = time.time()
    log = open(os.path.join(a.out, "train.log"), "a")

    def P(*s):
        msg = " ".join(str(x) for x in s) + f" t={time.time() - t0:.0f}s"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    P("args", json.dumps(vars(a)))
    own = OWN1 + (EXT if a.ext else [])
    Fo = len(own)
    D = {}
    for s in ("full", "mini"):
        X = np.load(f"{a.prep}/{s}_X.npy", mmap_mode="r")
        M = X.shape[0] if s == "mini" else min(a.ntrain, X.shape[0])
        D[s] = dict(X=X[:M], W=np.load(f"{a.prep}/{s}_W16.npy", mmap_mode="r")[:M], M=M,
                    **{k: torch.from_numpy(np.ascontiguousarray(np.load(f"{a.prep}/{s}_{k}.npy", mmap_mode="r")[:M])).to(dev)
                       for k in ("Phi", "P", "T", "E")})
    tr, ho = D["full"], D["mini"]
    AK = {"ds": [0], "dv": [1], "both": [0, 1]}[a.aux_k] if (a.aux_w > 0 or a.aux_in) else []
    naux = len(AK)
    if naux:
        for s, S in (("full", tr), ("mini", ho)):
            S["AY"] = torch.from_numpy(np.ascontiguousarray(np.load(f"{a.prep}/{s}_AY.npy", mmap_mode="r")[:S["M"]][:, AK])).to(dev)
            S["AM"] = torch.from_numpy(np.load(f"{a.prep}/{s}_AM.npy")[:S["M"]]).to(dev)
        AN = torch.from_numpy(np.load(f"{a.prep}/mini_AN.npy")[AK]).to(dev)                  # (naux, L) noise var
        # per-layer scale of each aux target on the labelled training MLPs; layer 0 (chain exact) and dv at the
        # last layer (not formed) carry no signal and are left out of the loss
        asc = torch.sqrt((tr["AY"][tr["AM"]].double() ** 2).mean(dim=(0, 3))).float() + 1e-30   # (naux, L)
        alive = torch.ones(naux, L, device=dev); alive[:, 0] = 0
        for j, c in enumerate(AK):
            if c == 1:
                alive[j, L - 1] = 0
        P("aux", a.aux_k, "labelled train", int(tr["AM"].sum()), "holdout", int(ho["AM"].sum()),
          "scale", np.array2string(asc.cpu().numpy(), precision=2))
    if a.wgpu:
        # --fm: one block for full + mini so the merged training set is a view, not a second copy
        Wall = torch.empty(tr["M"] + ho["M"], L, N, N, dtype=torch.float16, device=dev) if a.fm else None
        for S, o in ((tr, 0), (ho, tr["M"])):
            Wg = Wall[o:o + S["M"]] if a.fm else torch.empty(S["M"], L, N, N, dtype=torch.float16, device=dev)
            for i in range(0, S["M"], 25):
                Wg[i:i + 25] = torch.from_numpy(np.ascontiguousarray(S["W"][i:i + 25])).to(dev)
            S["Wg"] = Wg

    def feed(S, batches):
        return GFeed(S["Wg"], batches) if a.wgpu else WFeed(S["W"], batches, dev)
    es = torch.sqrt((tr["E"].double() ** 2).mean(dim=(0, 2))).float()          # (L,)
    P("local eps rms per layer", np.array2string(es.cpu().numpy(), precision=3))

    # standardized [own | v1 transported] per layer (fp16 on GPU)
    cols = own + V1T
    with torch.no_grad():
        acc = torch.zeros(L, len(cols), dtype=torch.float64, device=dev); acc2 = torch.zeros_like(acc)
        for i in range(0, tr["M"], 50):
            z = tf(torch.from_numpy(np.ascontiguousarray(tr["X"][i:i + 50][..., cols])).to(dev)).double()
            acc += z.sum(dim=(0, 2)); acc2 += (z * z).sum(dim=(0, 2))
        cnt = tr["M"] * N
        mu = (acc / cnt).float(); sd = (torch.sqrt(torch.clamp(acc2 / cnt - (acc / cnt) ** 2, min=0)) + 1e-6).float()

    def zbase(S):
        out = torch.empty(S["M"], L, N, len(cols), dtype=torch.float16, device=dev)
        for i in range(0, S["M"], 50):
            z = tf(torch.from_numpy(np.ascontiguousarray(S["X"][i:i + 50][..., cols])).to(dev))
            out[i:i + 50] = ((z - mu[None, :, None, :]) / sd[None, :, None, :]).half()
        return out

    tr["Z"] = zbase(tr); ho["Z"] = zbase(ho)
    if a.fm:
        # standardisation and es stay fitted on full (as the held-out runs); training then sees all 1100 MLPs
        tr = dict(M=tr["M"] + ho["M"], Wg=Wall, **{k: torch.cat([tr[k], ho[k]])
                                                    for k in ("Phi", "P", "T", "E", "Z") + (("AY", "AM") if naux else ())})
    P("zbase", tuple(tr["Z"].shape))
    OH = torch.eye(L, device=dev)

    # xall standardization from 64 training MLPs
    mu2 = torch.zeros(L, 2 * Fo, device=dev); sd2 = torch.ones(L, 2 * Fo, device=dev)
    if a.xall:
        with torch.no_grad():
            s1 = torch.zeros(L, 2 * Fo, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1); nb = 0
            for b, Wb in feed(tr, [np.arange(i, min(i + 8, tr["M"], 64)) for i in range(0, min(64, tr["M"]), 8)]):
                bt = torch.from_numpy(b).to(dev)
                for l in range(1, L):
                    Wl = Wb[:, l].float(); zp = tr["Z"][bt, l - 1, :, :Fo].float()
                    t = torch.cat([torch.bmm(Wl.transpose(1, 2), zp), torch.bmm((Wl * Wl).transpose(1, 2), zp)], -1).double()
                    s1[l] += t.sum(dim=(0, 1)); s2[l] += (t * t).sum(dim=(0, 1))
                nb += len(b)
            m_ = s1 / (nb * N); mu2[1:] = m_[1:].float()
            sd2[1:] = (torch.sqrt(torch.clamp(s2[1:] / (nb * N) - m_[1:] ** 2, min=0)) + 1e-6).float()

    ne = a.emb if a.embw else 2 * a.emb        # --embw: the state rides W only (no W*W leg)
    nin = naux if a.aux_in else 0
    naux = naux if a.aux_w > 0 else 0
    fin = len(cols) + (2 * Fo if a.xall else 0) + ne + (3 if a.uin else 0) + nin + L
    net = Net(fin, a.hidden, a.depth, 1 + a.emb + naux, a.act).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location=dev); net.load_state_dict(ck["net"]); P("init from", a.init)
    P("fin", fin, "params", sum(p.numel() for p in net.parameters()))
    recurrent = a.emb > 0 or a.uin

    def layer_in(S, bt, l, Wl, h, u):
        """inputs (B, N, fin) of layer l for MLPs bt; Wl (B,N,N) f32 or None at l=0."""
        B = len(bt)
        parts = [S["Z"][bt, l].float()]
        if a.xall:
            if l == 0:
                parts.append(torch.zeros(B, N, 2 * Fo, device=dev))
            else:
                zp = S["Z"][bt, l - 1, :, :Fo].float()
                t = torch.cat([torch.bmm(Wl.transpose(1, 2), zp), torch.bmm((Wl * Wl).transpose(1, 2), zp)], -1)
                parts.append((t - mu2[l]) / sd2[l])
        if a.emb:
            if l == 0:
                parts.append(torch.zeros(B, N, ne, device=dev))
            elif a.embw:
                parts.append(torch.bmm(Wl.transpose(1, 2), h))
            else:
                parts.append(torch.cat([torch.bmm(Wl.transpose(1, 2), h), torch.bmm((Wl * Wl).transpose(1, 2), h)], -1))
        if a.uin:
            if l == 0:
                parts.append(torch.zeros(B, N, 3, device=dev))
            else:
                av = torch.bmm(u[:, None, :], Wl).squeeze(1) / es[l]
                cv = torch.bmm(((u / es[l - 1]) ** 2)[:, None, :], Wl * Wl).squeeze(1)
                parts.append(torch.stack([av, S["Phi"][bt, l] * av, cv], -1))
        if nin:
            parts.append((S["AY"][bt, :, l] / asc[None, :, l, None]).transpose(1, 2))
        parts.append(OH[l].expand(B, N, L))
        return torch.cat(parts, -1)

    def run(S, bt, Wb, want_local, aterms=None):
        """forward one MLP batch through the 16 layers; returns u15 (B,N), local loss terms; appends the
        per-layer aux squared errors (B, naux) to aterms when given."""
        B = len(bt); u = None; h = None; lterms = []
        for l in range(L):
            Wl = Wb[:, l].float() if l > 0 else None
            x = layer_in(S, bt, l, Wl, h, u)
            o = net(x.reshape(-1, fin)).reshape(B, N, 1 + a.emb + naux)
            g = o[..., 0]
            if a.aux_add:
                g = g + o[..., 1 + a.emb] * (asc[0, l] / es[l])
            if a.emb:
                h = torch.tanh(o[..., 1:1 + a.emb])
            if aterms is not None:
                yt = S["AY"][bt, :, l] / asc[None, :, l, None]                         # (B, naux, N)
                aterms.append(((yt - o[..., 1 + a.emb:].transpose(1, 2)) ** 2).mean(dim=2) * alive[None, :, l])
            gp = g * es[l]
            u = gp if l == 0 else S["Phi"][bt, l] * torch.bmm(u[:, None, :], Wl).squeeze(1) + gp
            if want_local:
                lterms.append(((S["E"][bt, l] / es[l] - g) ** 2).mean(dim=1))
        return u, lterms

    aux_ho = {}

    def holdout(per_layer=False):
        net.eval(); B_ = C_ = 0.0; per = []; ln = torch.zeros(L, device=dev); ld = torch.zeros(L, device=dev)
        am = torch.zeros(naux, L, device=dev, dtype=torch.float64); ay2 = torch.zeros_like(am); na = 0
        with torch.no_grad():
            for b, Wb in feed(ho, [np.arange(i, min(i + 10, ho["M"])) for i in range(0, ho["M"], 10)]):
                bt = torch.from_numpy(b).to(dev)
                at = [] if naux else None
                u, lt = run(ho, bt, Wb, True, at)
                if naux:
                    mk = ho["AM"][bt]
                    if bool(mk.any()):
                        am += torch.stack(at, -1)[mk].double().sum(0)
                        ay2 += ((ho["AY"][bt][mk] / asc[None, :, :, None]) ** 2).double().mean(-1).sum(0)
                        na += int(mk.sum())
                e = (ho["T"][bt, -1] - ho["P"][bt, -1]).double()
                bb = (e ** 2).mean(dim=1); cc = ((e - u.double()) ** 2).mean(dim=1)
                B_ += float(bb.sum()); C_ += float(cc.sum()); per += list(zip(bb.tolist(), cc.tolist()))
                ln += torch.stack(lt).sum(dim=1)
                ld += ((ho["E"][bt] / es[None, :, None]) ** 2).mean(dim=2).sum(dim=0)
        net.train()
        if naux and na:
            # explained share of the aux signal: 1 - (MSE - noise) / (E y^2 - noise), noise from the MC halves
            nz = AN.double() / asc.double() ** 2
            aux_ho["share"] = (1 - (am / na - nz) / (ay2 / na - nz)).cpu().numpy()
        return C_ / B_, per, (ln / ld).cpu().numpy()

    best = None

    def keep(r, tag):
        nonlocal best
        if best is None or r < best[0]:
            best = (r, {k: v.detach().clone() for k, v in net.state_dict().items()}, tag)

    # ---- local stage (row-sampled) ----
    if not recurrent and a.epochs > 0 and not a.init:
        Fin_rows = tr["M"] * L * N
        Pin = torch.empty(Fin_rows, fin, dtype=torch.float16, device=dev)
        Ytr = torch.empty(Fin_rows, device=dev)
        with torch.no_grad():
            for b, Wb in feed(tr, [np.arange(i, min(i + 8, tr["M"])) for i in range(0, tr["M"], 8)]):
                bt = torch.from_numpy(b).to(dev)
                for l in range(L):
                    Wl = Wb[:, l].float() if l > 0 else None
                    x = layer_in(tr, bt, l, Wl, None, None)              # (B,N,fin)
                    rows = (bt[:, None] * L + l) * N + torch.arange(N, device=dev)[None, :]
                    Pin[rows.reshape(-1)] = x.reshape(-1, fin).half()
                    Ytr[rows.reshape(-1)] = (tr["E"][bt, l] / es[l]).reshape(-1)
        P("local inputs", tuple(Pin.shape))
        bs = 8192; steps = a.epochs * (Fin_rows // bs)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
        for ep in range(a.epochs):
            perm = torch.randperm(Fin_rows, device=dev); tot = 0.0
            for i in range(Fin_rows // bs):
                ix = perm[i * bs:(i + 1) * bs]
                loss = ((net(Pin[ix].float())[:, 0] - Ytr[ix]) ** 2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step(); tot += loss.item()
            P(f"local ep {ep} train nmse {tot / (Fin_rows // bs):.4f}")
        del Pin, Ytr; torch.cuda.empty_cache()
        r, _, lnm = holdout(); keep(r, "local")
        P(f"after local: holdout ratio {r:.4f}", "local nmse", np.round(lnm, 3))

    def seq_stage(epochs, lr, loss_kind, beta, tag, sched="const"):
        nonlocal best
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        nb_ep = (tr["M"] + a.batch - 1) // a.batch
        sch = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * nb_ep, pct_start=0.15)
               if sched == "onecycle" else None)
        rng = np.random.default_rng(a.seed + 7)
        for ep in range(epochs):
            order = rng.permutation(tr["M"])
            batches = [order[i:i + a.batch] for i in range(0, tr["M"], a.batch)]
            tot = 0.0; nb = 0
            for b, Wb in feed(tr, batches):
                bt = torch.from_numpy(b).to(dev)
                at = [] if naux and loss_kind == "local" else None
                u, lt = run(tr, bt, Wb, loss_kind == "local", at)
                e = tr["T"][bt, -1] - tr["P"][bt, -1]
                fin_l = (((e - u) ** 2).mean(dim=1) / ((e ** 2).mean(dim=1) + 1e-30)).mean()
                if loss_kind == "local":
                    loss = torch.stack(lt).mean() + beta * fin_l
                    if at is not None:
                        mk = tr["AM"][bt]
                        if bool(mk.any()):
                            loss = loss + a.aux_w * (torch.stack(at, -1)[mk].sum(-1) / alive.sum(-1)).mean()
                else:
                    loss = fin_l
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if sch is not None:
                    sch.step()
                tot += loss.item(); nb += 1
            if a.fm:
                best = (0.0, {k: v.detach().clone() for k, v in net.state_dict().items()}, f"{tag}{ep}")
                P(f"{tag} ep {ep} train loss {tot / nb:.4f} (fm: last epoch kept)")
                continue
            r, _, lnm = holdout(); keep(r, f"{tag}{ep}")
            P(f"{tag} ep {ep} train loss {tot / nb:.4f} holdout ratio {r:.4f}", "local nmse", np.round(lnm, 3),
              *(["aux share", np.round(aux_ho["share"], 3)] if "share" in aux_ho else []))

    if a.seq_epochs > 0:
        seq_stage(a.seq_epochs, a.seq_lr, "local", a.seq_beta, "seq", "onecycle")
    if a.e2e_epochs > 0:
        if best is not None:
            net.load_state_dict(best[1])
        seq_stage(a.e2e_epochs, a.e2e_lr, "final", 0.0, "e2e", a.e2e_sched)
    net.load_state_dict(best[1])
    rb, perb, lnm = holdout()
    P(f"best {best[2]} holdout ratio {rb:.4f}", "local nmse", np.round(lnm, 3))
    torch.save({"net": net.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "mu2": mu2.cpu(), "sd2": sd2.cpu(),
                "es": es.cpu(), "own": own, "cols": cols, "fin": fin, "naux": naux, "args": vars(a)},
               os.path.join(a.out, "lt2_model.pt"))
    json.dump({"holdout_ratio_best": rb, "best": best[2], "mini_raw": [p[0] for p in perb],
               "mini_corr": [p[1] for p in perb], "local_nmse": lnm.tolist(), "args": vars(a)},
              open(os.path.join(a.out, "result.json"), "w"), indent=1)
    P("saved", a.out)


if __name__ == "__main__":
    main()
