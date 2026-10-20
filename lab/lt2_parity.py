"""Check lt2_export.py folding: carried correction u15 from the torch model (training path) vs a numpy
re-implementation of the estimator's _lt2_layer on the exported npz, on a few prepped MLPs.

usage: lt2_parity.py <lt2_model.pt> <lt2_weights.npz> <prep_dir> [split] [k0,k1,...]
"""
import sys
import numpy as np
import torch

sys.argv += [] if len(sys.argv) > 4 else ["mini"]
L, N = 16, 1024


def tf(x):
    return np.sign(x) * np.log1p(np.abs(x) * 1e3)


def gelu(h):
    from scipy.special import ndtr
    return h * ndtr(h)


def main():
    src, npz, prep, split = sys.argv[1:5]
    ks = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "0,1").split(",")]
    ck = torch.load(src, map_location="cpu"); a = ck["args"]
    d = np.load(npz)
    fl = [int(x) for x in d["flags"]]
    Fo, xall, D, uin, nm, relu = fl[:6]
    embw = fl[6] if len(fl) > 6 else 0
    act = (lambda h: np.maximum(h, 0)) if relu else gelu
    X = np.load(f"{prep}/{split}_X.npy", mmap_mode="r"); Wm = np.load(f"{prep}/{split}_W16.npy", mmap_mode="r")
    Phi = np.load(f"{prep}/{split}_Phi.npy", mmap_mode="r")
    own = ck["own"]; v1t = list(range(27, 43))
    # torch path (same as lt2_train.run with batch 1)
    sys.path.insert(0, "lab")
    import lt2_train as T
    net = T.Net(ck["fin"], a["hidden"], a["depth"], 1 + D + ck.get("naux", 0), a["act"]).double()
    net.load_state_dict({k: v.double() for k, v in ck["net"].items()})
    mu, sd = ck["mu"].double(), ck["sd"].double(); mu2, sd2 = ck["mu2"].double(), ck["sd2"].double()
    es = ck["es"].double()
    for k in ks:
        Wk = np.asarray(Wm[k], dtype=np.float64); Xk = np.asarray(X[k], dtype=np.float64); Pk = np.asarray(Phi[k], np.float64)
        # torch
        # standardise in float32 exactly as lt2_train does (constant columns -> exactly 0)
        Z = ((T.tf(torch.from_numpy(np.asarray(X[k])[..., own + v1t])) - mu.float()[:, None, :])
             / sd.float()[:, None, :]).double()
        u = h = None
        for l in range(L):
            Wl = torch.from_numpy(Wk[l])
            parts = [Z[l]]
            if xall:
                parts.append(torch.zeros(N, 2 * Fo, dtype=torch.float64) if l == 0 else
                             (torch.cat([Wl.T @ Z[l - 1, :, :Fo], (Wl * Wl).T @ Z[l - 1, :, :Fo]], -1) - mu2[l]) / sd2[l])
            if D:
                parts.append(torch.zeros(N, (1 if embw else 2) * D, dtype=torch.float64) if l == 0 else
                             (Wl.T @ h if embw else torch.cat([Wl.T @ h, (Wl * Wl).T @ h], -1)))
            if uin:
                if l == 0:
                    parts.append(torch.zeros(N, 3, dtype=torch.float64))
                else:
                    av = (u @ Wl) / es[l]; cv = ((u / es[l - 1]) ** 2) @ (Wl * Wl)
                    parts.append(torch.stack([av, torch.from_numpy(Pk[l]) * av, cv], -1))
            parts.append(torch.eye(L, dtype=torch.float64)[l].expand(N, L))
            o = net(torch.cat(parts, -1)).detach()
            g = o[:, 0] * es[l]
            if D:
                h = torch.tanh(o[:, 1:1 + D])
            u = g if l == 0 else torch.from_numpy(Pk[l]) * (u @ Wl) + g
        ut = u.numpy()
        # numpy estimator path (natural order; W @ x in estimator space == x @ W_natural)
        st = None
        for l in range(L):
            Wl = Wk[l]; WW = Wl * Wl
            Z0 = tf(Xk[l][:, own]); Zv = tf(Xk[l][:, v1t])     # dump v1t columns are the raw W/WW prev8
            Zs0 = (Z0 - d["mz"][l][:Fo]) * d["iz"][l][:Fo]; Zsv = (Zv - d["mz"][l][Fo:]) * d["iz"][l][Fo:]
            parts = [Zs0, Zsv]
            if xall:
                parts.append(np.zeros((N, 2 * Fo)) if st is None else
                             (np.concatenate([Wl.T @ st["zo"], WW.T @ st["zo"]], 1) - d["m2"][l]) * d["i2"][l])
            if D:
                parts.append(np.zeros((N, (1 if embw else 2) * D)) if st is None else
                             (Wl.T @ st["h"] if embw else np.concatenate([Wl.T @ st["h"], WW.T @ st["h"]], 1)))
            Wu = None if st is None else st["u"] @ Wl
            if uin:
                parts.append(np.zeros((N, 3)) if st is None else np.stack([Wu, Pk[l] * Wu, (st["u"] ** 2) @ WW], 1))
            x = np.concatenate(parts, 1)
            W1l = d["W1"].astype(np.float64).copy()
            if uin:
                W1l[-3:] *= d["usc"][l][:, None]
            hh = act(x @ W1l + d["b1"][l])
            for i in range(nm):
                hh = act(hh @ d[f"Wm{i}"] + d[f"bm{i}"])
            o = hh @ d["Wo"] + d["bo"]
            g = o[:, 0] * d["es"][l]
            u2 = g if st is None else Pk[l] * Wu + g
            st = {"u": u2, "zo": Zs0}
            if D:
                st["h"] = np.tanh(o[:, 1:])
        un = st["u"]
        rel = np.abs(un - ut).max() / np.abs(ut).max()
        print(f"{split} {k}: max|u_np - u_torch| / max|u| = {rel:.2e}  rms u {np.sqrt((ut ** 2).mean()):.3e}")


if __name__ == "__main__":
    main()
