"""Export an lt2_train.py model to the estimator's lt2_weights.npz (standardisation, layer one-hot, es folded).

usage: lt2_export.py <lt2_model.pt> <out.npz>
Per layer l the estimator forms x = [tf(raw own|v1t) | xall | emb | uin] and computes
  h = act(x @ W1_l + b1[l]); h = act(h @ Wk + bk) (depth-1 times); g = es_l * (h @ Wo[:, 0] + bo[0]),
  emb state = tanh(h @ Wo[:, 1:] + bo[1:]), u = Phi * (W u_prev) + g; W1_l = W1 with the 3 uin rows scaled by usc[l].
z = (tf(raw) - mz[l]) * iz[l]; xall inputs are the previous layer's own part of z carried through W and W*W,
then (t - m2[l]) * i2[l]; uin inputs are raw W u_prev, Phi * W u_prev, WW (u_prev^2) (1/es folded into W1).
"""
import sys
import numpy as np
import torch


def main(src, dst):
    ck = torch.load(src, map_location="cpu")
    a = ck["args"]; sd_ = {k: v.double().numpy() for k, v in ck["net"].items()}
    mu, sd = ck["mu"].double().numpy(), ck["sd"].double().numpy()          # (L, C)
    mu2, sd2 = ck["mu2"].double().numpy(), ck["sd2"].double().numpy()      # (L, 2Fo)
    es = ck["es"].double().numpy()
    L = mu.shape[0]; C = mu.shape[1]; Fo = len(ck["own"]); D = int(a["emb"])
    lin = sorted({int(k.split(".")[1]) for k in sd_ if k.startswith("f.") and k.endswith(".weight")})
    W1 = sd_[f"f.{lin[0]}.weight"]; b1 = sd_[f"f.{lin[0]}.bias"]          # (H, fin)
    Wo = sd_[f"f.{lin[-1]}.weight"][:1 + D]; bo = sd_[f"f.{lin[-1]}.bias"][:1 + D]   # (1+D, H); aux outputs dropped
    mids = lin[1:-1]
    # standardisation is applied elementwise in the estimator (folding 1/sd of the tiny-variance columns into W1
    # cancels catastrophically in float32); columns at the sd floor are constant within the layer -> weight 0
    iz = np.where(sd > 1.01e-6, 1.0 / sd, 0.0); i2 = np.where(sd2 > 1.01e-6, 1.0 / sd2, 0.0)
    # one shared W1 / Wo; per layer only the bias (with the layer one-hot), the uin row scalings and es differ
    embw = bool(a.get("embw", False))
    c = C + (2 * Fo if a["xall"] else 0) + (D if embw else 2 * D)
    usc = np.zeros((L, 3))
    b1L = []
    for l in range(L):
        if a["uin"]:
            usc[l] = [1.0 / es[l], 1.0 / es[l], 1.0 / es[l - 1] ** 2 if l > 0 else 0.0]
        b1L.append(b1 + W1[:, c + (3 if a["uin"] else 0) + l])
    fin = c + (3 if a["uin"] else 0)
    out = dict(W1=W1[:, :fin].T.astype(np.float32), b1=np.stack(b1L).astype(np.float32), usc=usc.astype(np.float32),
               Wo=Wo.T.astype(np.float32), bo=bo.astype(np.float32), es=es.astype(np.float32),
               mz=mu.astype(np.float32), iz=iz.astype(np.float32), m2=mu2.astype(np.float32), i2=i2.astype(np.float32),
               flags=np.array([Fo, int(bool(a["xall"])), D, int(bool(a["uin"])), len(mids),
                               0 if a["act"] == "gelu" else 1, int(embw)], dtype=np.int64))
    for i, m in enumerate(mids):
        out[f"Wm{i}"] = sd_[f"f.{m}.weight"].T.astype(np.float32)
        out[f"bm{i}"] = sd_[f"f.{m}.bias"].astype(np.float32)
    np.savez(dst, **out)
    print("exported", dst, {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
