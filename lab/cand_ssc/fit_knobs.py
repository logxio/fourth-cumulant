"""Gauss-Newton fit of SSC knob deltas on final-layer MSE from finite-difference runs.

usage: fit_knobs.py <pert.json> <truth_src> <run npz ...> [--ridge r] [--layers final|all]
  pert.json : {knob_or_column: delta} used to build the perturbed cfgs (same order as cfgs)
  truth_src : mc npz with keys s<seed> (small-width MC) or 'inrun' (truth stored in the run npz)
Each run npz holds preds (nets, cfgs, layers, n), including the final layer,
with cfg 'base' first or merged by net name. A final-layer-only archive suffices.
Prints per-knob 1-D optimum, the joint ridge fit, its in-sample and leave-one-net-out gains.
"""
import argparse, json
import numpy as np


def load_runs(paths):
    by_net = {}
    for p in paths:
        z = np.load(p)
        for ni, net in enumerate(z["nets"]):
            d = by_net.setdefault(str(net), {})
            for ci, c in enumerate(z["cfgs"]):
                d[str(c)] = z["preds"][ni, ci]
            if "truth" in z.files and np.any(z["truth"][ni]):
                d["__truth"] = z["truth"][ni]
    return by_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pert"); ap.add_argument("truth"); ap.add_argument("runs", nargs="+")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--layers", default="final")
    ap.add_argument("--cfgdir", default="jc")
    ap.add_argument("--center", default="base")
    a = ap.parse_args()
    pert = json.load(open(a.pert))
    names = list(pert)
    runs = load_runs(a.runs)
    mc = None if a.truth == "inrun" else np.load(a.truth)
    nets, P0, J, T = [], [], [], []
    for net, d in sorted(runs.items()):
        cols = [f"{a.cfgdir}/{k}.json" for k in names]
        if a.center not in d or any(c not in d for c in cols):
            continue
        if mc is not None:
            key = "s" + net.replace("seed", "")
            if key not in mc.files:
                continue
            t = mc[key]
        else:
            t = d["__truth"]
        sl = slice(-1, None) if a.layers == "final" else slice(None)
        p0 = d[a.center][sl].ravel()
        Jn = np.stack([(d[c][sl].ravel() - p0) / pert[k] for c, k in zip(cols, names)], axis=1)
        nets.append(net); P0.append(p0); J.append(Jn); T.append(t[sl].ravel())
    m = len(nets)
    print(f"{m} nets: {nets}")
    if m == 0:
        return
    base = np.array([np.mean((p - t) ** 2) for p, t in zip(P0, T)])
    print(f"base mean MSE {base.mean():.4e}")
    # per-knob 1-D optimum (pooled over nets, normalized per net by its base MSE)
    for j, k in enumerate(names):
        num = sum(J[i][:, j] @ (T[i] - P0[i]) / base[i] for i in range(m))
        den = sum(J[i][:, j] @ J[i][:, j] / base[i] for i in range(m))
        th = num / den
        new = np.array([np.mean((P0[i] + J[i][:, j] * th - T[i]) ** 2) for i in range(m)])
        print(f"  {k:9s} 1-D opt delta {th:+.4f}  pred MSE ratio {np.mean(new / base):.4f}  (pert {pert[k]})")

    def solve(idx):
        A = sum(J[i].T @ J[i] / base[i] for i in idx)
        b = sum(J[i].T @ (T[i] - P0[i]) / base[i] for i in idx)
        sc = np.sqrt(np.diag(A)) + 1e-30
        An = A / sc[:, None] / sc[None, :]
        th = np.linalg.solve(An + a.ridge * len(idx) * np.eye(len(names)), b / sc) / sc
        return th

    th = solve(range(m))
    new = np.array([np.mean((P0[i] + J[i] @ th - T[i]) ** 2) for i in range(m)])
    print("joint fit deltas:", {k: round(float(v), 4) for k, v in zip(names, th)})
    print(f"in-sample pred MSE ratio {np.mean(new / base):.4f}  mean {new.mean():.4e}")
    if m >= 3:
        lo = []
        for i in range(m):
            thi = solve([q for q in range(m) if q != i])
            lo.append(np.mean((P0[i] + J[i] @ thi - T[i]) ** 2) / base[i])
        print(f"leave-one-out pred MSE ratio {np.mean(lo):.4f}  per net {np.round(lo, 3)}")
    json.dump({k: float(v) for k, v in zip(names, th)}, open("fit_deltas.json", "w"))


if __name__ == "__main__":
    main()
