"""End-to-end compact K=4 chain from layer s0 on one official 1024 x 16 network (end-to-end study).

Prefix: chain B (#335301 lineage, lab/cand_k4x/base_b.py) to the layer-4 pre-activation, metered by
lab/cand_k4x/r4/prefix_meter.py, which writes the snapshot B_mini<i>_pre4.npz and its units.
Suffix (this file): chain B's K3 factors expanded once into a dense kappa3, chain B's K4 slices as the
seed group, then the reference K=4 recursion with the K4 state held compactly (compact.py).

With --meter every torch op of the suffix executes through flopscope.numpy (fsmode.FlopscopeMode)
inside one flopscope.BudgetContext, so the units are flopscope's own count.

usage: chain.py --index 0 --snapshots DIR --out DIR [--pair-rank 8] [--core-rank 128] [--m0 2] [--meter]
"""
import argparse, json, os, sys, time, resource, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for sub in ("tools", "refs/mlp_cumulant_propagation/src", "lab", "lab/cand_k4x", "lab/e2e_s1"):
    sys.path.insert(0, str(ROOT / sub))
import torch
torch.set_grad_enabled(False)

SPEC = "star+three2,slice4+slice31,!slice22+slice211"
UNIT = 2 ** 31


def rss_gib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair-rank", type=int, default=8)
    p.add_argument("--pair-iters", type=int, default=2)
    p.add_argument("--core-rank", type=int, default=128)
    p.add_argument("--m0", type=int, default=2)
    p.add_argument("--slice-mode", choices=("pairs", "dense"), default="pairs")
    p.add_argument("--feedback-rank", type=int, default=0, help="r4 old-feedback k3 (0 = off)")
    p.add_argument("--m0f", type=int, default=1)
    p.add_argument("--meter", action="store_true")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--layers", type=int, default=16)
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    import compact as cm
    kf = cm.kf
    import b_suffix
    from mlpdata import load_truth, weights_from_seed
    cm.CFG.update(pair_rank=a.pair_rank, pair_iters=a.pair_iters, core_rank=a.core_rank, m0=a.m0,
                  slice_mode=a.slice_mode)
    config = dict(index=a.index, start=a.start, spec=SPEC, seed_spec="slice4+slice31+slice22+slice211",
                  pair_rank=a.pair_rank, pair_iters=a.pair_iters, pair_over=cm.CFG["pair_over"],
                  core_rank=a.core_rank, m0=a.m0, dtype="float32", meter=a.meter, layers=a.layers,
                  old_source_rule="two-leg groups with layer-birth > m0 enter one shared k-basis core (r4 sketch rule)",
                  slice_mode=a.slice_mode, feedback_rank=a.feedback_rank, m0f=a.m0f,
                  slice211_rule=("(2,1,1) birth kernel compressed along its repeated-index axis, rank pair_rank, kept for life"
                                 if a.slice_mode == "pairs" else "groups holding slice211 keep the dense n^3 birth kernel (k4full)"),
                  feedback_rule=(f"r4 old-feedback projection: slice211 groups older than m0f read on a k3={a.feedback_rank} Tucker basis"
                                 if a.feedback_rank else "no old-feedback projection"))
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    truth = load_truth("mini")
    weights = weights_from_seed(truth["mlp_seed"][a.index])
    snap_path = a.snapshots / f"B_mini{a.index}_pre{a.start}.npz"
    snapshot = np.load(snap_path)
    means = [np.asarray(x, dtype=np.float64) for x in snapshot["prefix"]]
    assert len(means) == a.start
    rows = []

    if a.meter:
        import flopscope as flops
        from fsmode import FlopscopeMode
        ctx = flops.BudgetContext(flop_budget=10 ** 18, quiet=True)
        mode = FlopscopeMode()
        ctx.__enter__()
        mode.__enter__()
        used = lambda: ctx.flops_used / UNIT
        cm.METER = used
    else:
        ctx = mode = None
        used = lambda: float("nan")
    t_start = time.time()
    try:
        phase_units = {}
        u0 = used()
        mu, cov, k3, bridge = b_suffix.expand(snapshot, device="cpu")
        K = kf.coerce_input({1: mu, 2: cov, 3: k3}, k_max=4, kind=kf.Kind.SIMPLE)
        del mu, cov, k3
        phase_units["B_switch_expand"] = used() - u0
        u0 = used()
        n = 1024
        inv = torch.as_tensor(snapshot["inv_perm"].astype("int64"))

        def unsort(name):
            value = torch.as_tensor(snapshot[name], dtype=torch.float32)
            for axis in range(value.ndim):
                value = value.index_select(axis, inv)
            return value

        cov_seed = unsort("C")
        cov_seed.fill_diagonal_(0)
        g_seed = cov_seed * float(snapshot["lam_feed"])
        d211 = g_seed.unsqueeze(0).expand(n, -1, -1).clone() / 3
        seed = cm.DSTensor(d=4, n=n, slices={(4,): unsort("g4row"), (3, 1): unsort("wk431").T,
                                             (2, 2): unsort("wk4m"), (2, 1, 1): d211},
                           autozero=True, dtype=torch.float32)
        kf.SPEC = kf.parse_spec(config["seed_spec"])
        weight0 = torch.as_tensor(weights[a.start].T.copy(), dtype=torch.float32)

        def start_vectors(Kt, weight):
            mu = Kt[1].core
            v = mu / torch.linalg.vector_norm(mu)
            return torch.cat([v[:, None], weight[:, : cm.CFG["pair_rank"] + cm.CFG["pair_over"] - 1]], 1)

        cm.BIRTH.update(layer=a.start, start=start_vectors(K, weight0))
        K[4] = cm.CSK4.from_dstensor(seed)
        del seed, d211, g_seed, cov_seed
        phase_units["B_K4_seed"] = used() - u0
        seed_kept = [getattr(g, "kept", None) for g in K[4].groups]
        fp = None
        if a.feedback_rank:
            fp = cm.FeedbackProjection(cm.CSK4, a.feedback_rank, a.m0f, lambda g: g.birth)
            fp.install()
        for layer in range(a.start, a.layers):
            t0 = time.time()
            u_layer = used()
            weight = torch.as_tensor(weights[layer].T.copy(), dtype=torch.float32)
            if layer > a.start:
                K = kf.linear_kprop(K, weight, k_max=4, set_metric=2.0)
            if fp is not None:
                fp.start_layer(layer, weight, K, kf.WICK_COEF_D["relu"])
            u_lin = used()
            last = layer == a.layers - 1
            kf.SPEC = [] if last else kf.parse_spec(SPEC)
            cm.BIRTH.update(layer=layer, start=start_vectors(K, weight))
            cm.STATS.clear()
            K = cm.nonlin(K, nonlin_wick_coef=kf.WICK_COEF_D["relu"])
            u_nl = used()
            proj = cm.project_old(K[4], layer, weight, None)
            u_proj = used()
            m = K[1].to_tensor().double().numpy().copy()
            means.append(m)
            groups = K[4].groups
            row = dict(layer=layer, seconds=time.time() - t0,
                       units_linear=u_lin - u_layer, units_nonlinear=u_nl - u_lin, units_projection=u_proj - u_nl,
                       units_total_so_far=used(), host_peak_rss_gib=rss_gib(),
                       mse=float(np.mean((m - truth["all_layer_means"][a.index, layer]) ** 2)),
                       groups=dict(dense=sum(g.kind == "dense" for g in groups), slice=sum(g.kind == "slice" for g in groups),
                                   core=sum(g.kind == "core" for g in groups)),
                       pair_kept=[g.kept for g in groups if g.kind == "slice" and g.birth == layer and g.kept is not None],
                       feedback_reads=(len([r for r in fp.reads if r["layer"] == layer]) if fp is not None else 0),
                       projection=proj, seconds_units_by_part={k: [round(v[0], 3), round(v[1], 3)] for k, v in cm.STATS.items()})
            rows.append(row)
            (a.out / "per_layer.json").write_text(json.dumps(rows, indent=2) + "\n")
            np.save(a.out / "prediction_partial.npy", np.stack(means))
            print(json.dumps(row), flush=True)
    finally:
        if mode is not None:
            mode.__exit__(None, None, None)
            ctx.__exit__(None, None, None)
    wall = time.time() - t_start
    pred = np.stack(means)
    assert pred.shape == (a.layers, 1024) and np.isfinite(pred).all()
    per = np.mean((pred - truth["all_layer_means"][a.index]) ** 2, axis=1)
    prefix_units = None
    pm = a.snapshots / "prefix_meter.json"
    if pm.exists():
        prefix_units = json.loads(pm.read_text()).get("units")
    suffix_units = used()
    result = dict(config=config, snapshot=str(snap_path), bridge=bridge, seed_pair_kept=seed_kept,
                  final_mse=float(per[-1]), per_layer_mse=per.tolist(),
                  suffix_units=suffix_units, prefix_units=prefix_units,
                  total_units=(suffix_units + prefix_units) if prefix_units is not None else None,
                  c_over_b=((suffix_units + prefix_units) / 1024) if prefix_units is not None else None,
                  phase_units=phase_units, suffix_wall_s=wall, host_peak_rss_gib=rss_gib(),
                  threads=torch.get_num_threads(), layers=rows,
                  op_report=(mode.report() if mode is not None and hasattr(mode, "report") else None))
    np.save(a.out / "prediction.npy", pred)
    (a.out / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("RESULT", json.dumps({k: result[k] for k in ("final_mse", "suffix_units", "prefix_units", "total_units",
                                                        "c_over_b", "suffix_wall_s", "host_peak_rss_gib")}), flush=True)


if __name__ == "__main__":
    main()
