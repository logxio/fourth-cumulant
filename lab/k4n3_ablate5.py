"""Official-width reference-chain ablation of fourth-order feedback channels.

AW_SPEC is fixed before importing k4full.  The output is a prediction file and
one JSON receipt per case; every score is recomputed from the official truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

os.environ.setdefault("AW_SPEC", "star,three2,slice4+slice31,!slice22+slice211")
import k4full


def graph_class(vectors):
    """Classify the actual reference vector partition, without merging powers."""
    triples = [v for v in vectors if sum(x != 0 for x in v) == 3]
    if triples:
        assert len(triples) == 1
        degree = sum(triples[0])
        if degree == 4:
            return "k4_core"
        assert degree == 3
        return "k3_core_edge" if len(vectors) > 1 else "k3_core"
    edges = [v for v in vectors if sum(x != 0 for x in v) == 2]
    if len(edges) == 1:
        return "edge_k" + str(sum(edges[0]))
    if len(edges) == 2:
        peak = max(sum(v) for v in edges)
        return {2: "tree2_c", 3: "tree2_k3", 4: "tree2_k4"}[peak]
    if len(edges) == 3:
        pairs = [tuple(i for i, value in enumerate(v) if value) for v in edges]
        if len(set(pairs)) == 3:
            return "triangle_c"
        return "parallel3_c"
    raise ValueError(f"unclassified pK(2,1,1) term: {vectors}")


def catalog():
    from mlp_kprop.factor_k4 import factored_keeps_term
    all_terms = k4full.nonlin.__globals__["get_all_terms_iso"](
        k_max=4, d_max=4, augment=False)
    rows = []
    for vectors, multiplicity in all_terms[(2, 1, 1)].items():
        if factored_keeps_term(4, (2, 1, 1), vectors):
            rows.append(dict(group=graph_class(vectors), multiplicity=multiplicity,
                             vectors=vectors))
    return rows


def run_ablation(weights, device, rank, dropped):
    ns = k4full.nonlin.__globals__
    original_terms = ns["get_all_terms_iso"]
    original_eval = ns["eval_part"]
    original_from_ds = k4full.SK4.from_dstensor
    triangle = ((0, 1, 1), (1, 0, 1), (1, 1, 0))

    def filtered_terms(*args, **kwargs):
        terms = original_terms(*args, **kwargs)
        out = {part: dict(items) for part, items in terms.items()}
        for part, name in (((4,), "4"), ((3, 1), "31"),
                           ((2, 2), "22"), ((2, 1, 1), "211")):
            if name in dropped:
                # Keep a zero-valued slice so the reference pK->K conversion
                # and SK4.from_dstensor retain their expected partition keys.
                first = next(iter(out[part]))
                out[part] = {first: 0}
        if "211" not in dropped:
            out[(2, 1, 1)] = {
                vectors: count for vectors, count in out[(2, 1, 1)].items()
                if "211:" + graph_class(vectors) not in dropped
            }
        return out

    def triangle_ranked(cumulants, vectors, dim, output_zero_repeated=True):
        if dim != 3 or vectors != triangle:
            return original_eval(cumulants, vectors, dim, output_zero_repeated)
        if rank == 0:
            return None
        from mlp_kprop.diagslice import zero_repeated
        cov = zero_repeated(cumulants[2].core)
        eig, basis = torch.linalg.eigh(cov)
        keep = torch.argsort(eig.abs(), descending=True)[:rank]
        approx = zero_repeated((basis[:, keep] * eig[keep]) @ basis[:, keep].T)
        return approx[None, :, :] * approx[:, None, :] * approx[:, :, None]

    def from_ds_with_empty_slices(ds):
        for part in k4full.PARTS:
            if part not in ds.slices:
                ds.slices[part] = torch.zeros((ds.n,) * len(part),
                                               device=ds.device, dtype=ds.dtype)
        return original_from_ds(ds)

    ns["get_all_terms_iso"] = filtered_terms
    ns["eval_part"] = triangle_ranked
    k4full.SK4.from_dstensor = staticmethod(from_ds_with_empty_slices)
    try:
        return k4full.run_chain(weights, device, torch.float32)
    finally:
        ns["get_all_terms_iso"] = original_terms
        ns["eval_part"] = original_eval
        k4full.SK4.from_dstensor = staticmethod(original_from_ds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--official-index", type=int, default=0)
    p.add_argument("--drop", default="", help="comma-separated: 4,31,22,211,211:CLASS")
    p.add_argument("--case", default="baseline")
    p.add_argument("--triangle-rank", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="outputs/research/k4n3-ablate5")
    args = p.parse_args()
    rows = catalog()
    groups = sorted(set(row["group"] for row in rows))
    if args.list:
        print(json.dumps(dict(groups=dict(Counter(row["group"] for row in rows)),
                              rows=rows), indent=2))
        return
    dropped = set(filter(None, args.drop.split(",")))
    valid = {"4", "31", "22", "211"} | {"211:" + g for g in groups}
    unknown = dropped - valid
    if unknown:
        p.error(f"unknown drops: {sorted(unknown)}")
    from mlpdata import load_truth, weights_from_seed
    torch.set_num_threads(4)
    truth = load_truth("mini")
    weights = weights_from_seed(truth["mlp_seed"][args.official_index])
    target = truth["all_layer_means"][args.official_index].astype(np.float64)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.monotonic()
    prediction = run_ablation(weights, args.device, args.triangle_rank, dropped)
    per_layer = np.mean((prediction - target[:len(prediction)]) ** 2, axis=1)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{args.case}_mlp{args.official_index}.npy", prediction)
    receipt = dict(case=args.case, drop=sorted(dropped), index=args.official_index,
                   triangle_rank=args.triangle_rank, width=weights[0].shape[0],
                   layers=len(prediction), final_mse=float(per_layer[-1]),
                   per_layer_mse=per_layer.tolist(), elapsed_s=time.monotonic()-t0,
                   memory=k4full._mem(), spec=os.environ["AW_SPEC"])
    path = output / f"{args.case}_mlp{args.official_index}.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
