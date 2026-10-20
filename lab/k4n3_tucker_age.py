"""Age-gated three-axis Tucker probe for the K4 (2,1,1) feedback slice.

The latest ``m0f + 1`` births feed K3 exactly. Older births are projected
together at the point where the reference reads their transported diagonal
slice. The reference still carries dense groups, so this measures accuracy,
not actual running cost or memory savings.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

assert os.environ.get("AW_SPEC") == "star,three2,slice4+slice31,!slice22+slice211"
import k4full


def mode_product(tensor: torch.Tensor, basis: torch.Tensor, axis: int) -> torch.Tensor:
    """Contract one tensor axis with basis.T, leaving that axis in place."""
    out = torch.tensordot(basis.T, tensor, dims=([1], [axis]))
    return out.movedim(0, axis)


def reconstruct(core: torch.Tensor, basis: torch.Tensor, axis: int) -> torch.Tensor:
    out = torch.tensordot(basis, core, dims=([1], [axis]))
    return out.movedim(0, axis)


def shared_basis(tensor: torch.Tensor, old: torch.Tensor | None, rank: int) -> torch.Tensor:
    n = tensor.shape[0]
    gram = torch.zeros((n, n), device=tensor.device, dtype=tensor.dtype)
    for axis in range(3):
        unfold = tensor.movedim(axis, 0).contiguous().reshape(n, -1)
        gram.addmm_(unfold, unfold.T)
        del unfold
    if old is not None and old.numel():
        # Treat the transported old subspace as one additional observation of
        # the same scale as one mode of the newly born three-axis tensor.
        old, _ = torch.linalg.qr(old, mode="reduced")
        weight = gram.trace() / (3 * old.shape[1])
        gram.addmm_(old, old.T, beta=1, alpha=float(weight))
    gram = (gram + gram.T) * 0.5
    _, vectors = torch.linalg.eigh(gram)
    return vectors[:, -rank:].contiguous()


def run_chain(weights: list[np.ndarray], rank: int, device: str,
              *, m0f: int, m0: int = 2, old_rank: int = 128,
              layers: int = 0) -> tuple[np.ndarray, list[float]]:
    n, length = weights[0].shape[0], layers or len(weights)
    dt = torch.float32
    mlp = k4full.MLP(hidden_dim=n, num_layers=len(weights) + 1,
                     nonlin="relu", init_kind="manual", w_var=2.0).to(device=device, dtype=dt)
    for i, weight in enumerate(weights):
        mlp.Ws[i].weight.data = torch.from_numpy(weight.T.copy()).to(
            device=device, dtype=dt).contiguous()
    cumulants = k4full.coerce_input(
        {1: torch.zeros(n, device=device, dtype=dt),
         2: torch.eye(n, device=device, dtype=dt)},
        k_max=4, kind=k4full.Kind.SIMPLE)
    wick = k4full.WICK_COEF_D["relu"]
    original_dslices = k4full.SK4._dslices
    birth: dict[int, int] = {}
    means, rel = [], []
    prior_q: torch.Tensor | None = None
    transported_q: torch.Tensor | None = None
    layer = -1
    projected_layer = -1
    projected_q: torch.Tensor | None = None

    def aged_dslices(self, groups):
        nonlocal prior_q, projected_layer, projected_q
        old = [g for g in groups if "slice211" in g.types
               and layer - birth.get(id(g.Bb), layer) > m0f]
        if not old or rank >= n:
            return original_dslices(self, groups)
        young = [g for g in groups if g not in old]
        fresh = original_dslices(self, young)
        aged = original_dslices(self, old)
        slice211 = aged[(2, 1, 1)]
        if projected_layer != layer:
            projected_q = shared_basis(slice211, transported_q, rank)
            projected_layer = layer
        assert projected_q is not None
        core = slice211
        for axis in range(3):
            core = mode_product(core, projected_q, axis)
        projected = core
        for axis in range(3):
            projected = reconstruct(projected, projected_q, axis)
        projected = k4full.zero_repeated(projected)
        rel[layer] = float((projected - slice211).norm() /
                           slice211.norm().clamp_min(1e-30))
        aged[(2, 1, 1)] = projected
        prior_q = projected_q
        return {part: fresh[part] + aged[part] for part in fresh}

    k4full.SK4._dslices = aged_dslices
    try:
        for layer in range(length):
            k4full.LAST_LAYER = layer == length - 1
            rel.append(0.0)
            weight = mlp.Ws[layer].weight
            transported_q = weight @ prior_q if prior_q is not None else None
            cumulants = k4full.linear_kprop(cumulants, weight,
                                            k_max=4, set_metric=mlp.init_scale[layer])
            if transported_q is not None:
                mean = cumulants[1].core
                var = cumulants[2].core.diag()
                w1 = wick(mean=mean, var=var, k=1, p=1)
                transported_q = w1[:, None] * transported_q
            cumulants = k4full.nonlin(cumulants, nonlin_wick_coef=wick)
            means.append(cumulants[1].to_tensor().double().cpu().numpy())
            groups = cumulants[4].groups
            for group in groups:
                birth.setdefault(id(group.Bb), layer)
            old = [group for group in groups
                   if group.types[0] in ("star", "three2")
                   and layer - birth[id(group.Bb)] > m0]
            if old and old_rank < n:
                unique = {birth[id(group.Bb)]: group.Q for group in old}
                sketch = weight[:, :min(old_rank, n)].contiguous()
                for _ in range(2):
                    image = sum((q @ (q.T @ sketch) for q in unique.values()),
                                torch.zeros_like(sketch))
                    sketch, _ = torch.linalg.qr(image, mode="reduced")
                for group in old:
                    group.Q = sketch @ (sketch.T @ group.Q)
    finally:
        k4full.SK4._dslices = original_dslices
    assert len(rel) == length, (len(rel), length)
    return np.stack(means), rel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--m0f", type=int, required=True)
    parser.add_argument("--m0", type=int, default=2)
    parser.add_argument("--old-rank", type=int, default=128)
    parser.add_argument("--official-index", type=int, default=0)
    parser.add_argument("--synthetic-n", type=int, default=0)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--layers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.synthetic_n:
        n = args.synthetic_n
        rng = np.random.default_rng(args.seed)
        weights = [(rng.standard_normal((n, n)) * (2 / n) ** .5).astype("f")
                   for _ in range(args.layers or 16)]
        target = None
        index = f"seed{args.seed}"
    else:
        from mlpdata import load_truth, weights_from_seed
        truth = load_truth("mini")
        weights = weights_from_seed(truth["mlp_seed"][args.official_index])
        target = truth["all_layer_means"][args.official_index].astype(np.float64)
        index = f"mlp{args.official_index}"
    start = time.monotonic()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    pred, rel = run_chain(weights, args.rank, args.device,
                          m0f=args.m0f, m0=args.m0, old_rank=args.old_rank,
                          layers=args.layers)
    receipt = dict(rank=args.rank, m0f=args.m0f, m0=args.m0, old_rank=args.old_rank,
                   width=weights[0].shape[0], index=index, layers=len(pred),
                   projection_rel=rel, max_projection_rel=max(rel),
                   elapsed_s=time.monotonic()-start, memory=k4full._mem(),
                   spec=os.environ["AW_SPEC"])
    if target is not None:
        errors = np.mean((pred - target[:len(pred)]) ** 2, axis=1)
        receipt.update(final_mse=float(errors[-1]), per_layer_mse=errors.tolist(),
                       relative_to_3_90e_9=float(errors[-1] / 3.90e-9))
    if args.self_check:
        baseline = k4full.run_chain(weights, args.device, torch.float32,
                                    layers=args.layers)
        differences = np.max(np.abs(pred - baseline), axis=1)
        receipt["per_layer_max_abs_vs_reference"] = differences.tolist()
        if args.m0f >= len(pred) or args.rank >= weights[0].shape[0]:
            assert np.array_equal(pred, baseline), differences.max()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = f"tucker_age{args.m0f}_k{args.rank}_{index}"
    np.save(output / f"{stem}.npy", pred)
    (output / f"{stem}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
