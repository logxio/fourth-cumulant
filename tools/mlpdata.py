"""Rebuild public v2-phase2 MLPs from their seeds (whestbench seed protocol 3.0).

ss = SeedSequence(mlp_seed).spawn(3): ss[0] weights, ss[1] ground-truth samples,
ss[2].generate_state(1)[0] = mlp.seed handed to the estimator.
"""
import numpy as np

WIDTH, DEPTH = 1024, 16


def weights_from_seed(mlp_seed: int, width: int = WIDTH, depth: int = DEPTH) -> list[np.ndarray]:
    weight_ss, _, _ = np.random.SeedSequence(int(mlp_seed)).spawn(3)
    rng = np.random.default_rng(weight_ss)
    scale = float(np.sqrt(2.0 / width))
    return [(rng.standard_normal((width, width)) * scale).astype(np.float32) for _ in range(depth)]


def estimator_seed(mlp_seed: int) -> int:
    _, _, est_ss = np.random.SeedSequence(int(mlp_seed)).spawn(3)
    return int(est_ss.generate_state(1)[0])


def load_truth(split: str = "mini"):
    import os
    root = os.environ.get("AW_DATA", "data")
    return np.load(f"{root}/{split}_truth.npz")
