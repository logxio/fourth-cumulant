"""Accuracy probe: start generating K4 groups at zero-based layer s0.

This is the exact research K3/K4 recurrence, not the compressed B estimator.
The prefix retains the degree-four nonlinear formulas with an empty K4 bank.
No truth is used by the recurrence or by a basis.  Truth is read for scoring.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for folder in ('tools', 'refs/mlp_cumulant_propagation/src', 'lab'):
    sys.path.insert(0, str(ROOT / folder))
import numpy as np
import torch
import k4full as f
from mlpdata import load_truth, weights_from_seed


def run(weights, s0, spec, device, dtype):
    n = weights[0].shape[0]
    K = f.coerce_input({1: torch.zeros(n, device=device, dtype=dtype),
                        2: torch.eye(n, device=device, dtype=dtype)},
                       k_max=4, kind=f.Kind.SIMPLE)
    saved_spec = f.SPEC
    means, layers = [], []
    try:
        for layer, weight in enumerate(weights):
            f.SPEC = f.parse_spec(spec) if layer >= s0 else []
            f.LAST_LAYER = layer == len(weights)-1
            w = torch.as_tensor(weight.T.copy(), device=device, dtype=dtype)
            t0 = time.monotonic()
            K = f.linear_kprop(K, w, k_max=4, set_metric=2.0)
            K = f.nonlin(K, nonlin_wick_coef=f.WICK_COEF_D['relu'])
            means.append(K[1].to_tensor().double().cpu().numpy())
            row = dict(layer=layer, groups=len(K[4].groups),
                       seconds=time.monotonic()-t0, memory=f._mem())
            layers.append(row)
            print(json.dumps(row), flush=True)
    finally:
        f.SPEC = saved_spec
    return np.stack(means), layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index', type=int, default=0)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--spec', default='star,three2,slice4+slice31,!slice22+slice211')
    p.add_argument('--device', default='cuda')
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--layers', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.width == 1024:
        truth = load_truth('mini')
        weights = weights_from_seed(truth['mlp_seed'][a.index])[:a.layers]
        target = truth['all_layer_means'][a.index, :a.layers].astype(float)
        dtype = torch.float32
    else:
        rng = np.random.default_rng(5)
        weights = [(rng.standard_normal((a.width, a.width))*(2/a.width)**.5).astype('f')
                   for _ in range(a.layers)]
        target = None
        dtype = torch.float64
    t0 = time.monotonic()
    pred, rows = run(weights, a.start, a.spec, a.device, dtype)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output.with_suffix('.npy'), pred)
    result = dict(index=a.index, start=a.start, spec=a.spec, width=a.width,
                  prefix='degree4 nonlinear recurrence with empty K4 bank; not B',
                  layers=rows, seconds=time.monotonic()-t0, memory=f._mem())
    if target is not None:
        result['per_layer_mse'] = np.mean((pred-target)**2, axis=1).tolist()
        result['mse'] = result['per_layer_mse'][-1]
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
