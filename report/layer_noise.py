"""Per-layer noise of the official 10^9-sample ground truth, official public networks 0-4.

The truth noise at layer l is the neuron-averaged variance of the post-ReLU activation over
Gaussian inputs, divided by 10^9. We estimate that variance with 2^15 inputs per network
(relative error about 1%) and check the convention against the truth's own layer means.
Writes report/data/layer_noise.json. CPU, about 0.5 GB, under a minute per network.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mlpdata import load_truth, weights_from_seed  # noqa: E402

N, BATCH = 2 ** 15, 4096
truth = load_truth("mini")
out = {}
for i in range(5):
    seed = int(truth["mlp_seed"][i])
    W = weights_from_seed(seed)
    rng = np.random.default_rng(1000 + i)
    s1 = np.zeros((16, 1024)); s2 = np.zeros((16, 1024))
    for _ in range(N // BATCH):
        h = rng.standard_normal((BATCH, 1024)).astype(np.float32)
        for l in range(16):
            h = np.maximum(h @ W[l], 0.0)
            s1[l] += h.sum(0, dtype=np.float64); s2[l] += (h.astype(np.float64) ** 2).sum(0)
    mean = s1 / N
    var = s2 / N - mean ** 2
    gt = truth["all_layer_means"][i].astype(np.float64)
    corr = [float(np.corrcoef(mean[l], gt[l])[0, 1]) for l in range(16)]
    out[i] = dict(seed=seed, avg_variance=var.mean(1).tolist(), noise_1e9=(var.mean(1) / 1e9).tolist(),
                  mean_corr_vs_truth=corr, truth_avg_variance_output=float(truth["avg_variance"][i]))
    print(i, "noise L1 %.2e L8 %.2e L16 %.2e" % (out[i]["noise_1e9"][0], out[i]["noise_1e9"][7], out[i]["noise_1e9"][15]),
          "corr min %.4f" % min(corr), "truth avg_var %.4f vs ours %.4f" % (out[i]["truth_avg_variance_output"], var.mean(1)[15]),
          flush=True)
Path(__file__).with_name("data").joinpath("layer_noise.json").write_text(json.dumps(out, indent=1))
