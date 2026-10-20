"""Restore check for a recovered reduce_s1 task root: every scored output re-scored from its own prediction."""
import json
import math
import sys
from pathlib import Path

import numpy as np

r = Path(sys.argv[1])
truth = np.load(r / 'data/mini_truth.npz', allow_pickle=False)
checked = []
for f in sorted((r / 'outputs').rglob('*.json')):
    m = json.loads(f.read_text())
    if not isinstance(m, dict):
        continue
    npy = f.with_suffix('.npy')
    if 'plan' in m and npy.exists():
        p = np.load(npy, allow_pickle=False).astype(float)
        assert p.shape == (16, 1024 if not m.get('synthetic') else m['synthetic']) and np.isfinite(p).all(), f
        if 'mse' in m:
            raw = float(np.mean((p[-1] - truth['all_layer_means'][m['index'], -1]) ** 2))
            assert math.isclose(raw, m['mse'], rel_tol=1e-9), (f, raw, m['mse'])
        checked.append(dict(file=str(f.relative_to(r)), mse=m.get('mse'), units=m.get('units')))
    elif 'case' in m and npy.exists():
        p = np.load(npy, allow_pickle=False).astype(float)
        raw = float(np.mean((p[-1] - truth['all_layer_means'][m['index'], -1]) ** 2))
        assert math.isclose(raw, m['raw'], rel_tol=1e-9), (f, raw, m['raw'])
        checked.append(dict(file=str(f.relative_to(r)), mse=raw, units=m.get('paper_U')))
assert checked, 'no outputs'
print(json.dumps(dict(checked=checked, training_resume='not applicable: inference only')))
