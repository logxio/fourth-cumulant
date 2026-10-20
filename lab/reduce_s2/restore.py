import sys,json,math
from pathlib import Path
import numpy as np
r=Path(sys.argv[1])
if (r/'metrics.json').exists():
    m=json.loads((r/'metrics.json').read_text());assert len(m['samples'])==10
    for i in range(10):
        a=np.load(r/f'case{i}.npz');assert all(np.isfinite(a[k]).all() for k in a.files)
    print('10 kernel predictions restored');sys.exit(0)
t=np.load(r/'data/mini_truth.npz')
checked=[]
for p in (r/'outputs').rglob('*.json'):
    m=json.loads(p.read_text())
    if 'mse' not in m:continue
    pred=np.load(p.with_suffix('.npy'))
    raw=float(np.mean((pred[-1].astype(float)-t['all_layer_means'][m['index'],-1])**2))
    assert math.isclose(raw,m['mse'],rel_tol=1e-9),(p,raw,m['mse'])
    assert np.isfinite(pred).all()
    checked.append(str(p.relative_to(r)))
assert checked,'no scored outputs'
print(json.dumps({'rescored':checked}))
