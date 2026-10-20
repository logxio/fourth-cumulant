"""Ten small kernels: dense/pair parity, output package, receiver and readback."""
import os, sys, json, tarfile, tempfile, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
os.environ['AW_SPEC']='star,three2,slice4+slice31,!slice22+slice211'
for p in (ROOT, ROOT/'lab', ROOT/'tools', ROOT/'refs/mlp_cumulant_propagation/src'):
    sys.path.insert(0,str(p))
import numpy as np
import torch
import k4full as f
from lab.reduce_s2 import pairs
torch.set_num_threads(2)
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
rows=[]
for seed in range(10):
    torch.manual_seed(seed)
    n=12+seed
    B=torch.randn(n,n,n,dtype=torch.float64); B=(B+B.transpose(0,1))/2
    Q=torch.randn(n,n,dtype=torch.float64)/n**0.5
    X=None if seed%2==0 else torch.randn(n,n,dtype=torch.float64)
    g=f.Group(('slice211',) if X is None else ('three2',),bool(seed%3),X,None,B,Q)
    # Compression is at birth, followed by independently selected transport.
    g.Q=torch.eye(n,dtype=torch.float64)
    c=pairs.compress(g,n,'nontree' if X is None else 'ck3',4)
    g.Q=Q;c.Q=Q
    ref={p:torch.zeros((n,)*len(p),dtype=torch.float64) for p in f.PARTS}
    got={p:torch.zeros_like(x) for p,x in ref.items()}
    f.group_dslices(g,ref);pairs.read(c,got)
    err=max(float((got[p]-ref[p]).abs().max()) for p in ref)
    scale=max(float(t.abs().max()) for t in ref.values())
    assert err<1e-11*max(1,scale),(seed,err,scale)
    np.savez(out/f'case{seed}.npz',**{str(p):got[p].numpy() for p in got})
    rows.append(dict(seed=seed,n=n,max_abs=err,relative=err/max(1,scale)))
(out/'metrics.json').write_text(json.dumps({'samples':rows,'max_relative':max(r['relative'] for r in rows)},indent=2))
(out/'config.json').write_text(json.dumps({'kind':'10-kernel exact full-rank transport/read smoke','samples':10}))
shutil.copy2(__file__,out/'smoke.py')
with tempfile.TemporaryDirectory() as td:
    arc=Path(td)/'outputs.tar.gz'
    names={f'case{i}.npz' for i in range(10)}|{'metrics.json','config.json','smoke.py'}
    with tarfile.open(arc,'w:gz') as t:
        for name in sorted(names): t.add(out/name,arcname=name)
    recv=Path(td)/'received';recv.mkdir()
    with tarfile.open(arc) as t:
        assert all(m.name in {f'case{i}.npz' for i in range(10)}|{'metrics.json','config.json','smoke.py'} for m in t.getmembers())
        t.extractall(recv,filter='data')
    for i in range(10):
        a=np.load(out/f'case{i}.npz');b=np.load(recv/f'case{i}.npz')
        for k in a.files: assert np.array_equal(a[k],b[k])
print(json.dumps({'samples':10,'max_relative':max(r['relative'] for r in rows),'pack_receive_readback':'passed'}))
