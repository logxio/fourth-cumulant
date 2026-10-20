"""Full-rank matrix pairs exactly reproduce all four same-side slices."""
import json, time, resource
from pathlib import Path
from pair_read import torch, f, PairGroup, pair_slices
torch.set_num_threads(2)
rows=[]
for n in (16,32,64):
    rng=torch.Generator().manual_seed(981+n)
    b=torch.randn(n,n,n,generator=rng,dtype=torch.float64)
    b=(b+b.transpose(1,2))/2
    ix=torch.arange(n)
    b[ix,ix,:]=0;b[ix,:,ix]=0;b[:,ix,ix]=0
    q=torch.randn(n,n,generator=rng,dtype=torch.float64)/n**.5
    original=f.Group(('slice211',),False,None,None,b.permute(1,2,0),q)
    pair=PairGroup(torch.eye(n,dtype=torch.float64),b,q,0)
    x={p:torch.zeros((n,)*len(p),dtype=torch.float64) for p in f.PARTS}
    y={p:torch.zeros_like(v) for p,v in x.items()}
    f.group_dslices(original,x);pair_slices(pair,y)
    errors={str(p):float((x[p]-y[p]).abs().max()) for p in x}
    assert max(errors.values())<1e-10,errors
    times=[]
    one=PairGroup(pair.u[:,:1],b[:1],q,0)
    for _ in range(10):
        for v in y.values():v.zero_()
        t=time.perf_counter();pair_slices(one,y);times.append(time.perf_counter()-t)
    rows.append(dict(n=n,max_abs=errors,rank1_median_seconds=sorted(times)[5]))
result=dict(rows=rows,passed=True,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
Path('outputs/research/k4s11/operator.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
