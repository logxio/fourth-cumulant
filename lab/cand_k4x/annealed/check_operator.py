"""Match Monte Carlo's direct contractions to the original reference reader."""
import json
from pathlib import Path
from probe import f,torch
from wick import direct_samples
torch.manual_seed(1110);torch.set_num_threads(2);rows=[]
for n in (16,32,64):
    P=torch.randn(n,n,dtype=torch.float64)/n**.5
    W=torch.randn(n,n,dtype=torch.float64)*(2/n)**.5
    B=torch.randn(n,n,n,dtype=torch.float64)/n
    B=(B+B.transpose(0,1))/2
    i=torch.arange(n);B[i,i,:]=0;B[i,:,i]=0;B[:,i,i]=0
    for crossed in (False,True):
        g=f.Group(('slice211',),crossed,None,None,B,W@P)
        out={p:torch.zeros((n,)*len(p),dtype=torch.float64) for p in f.PARTS}
        f.group_dslices(g,out)
        exact=torch.stack([out[(4,)][0],out[(3,1)][0,1],out[(2,2)][0,1],out[(2,1,1)][0,1,2]])
        got=direct_samples(P,B,W[None,:3],crossed)[0]
        err=float((got-exact).abs().max());assert err<1e-12
        rows.append(dict(n=n,crossed=crossed,max_abs=err))
Path('outputs/research/k4s10/check_operator.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2))
print(json.dumps(rows))
