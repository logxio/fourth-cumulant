"""CPU Monte Carlo validation; fixed upstream tensors, current W only."""
import argparse,json,time
from pathlib import Path
import torch
from wick import expected,direct_samples

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--samples',type=int,default=32768);a=p.parse_args()
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(1010)
rows=[];start=time.monotonic()
for n in (16,32,64):
    B=torch.randn(n,n,n,dtype=torch.float64)/n
    B=(B+B.transpose(0,1))/2
    i=torch.arange(n);B[i,i,:]=0;B[i,:,i]=0;B[:,i,i]=0
    for kind in ('birth','old'):
        P=torch.diag(torch.rand(n,dtype=torch.float64)) if kind=='birth' else torch.randn(n,n,dtype=torch.float64)/n**.5
        for crossed in (False,True):
            d4,d22,ta,tb=expected(P,B,2/n,crossed)
            analytic=torch.tensor([d4,0,d22,0],dtype=torch.float64)
            sums=torch.zeros(4,dtype=torch.float64);sq=sums.clone()
            for k in range(0,a.samples,256):
                size=min(256,a.samples-k)
                W=torch.randn(size,3,n,dtype=torch.float64)*(2/n)**.5
                v=direct_samples(P,B,W,crossed)
                sums+=v.sum(0);sq+=(v*v).sum(0)
            mean=sums/a.samples
            se=((sq-sums*sums/a.samples)/(a.samples-1)/a.samples).sqrt()
            z=(mean-analytic)/se
            row=dict(n=n,kind=kind,crossed=crossed,samples=a.samples,analytic=analytic.tolist(),mean=mean.tolist(),se=se.tolist(),z=z.tolist(),a=float(ta),b=float(tb))
            assert float(z.abs().max())<5, row
            if kind=='birth': assert float(analytic.abs().max())<1e-14
            rows.append(row);print(json.dumps(row),flush=True)
result=dict(passed=True,seconds=time.monotonic()-start,max_abs_z=max(abs(z) for r in rows for z in r['z']),rows=rows)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
