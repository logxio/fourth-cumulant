"""Flopscope bill for one closed trace read, with synthetic shape inputs.

Inputs are allocated before BudgetContext; measures this operator only.
The fixed dense source is consumed in chunks. This is a cost microbenchmark,
not a prediction or a full estimator bill.
"""
import json,time
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
n=1024;chunk=16
rng=np.random.default_rng(10)
P=f.array((rng.standard_normal((n,n))/n**.5).astype('f'))
B=f.array(rng.standard_normal((n,n,chunk)).astype('f'))
out4=f.zeros(n,dtype=f.float32);out22=f.zeros((n,n),dtype=f.float32)
t0=time.monotonic();ctx=flops.BudgetContext(flop_budget=2**41,quiet=True)
with ctx:
 H=P.T@P
 hd=f.einsum('ii->i',H)
 a=0.;b=0.
 for r0 in range(0,n,chunk):
  h=H[:,r0:r0+chunk]
  a=a+f.sum(f.sum(B*H[:,:,None],axis=(0,1))*hd[r0:r0+chunk])
  b=b+f.sum(B*h[:,None,:]*h[None,:,:])
 d4=(2/n)**2*(a+2*b);d22=(2/n)**2*a/3
 out4=out4+d4;out22=out22+d22
result=dict(n=n,units=ctx.flops_used/2**31,flops=ctx.flops_used,seconds=time.monotonic()-t0,
 scope='one source conditional trace and output fill; synthetic shape input, excludes source construction/storage and all other chain operations')
Path('outputs/research/k4s10/trace_meter.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
