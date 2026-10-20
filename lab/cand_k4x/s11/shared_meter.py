"""1024-shape bills: free old cubic core plus actual young-source schedule.
Core assembly and old thin residuals are explicitly excluded (optimistic).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time,signal,resource,gc
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from power_birth import PowerBirth
from shared_core import CoreBirth
P=Path('outputs/research/k4s11/r5');signal.alarm(280);start=time.monotonic();n=1024;r=352;rng=np.random.default_rng(515)
def arr(shape):return f.array((rng.normal(size=shape)*.03).astype('f'))
C=arr((n,n));C=(C+C.T)/2;f.fill_diagonal(C,0)
p,w2,e1,e2,u0=[arr(n) for _ in range(5)];Q=arr((n,r));G=np.zeros((r,r,r),dtype='f')
for _ in range(4):
 v=rng.normal(size=r).astype('f')*.03;G+=v[:,None,None]*v[None,:,None]*v[None,None,:]
G=f.array(G);ctx=flops.BudgetContext(flop_budget=2**60,quiet=True);phases={}
def step(name,fn):
 before=ctx.flops_used;out=fn();phases[name]=(ctx.flops_used-before)/2**31;return out
with ctx:
 op=step('prepare_repeated_support',lambda:CoreBirth(Q,G,C,p,w2,e1,e2))
 u=u0/f.sqrt(f.sum(u0*u0)+1e-30)
 b=step('left_initial',lambda:op.left(u));u=step('right_power',lambda:op.right(b));u=u/f.sqrt(f.sum(u*u)+1e-30)
 b=step('left_final',lambda:op.left(u))
old=dict(rank=r,units=ctx.flops_used/2**31,phase_units=phases,wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 scope='optimistic free symmetric old K3 core; basis rank352; includes repeated-support preparation and 1 kernel power step; excludes core construction, thin residuals and shared radial/triangle terms')
(P/'old-core-meter.json').write_text(json.dumps(old,indent=2)+'\n');print(json.dumps(old),flush=True)
del op,Q,G,b;gc.collect()
cap=json.loads((P/'capture.json').read_text());rows=[]
for layer in range(4,15):
 info=next(x for x in cap['rows'] if x['layer']==layer);caps=[2*x for x in info['young_caps']]
 sources=[]
 for m in caps:
  sources.append((arr((n,m)),arr((n,m)),arr((n,10)),arr((m,10)),arr(m),arr(m),arr(m),arr((n,8)),arr((m,4)),arr((m,4)),arr(m),arr(m),8))
 ctx=flops.BudgetContext(flop_budget=2**60,quiet=True);t=time.monotonic()
 with ctx:
  yo=PowerBirth(sources,[(f.ones(n,dtype=f.float32),C*.003)],C,p,w2,e1,e2)
  u,b=yo.rank1(u0,1)
 row=dict(layer=layer,old_sources=info['old'],young_sources=info['young'],young_columns=caps,young_birth_units=ctx.flops_used/2**31,
          merged_birth_optimistic_units=ctx.flops_used/2**31+(old['units'] if info['old'] else 0),wall=time.monotonic()-t)
 rows.append(row);print(json.dumps(row),flush=True)
 (P/'birth-meter.json').write_text(json.dumps(dict(n=n,rows=rows,old=old,scope='actual n1024 synthetic shapes; young columns scaled 2x observed n512 caps; B n1024 caps are not observed here; old-core assembly and thin residuals charged zero',wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
 del yo,sources,b;gc.collect()
