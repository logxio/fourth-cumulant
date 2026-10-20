"""Actual 1024-shape bill: two births, age-one reads and K3 feedback."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,resource,signal
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from power_birth import PowerBirth
from b_pair import diag,off,feedback_slices
p0=argparse.ArgumentParser();p0.add_argument('--sources',type=int,default=1);p0.add_argument('--steps',type=int,default=1);a=p0.parse_args()
signal.alarm(280);n=1024;r=8;rf=4;rng=np.random.default_rng(19)
def arr(shape):return f.array((rng.normal(size=shape)*.05).astype('f'))
sources=[(arr((n,n)),arr((n,n)),arr((n,r+2)),arr((n,r+2)),arr(n),arr(n),arr(n),arr((n,2*rf)),arr((n,rf)),arr((n,rf)),arr(n),arr(n),r) for _ in range(a.sources)]
C=arr((n,n));C=(C+C.T)*.5;f.fill_diagonal(C,0)
p,w2,e1,e2,direction=[arr(n) for _ in range(5)];W=arr((n,n));base4=arr(n);base31=arr((n,n));base22=arr((n,n))
ctx=flops.BudgetContext(flop_budget=2**50,quiet=True);cost={};start=time.monotonic()
def step(name,fn):
    before=ctx.flops_used;out=fn();cost[name]=ctx.flops_used-before;return out
def birth(dm):return PowerBirth(sources,dm,C,p,w2,e1,e2).rank1(direction,a.steps)
def read(u,b,last):
    if last:
        da=f.sum(W*W*u[None,:],axis=1);db=f.sum((W@b)*W,axis=1)
        return base4+da*db
    aa=(W*u[None,:])@W.T;bb=(W@b)@W.T;da,db=diag(aa),diag(bb)
    q4=base4+da*db
    q31=base31+off((da[:,None]*bb+db[:,None]*aa)/2)
    q22=base22+off((da[:,None]*db[None,:]+db[:,None]*da[None,:])/6)
    return [(da,bb),(db,aa)]
def feed(pending):
    fb=[];d3=f.zeros(n,dtype=f.float32);d21=f.zeros((n,n),dtype=f.float32)
    for x,m in pending:
        xx=w2*x/4;mm=p[:,None]*m*p[None,:];fb.append((xx,mm))
        aa,bb=feedback_slices(xx,mm);d3=d3-aa;d21=d21-bb
    return fb
def final_fb(fb):
    d3=f.zeros(n,dtype=f.float32)
    for x,m in fb:d3=d3+(W@x)*f.sum((W@m)*W,axis=1)
    return d3
with ctx:
    radial=step('radial_seed',lambda:[(f.ones(n,dtype=f.float32),C*.003)])
    u,b=step('birth13',lambda:birth(radial))
    pending=step('read14',lambda:read(u,b,False))
    fb=step('feed14',lambda:feed(pending))
    dm=step('seed14',lambda:radial+[(x/6,m) for x,m in pending])
    u,b=step('birth14',lambda:birth(dm))
    step('last_d4',lambda:read(u,b,True))
    step('last_k3',lambda:final_fb(fb))
result=dict(n=n,sources=a.sources,k=2,age=1,steps=a.steps,direction='one or two exact kernel power steps, initialized with B mean',
            flops=ctx.flops_used,delta_units=ctx.flops_used/2**31,phase_flops=cost,
            phase_units={k:v/2**31 for k,v in cost.items()},wall=time.monotonic()-start,
            rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,target_units=90,gate_pass=ctx.flops_used/2**31<=90,
            scope='actual synthetic 1024 shape; two births, full age1 read then d4-only last read, K3 feed and last K3 consumer; excludes unchanged B operations')
Path(f'outputs/research/k4s11/r4/meter_s{a.sources}_p{a.steps}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
