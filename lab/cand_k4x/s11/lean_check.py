"""Compare the fused one-direction birth with the verified CP operator."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json,itertools,signal
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from b_pair import BirthOperator
from lean_birth import project
signal.alarm(280)
rng=np.random.default_rng(37);n=16;r=8;rf=4
def a(shape):return f.array(rng.normal(size=shape).astype('d')*.1)
src=(a((n,n)),a((n,n)),a((n,r+2)),a((n,r+2)),a(n),a(n),a(n),a((n,2*rf)),a((n,rf)),a((n,rf)),a(n),a(n),r)
C=a((n,n));C=(C+C.T)/2;f.fill_diagonal(C,0)
p,w2,e1,e2,direction=[a(n) for _ in range(5)];dm=[(a(n),C)]
ctx=flops.BudgetContext(flop_budget=2**50,quiet=True)
with ctx:
    A,P,Z,L,s,e,ww,zf,r1,r2,c1,c2,r=src
    M=P*s+3*A*e+Z@L.T;X=3*A+zf[:,:rf]@r1.T;Y=A*ww+zf[:,rf:]@r2.T
    F=A*c1+P*c2;yy=Z[:,r+1,None]*f.ones((1,n))
    cols=[[],[],[]]
    for arr in ((X,P,Y),(M,P,P),(F,P,yy)):
        for perm in itertools.permutations(range(3)):
            for j in range(3):cols[j].append(arr[perm[j]]/6 if j==0 else arr[perm[j]])
    cp=tuple(f.concatenate(c,axis=1) for c in cols)
    op=BirthOperator(cp,dm,C,p,w2,e1,e2)
    u,b=project([src],dm,C,p,w2,e1,e2,direction)
    ref=op.left(u[:,None])[0];ref=3*(ref+ref.T)
    err=float(np.max(np.abs(np.asarray(b-ref))))
    rel=float(np.linalg.norm(np.asarray(b-ref))/np.linalg.norm(np.asarray(ref)))
result=dict(max_abs=err,relative=rel,n=n)
Path('outputs/research/k4s11/r3/operator-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));assert rel<1e-10
