"""Independent dense/reference validation of the implicit non-tree operator."""
import json,signal
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from b_pair import BirthOperator,feedback_slices
from pair_read import torch,f
from k4n3_young_direct_chain import Recorder,tree_tensor,NS

signal.alarm(280)
torch.set_num_threads(2)
rng=np.random.default_rng(981);n=8;rank=3
cp=[rng.normal(size=(n,rank))*.1 for _ in range(3)]
# Symmetric K3 represented by the six CP permutations.
import itertools
cp=[np.concatenate([cp[p[a]]/6 if a==0 else cp[p[a]] for p in itertools.permutations(range(3))],axis=1) for a in range(3)]
T=np.einsum('ir,jr,kr->ijk',*cp)
C=rng.normal(size=(n,n))*.02;C=C@C.T+np.eye(n)
co=C.copy();np.fill_diagonal(co,0)
mu=rng.normal(size=n)*.2
v=rng.normal(size=n)*.01;m=rng.normal(size=(n,n))*.01;m=(m+m.T)/2
D=v[:,None,None]*m[None]
K=f.coerce_input({1:torch.tensor(mu),2:torch.tensor(C),3:torch.tensor(T)},k_max=4,kind=f.Kind.SIMPLE)
f.SPEC=f.parse_spec('slice4+slice31,!slice22+slice211')
ds=f.DSTensor(d=4,n=n,slices={(4,):torch.zeros(n,dtype=torch.float64),(3,1):torch.zeros(n,n,dtype=torch.float64),(2,2):torch.zeros(n,n,dtype=torch.float64),(2,1,1):torch.tensor(D)},autozero=True,dtype=torch.float64)
K[4]=f.SK4.from_dstensor(ds)
# Use the actual dslice reconstructed by SK4, including its conventions.
Dactual=K[4].to_dstensor().get_slice((2,1,1)).numpy() if hasattr(K[4],'to_dstensor') else None
rec=Recorder(light=False);rec.install();rec.start_layer();saved=rec.saved['multiply_wicks'];lookup={}
def capture(kp,k,p,wick_lookup):
    for aa,bb in ((1,1),(2,1),(1,2),(2,2)):lookup[aa,bb]=wick_lookup(aa,bb).numpy().copy()
    return saved(kp,k,p,wick_lookup)
rec.saved['multiply_wicks']=capture
f.nonlin(K,nonlin_wick_coef=f.WICK_COEF_D['relu'])
target=(rec.current.total-tree_tensor(rec.current.trees,n,torch.device('cpu'),torch.float64)).numpy()
p=lookup[1,1];w2=lookup[2,1];e1=lookup[1,2]-2*rec.p1.numpy()*p;e2=lookup[2,2]-2*rec.p1.numpy()*w2
rec.uninstall()
ctx=flops.BudgetContext(flop_budget=2**50,quiet=True)
with ctx:
    op=BirthOperator(tuple(fnp.array(x) for x in cp),[(fnp.array(v),fnp.array(m))],fnp.array(co),fnp.array(p),fnp.array(w2),fnp.array(e1),fnp.array(e2))
    dense=np.asarray(op.left(fnp.eye(n)))
    om=rng.normal(size=(5,n,n));u=rng.normal(size=(n,5))
    right=np.asarray(op.right(fnp.array(om)));left=np.asarray(op.left(fnp.array(u)))
    d3,d21=feedback_slices(fnp.array(v),fnp.array(m))
    ts=(v[:,None,None]*m[None]+v[None,:,None]*m[:,None,:]+v[None,None,:]*m[:,:,None])/3
    ix=np.arange(n);ref21=ts[ix,ix,:].copy();np.fill_diagonal(ref21,0)
results=dict(operator_right_max_abs=float(np.max(abs(right-np.einsum('ijk,sjk->is',dense,om)))),
    operator_left_max_abs=float(np.max(abs(left-np.einsum('is,ijk->sjk',u,dense)))),
    reference_birth_max_abs=float(np.max(abs(dense-target))),reference_birth_rel=float(np.linalg.norm(dense-target)/np.linalg.norm(target)),
    feedback_d3_max_abs=float(np.max(abs(np.asarray(d3)-ts[ix,ix,ix]))),feedback_d21_max_abs=float(np.max(abs(np.asarray(d21)-ref21))))
print(json.dumps(results,indent=2))
assert results['operator_right_max_abs']<1e-9 and results['operator_left_max_abs']<1e-9
assert results['reference_birth_rel']<1e-8
