"""Cloud meter for the literal B-triples -> existing CoreSource adapter.

Not an estimator: meters one newborn's same-side read. Reconstruction of B's
legs, repeated-index corrections, crossed slices and feedback are excluded.
"""
import sys

# The dense suffix and the historical B-triples adapter are different
# executable representations. Keep the old measurement reproducible; route
# the frozen suffix trace to its own explicitly modelled accounting entry.
if '--trace' in sys.argv:
    import argparse
    import hashlib
    import json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).with_name('r4')))
    from trace_cost import cost_trace
    parser = argparse.ArgumentParser(description='Frozen dense-chain trace cost model, not a whole-chain BudgetContext bill')
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--accuracy-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    accuracy = json.loads(args.accuracy_result.read_text())
    assert accuracy['config_sha256'] == config_hash, 'accuracy/config mismatch'
    result = cost_trace(json.loads(args.trace.read_text()))
    result.update(config_sha256=config_hash, config=json.loads(args.config.read_text()),
                  accuracy_result=str(args.accuracy_result), accuracy_raw=accuracy['final_mse'],
                  trace_sha256=hashlib.sha256(args.trace.read_bytes()).hexdigest(),
                  historical_63b_row='different B-triples representation; not additive to this dense suffix')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','unknown_ops')}, indent=2))
    raise SystemExit(0)

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np
from scipy.special import ndtr
import flopscope as flops
import flopscope.numpy as fnp
import reference_port as port
ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('AW_DATA',str(ROOT/'data'))
sys.path.insert(0,str(ROOT/'tools'))
from mlpdata import load_truth,weights_from_seed

p=argparse.ArgumentParser()
p.add_argument('--start',type=int,default=6)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
ss=np.load(f'outputs/research/k4s5/seed_states/B_mini0_pre{a.start}.npz')
A,P,Z,L=(ss[k] for k in ('A_st','P_st','Z_st','L_st'))
w2,s,e,c1,c2=(ss[k] for k in ('w2b_list','s_list','e_list','c1_list','c2_list'))
n=P.shape[1];rfb=int(ss['rfb']);rres=int(ss['rres'])
sigma=np.sqrt(np.diag(ss['C']))
alpha=ss['mu']/sigma
w1=ndtr(alpha).astype('f')
wick2=(np.exp(-alpha**2/2)/(np.sqrt(2*np.pi)*sigma)).astype('f')
xs,ys,zs=[],[],[]
for k in range(A.shape[0]):
    M=P[k]*s[k]+3*A[k]*e[k]+Z[k]@L[k].T
    X,Y=3*A[k],A[k]*w2[k]
    if rfb:
        F=ss['Zf_st'][k]
        X=X+F[:,:rfb]@ss['R1T_st'][k].T
        Y=Y+F[:,rfb:]@ss['R2T_st'][k].T
    triples=((X,P[k],Y),(M,P[k],P[k]),
             (A[k]*c1[k]+P[k]*c2[k],P[k],
              np.broadcast_to(Z[k,:,rres+1,None],P[k].shape)))
    for x,y,z in triples:
        for first,second,third in ((x,y,z),(y,x,z),(z,x,y)):
            xs.append(first/3);ys.append(w1[:,None]*second);zs.append(w1[:,None]*third)
x=np.concatenate(xs,axis=1);y=np.concatenate(ys,axis=1);z=np.concatenate(zs,axis=1)
del xs,ys,zs
c=ss['C'].copy();np.fill_diagonal(c,0)
u,v=w1[:,None]*c,np.eye(n,dtype='f')
core=12*wick2[:,None]*x
cols=x.shape[1]
del x
source=port.CoreSource(*(fnp.array(t) for t in (u,v,y,z,core)),tag='three2')
del u,v,y,z,core
weights=weights_from_seed(load_truth('mini')['mlp_seed'][0])
perm=np.argsort(ss['inv_perm'])
q=fnp.array(weights[a.start+1].T[:,perm].copy())
transport=flops.BudgetContext(flop_budget=2**41,quiet=True)
t0=time.monotonic()
with transport:
    source=source.transported(q)
transport_seconds=time.monotonic()-t0
ctx=flops.BudgetContext(flop_budget=2**41,quiet=True)
calls=[]
original_mm=port.mm
def metered_mm(a,b):
    before=ctx.flops_used
    result=original_mm(a,b)
    calls.append(dict(left=list(a.shape),right=list(b.shape),units=(ctx.flops_used-before)/2**31))
    return result
port.mm=metered_mm
t0=time.monotonic()
with ctx:
    values=source.same_slices()
elapsed=time.monotonic()-t0
result=dict(start=a.start,columns=cols,units=ctx.flops_used/2**31,
            transport_units=transport.flops_used/2**31,transport_seconds=transport_seconds,
            matmuls=calls,
            seconds=elapsed,maxrss_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            finite=all(bool(np.isfinite(np.asarray(v)).all()) for v in values[:3]),
            meaning='one uncompressed three2 newborn same-side read in literal existing CoreSource; not full K4 estimator',
            excluded=['B leg construction','repeated-index corrections','crossed slices','feedback'])
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
