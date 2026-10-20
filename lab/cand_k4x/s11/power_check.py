"""Validate the direct adjoint against dense fixed-state kernels."""
from direction_probe import *
from power_birth import PowerBirth
from lean_birth import project
import pickle
signal.alarm(280);rows=[]
for seed in (5,6,7):
 states=pickle.loads((OUT/f'states_s{seed}.pkl').read_bytes())['states']
 for li in (13,14):
  st=states[li];src=st['sources'][-1:];args=[st[k] for k in ('C','p','w2','e1','e2')]
  F=kernel(sum(source_tensor(x) for x in src),st['dm'],*args);flat=F.reshape(64,-1)
  def ff(a):return f.array(a) if hasattr(a,'shape') else a
  with flops.BudgetContext(flop_budget=2**60,quiet=True):
   op=PowerBirth([tuple(ff(x) for x in s) for s in src],[(ff(x),ff(m)) for x,m in st['dm']],*(ff(x) for x in args))
   u=normed(st['mean']);b=np.asarray(op.left(ff(u)));rr=np.asarray(op.right(ff(b)))
   uu,bb=op.rank1(ff(st['mean']),1);uu=np.asarray(uu)
   ul,bl=project([tuple(ff(x) for x in s) for s in src],[(ff(x),ff(m)) for x,m in st['dm']],*(ff(x) for x in args),ff(st['mean']))
  row=dict(seed=seed,layer=li,left_rel=float(np.sqrt(rel2(b,(u@flat).reshape(64,64)))),right_rel=float(np.sqrt(rel2(rr,flat@b.ravel()))),lean_rel=float(np.sqrt(rel2(np.asarray(bl),b))),power_cos2=float((uu@directions(F,st['mean'],0)['power1'])**2))
  rows.append(row)
assert max(r[k] for r in rows for k in ('left_rel','right_rel','lean_rel'))<1e-8,rows
(OUT/'power-check.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
