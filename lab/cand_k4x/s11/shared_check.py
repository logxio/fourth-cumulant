"""Shared-core structural equivalence and actual n512 basis diagnostics."""
from direction_probe import *
from shared_core import CoreBirth,merged_power
from power_birth import PowerBirth
import pickle
OUT=ROOT/'outputs/research/k4s11/r5';signal.alarm(280);start=time.monotonic();rng=np.random.default_rng(503)
def rel2(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
 den=float(np.sum(y*y));num=float(np.sum((x-y)**2))
 return num/den if den else (0.0 if num==0 else float('inf'))
n=32;q=11;q2=7;r=8;rf=4
Q=np.linalg.qr(rng.normal(size=(n,q)))[0];U=np.linalg.qr(rng.normal(size=(q,q2)))[0]
def src(rows,cols):
 def a(shape):return rng.normal(size=shape)*.03
 return (a((rows,cols)),a((rows,cols)),a((rows,r+2)),a((cols,r+2)),a(cols),a(cols),a(cols),a((rows,2*rf)),a((cols,rf)),a((cols,rf)),a(cols),a(cols),r)
def lift(s,basis):return tuple(basis@x if i in (0,1,2,7) else x for i,x in enumerate(s))
coords=[lift(src(q2,n),U) if j<3 else src(q,n) for j in range(6)]
old=[lift(x,Q) for x in coords];young=[src(n,n) for _ in range(8)]
G=sum(source_tensor(x) for x in coords);T=sum(source_tensor(x) for x in old+young)
C=rng.normal(size=(n,n))*.03;C=(C+C.T)/2;np.fill_diagonal(C,0)
p,w2,e1,e2,mean=[rng.normal(size=n)*.05 for _ in range(5)];dm=[(np.ones(n),C*.003)]
F=kernel(T,dm,C,p,w2,e1,e2);W=rng.normal(size=(n,n))/np.sqrt(n)
def ff(a):return f.array(a) if hasattr(a,'shape') else a
with flops.BudgetContext(flop_budget=2**60,quiet=True):
 co=CoreBirth(ff(Q),ff(G),*[ff(x) for x in (C,p,w2,e1,e2)])
 yo=PowerBirth([tuple(ff(x) for x in s) for s in young],[(ff(x),ff(m)) for x,m in dm],*[ff(x) for x in (C,p,w2,e1,e2)])
 eye=np.eye(n);got=np.stack([np.asarray(co.left(ff(u))+yo.left(ff(u))) for u in eye])
 u,b=merged_power(co,yo,ff(mean));u=np.asarray(u);b=np.asarray(b)
 rr=np.asarray(co.right(ff(b))+yo.right(ff(b)))
refs=first_read(F,W);reads=first_read(got,W)
check=dict(n=n,shared_rank=q,nested_rank=q2,old_sources=6,young_sources=8,birth_rel2=rel2(got,F),read_rel2=[rel2(x,y) for x,y in zip(reads,refs)],right_rel2=rel2(rr,F.reshape(n,-1)@b.ravel()),power_cos2=float((u@directions(F,mean,5)['power1'])**2),scope='synthetic all-old factors share Q, including thin Z/Zf; exact merged cubic core plus unmodified young sources')
assert check['birth_rel2']<1e-20 and max(check['read_rel2'])<1e-20 and check['right_rel2']<1e-20,check
st=pickle.loads((OUT/'state_n512_s5_l13.pkl').read_bytes());qc=st['Qc'].astype(float);left,sv,_=np.linalg.svd(qc,full_matrices=False)
rows=[]
for j,s in enumerate(st['sources']):
 A,P,Z,L,ss,ee,ww,zf,r1,r2,c1,c2,rank=s
 if j<st['meta']['old']:
  b=st['QU'] if j<st['meta']['tier2'] else st['Qc'];ia=j if j<st['meta']['tier2'] else j-st['meta']['tier2']
  fa=st['FA2'][ia] if j<st['meta']['tier2'] else st['FAo'][ia];fp=st['FP2'][ia] if j<st['meta']['tier2'] else st['FPo'][ia]
  rec=dict(A_rel2=rel2(b@fa,A),P_rel2=rel2(b@fp,P))
 else:rec={}
 residuals={}
 for rank in (16,32,64,128,352):
  qb=left[:,:rank];residuals[str(rank)]={name:rel2(qb@(qb.T@x),x) for name,x in [('A',A),('P',P),('Z',Z),('Zf',zf)]}
 rows.append(dict(source=j,tier='old' if j<6 else 'young',reconstruction=rec,projection_residuals=residuals))
result=dict(synthetic=check,actual_basis=dict(n=512,layer=13,rows=rows,singular_values=sv.tolist(),scope='actual B basis and legs; leg projection residuals are not K4/read errors'),wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(OUT/'shared-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(check));print('wall/rss',result['wall'],result['rss_bytes'])
