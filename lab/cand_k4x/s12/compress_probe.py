import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,signal,resource,pickle,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'lab/cand_k4x/s11'))
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from power_birth import PowerBirth
from shared_core import CoreBirth
from compress import old_tucker,joint
from direct_compress import direct
from sketch_compress import sketch
P=ROOT/'outputs/research/k4s11/r5';OUT=ROOT/'outputs/research/k4s12'
def ff(x):return f.array(x) if hasattr(x,'shape') else x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['ref','old','joint','direct','sketch']);ap.add_argument('--rank',type=int,default=64);a=ap.parse_args();signal.alarm(280);start=time.monotonic()
 st=pickle.loads((P/'state_n512_s5_l13.pkl').read_bytes());n=512;old=st['meta']['old']
 src=[tuple(ff(x) for x in s) for s in st['sources']];args=[ff(st[k]) for k in ('C_off','p','w2','e1','e2')];dm=[(ff(x),ff(m)) for x,m in st['dm']]
 ctx=flops.BudgetContext(flop_budget=2**60,quiet=True);cost={}
 def step(name,fn):
  before=ctx.flops_used;t=time.monotonic();z=fn();cost[name]=dict(U1024=(ctx.flops_used-before)/2**31,flops=ctx.flops_used-before,wall=time.monotonic()-t);return z
 with ctx:
  if a.mode=='ref':op=PowerBirth(src,dm,*args);left=op.left
  else:
   Q=ff(np.load(P/'old_basis_n512_l13.npy'));G=ff(np.load(P/'old_core_n512_l13.npy'))
   if a.mode=='old':
    B,H=step('old_tucker',lambda:old_tucker(Q,G,a.rank));other=PowerBirth(src[old:],dm,*args)
   else:
    C,p,w2,e1,e2=args
    B,H=step('joint_space_and_core',lambda:sketch(src,a.rank,p,w2,e1,e2,C) if a.mode=='sketch' else direct(src,a.rank,p,w2,e1,e2,C) if a.mode=='direct' else joint(Q,G,src[old:],a.rank,p,w2,e1,e2,C));other=PowerBirth([],dm,*args)
   op=step('core_consumer_prepare',lambda:CoreBirth(B,H,*args))
   left=lambda u:op.left(u)+other.left(u)
   u=ff(st['mu']);u=u/f.sqrt(f.sum(u*u)+1e-30)
   b=step('left1',lambda:left(u));u=step('right',lambda:op.right(b)+other.right(b));u=u/f.sqrt(f.sum(u*u)+1e-30);b=step('left2',lambda:left(u))
   np.savez(OUT/f'{a.mode}_r{a.rank}_factors.npz',basis=np.asarray(B),core=np.asarray(H),direction=np.asarray(u),pair=np.asarray(b))
  rng=np.random.default_rng(514);norms=[];errors=[]
  refpath=OUT/'reference.npz';ref=np.load(refpath) if a.mode!='ref' else None
  projections=[]
  for z in range(8):
   u=ff(rng.choice([-1.,1.],n).astype('f'));got=np.asarray(left(u),dtype=float)
   if ref is None:projections.append(got.astype('f'))
   else:
    rr=ref['projections'][z];norms.append(float(np.sum(rr*rr)));errors.append(float(np.sum((got-rr)**2)))
  W=st['next_weight'].astype(float);vals=[[],[],[],[]];samples=[]
  for z in range(16):
   i,j,k=map(int,rng.choice(n,3,replace=False));samples.append([i,j,k])
   ii,ij,jj,jk=[np.asarray(left(ff((W[x]*W[y]).astype('f'))),dtype=float) for x,y in ((i,i),(i,j),(j,j),(j,k))]
   for ch,value in enumerate((W[i]@ii@W[i],(W[i]@ii@W[j]+W[i]@ij@W[i])/2,(W[j]@ii@W[j]+W[i]@jj@W[i])/6,(W[j]@ii@W[k]+W[i]@jk@W[i])/6)):vals[ch].append(float(value))
  if ref is None:np.savez(refpath,projections=np.stack(projections),values=np.array(vals),coordinates=samples)
  row=dict(mode=a.mode,rank=a.rank,n=n,wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cost=cost,coordinates=samples)
  if ref is not None:
   row.update(birth_rel2=sum(errors)/sum(norms),read_rel2=[float(np.sum((np.array(v)-rr)**2)/np.sum(rr**2)) for v,rr in zip(vals,ref['values'])])
  (OUT/f'{a.mode}_r{a.rank}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
