"""n1024 synthetic shape, explicit source widths from captured B schedule.
Bill basis construction, all source projection/core building and birth consumption.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['OMP_NUM_THREADS']='2'
import sys,json,time,signal,resource,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'lab/cand_k4x/s11'))
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from compress import source_core,joint
from direct_compress import direct
from sketch_compress import sketch
from power_birth import PowerBirth
from shared_core import CoreBirth
P=ROOT/'outputs/research/k4s12'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sketch',action='store_true');ap.add_argument('--direct',action='store_true');ap.add_argument('--rank',type=int,default=32);ap.add_argument('--layer',type=int,default=14);a=ap.parse_args();signal.alarm(280);t=time.monotonic();n=1024;r=352;rng=np.random.default_rng(88)
 def arr(shape):return f.array((rng.normal(size=shape)*.01).astype('f'))
 info=next(x for x in json.loads((ROOT/'outputs/research/k4s11/r5/capture.json').read_text())['rows'] if x['layer']==a.layer)
 def src(m):return (arr((n,m)),arr((n,m)),arr((n,10)),arr((m,10)),arr(m),arr(m),arr(m),arr((n,8)),arr((m,4)),arr((m,4)),arr(m),arr(m),8)
 # Old sources retain n atoms. Young source retained widths use observed n512 caps doubled.
 old=[src(n) for _ in range(info['old'])];young=[src(2*x) for x in info['young_caps']]
 Q=arr((n,r));C=arr((n,n));C=(C+C.T)/2;f.fill_diagonal(C,0);p,w2,e1,e2,direction=[arr(n) for _ in range(5)]
 ctx=flops.BudgetContext(flop_budget=2**60,quiet=True);phases={}
 def step(name,fn):
  before=ctx.flops_used;tt=time.monotonic();z=fn();phases[name]=dict(U=(ctx.flops_used-before)/2**31,wall=time.monotonic()-tt);return z
 with ctx:
  # Q supplied by B; QR is needed to reproduce the orthonormal old-core coordinates.
  if not (a.direct or a.sketch):Q=step('old_basis_qr',lambda:f.linalg.qr(Q,mode='reduced')[0])
  def assemble():
   G=f.zeros((r,r,r),dtype=f.float32)
   for s in old:G=G+source_core(tuple(Q.T@x if i in (0,1,2,7) else x for i,x in enumerate(s)))
   return G
  if a.sketch:B,H=step('sketch_basis_projection_core',lambda:sketch(old+young,a.rank,p,w2,e1,e2,C))
  elif a.direct:B,H=step('direct_basis_projection_core',lambda:direct(old+young,a.rank,p,w2,e1,e2,C))
  else:
   G=step('old_projection_and_core',assemble)
   B,H=step('joint_basis_young_projection_all_core',lambda:joint(Q,G,young,a.rank,p,w2,e1,e2,C))
  radial=step('radial',lambda:PowerBirth([],[(f.ones(n,dtype=f.float32),C*.003)],C,p,w2,e1,e2))
  op=step('prepare',lambda:CoreBirth(B,H,C,p,w2,e1,e2))
  u=direction/f.sqrt(f.sum(direction*direction)+1e-30)
  b=step('left1',lambda:op.left(u)+radial.left(u));u=step('right',lambda:op.right(b)+radial.right(b));u=u/f.sqrt(f.sum(u*u)+1e-30);b=step('left2',lambda:op.left(u)+radial.left(u))
 row=dict(rank=a.rank,layer=a.layer,n=n,phases=phases,U=ctx.flops_used/2**31,wall=time.monotonic()-t,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,old_sources=len(old),young_sources=len(young),young_columns=[2*x for x in info['young_caps']],scope='synthetic n1024; old six n-column sources; young doubled n512 observed caps; actual full algorithm flops; B basis supplied; all new orthogonalization, old core assembly, joint basis, young projection, core, one power-step consumers charged')
 (P/f'meter_{"sketch_" if a.sketch else "direct_" if a.direct else ""}r{a.rank}_l{a.layer}.json').write_text(json.dumps(row,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
