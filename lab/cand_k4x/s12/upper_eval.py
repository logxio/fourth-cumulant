"""Full-source power-one age-one diagnostic, arbitrary suffix k; truth only scores output."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,time,resource,signal,argparse,hashlib,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'lab/cand_k4x/s11'),str(ROOT/'tools')]
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from whestbench import MLP,SetupContext
from mlpdata import load_truth,weights_from_seed,estimator_seed
def load_est(path):
 mod=types.ModuleType("frozen_b");exec(compile(Path(path).read_text(),str(path),"exec"),mod.__dict__);return mod.Estimator()
from power_birth import PowerBirth
from lean_pair import LeanPair
from lean_birth import recent_sources
from b_pair import diag,off
class FullPair(LeanPair):
 def __init__(self,k):super().__init__(100);self.k=k;self.birth_rows=[]
 def pre(self,v):
  orig=tuple(v.get(key) for key in ('g4row','wk4m','wk431','wk431r'))
  self.pending=[];self.birth=15-self.k<=v['li']<15
  if v['last']:return (orig[0]+self.final4,)+orig[1:] if self.final4 is not None else orig
  q4=q31=q22=None
  for born,a,b in self.current:
   da,db=diag(a),diag(b);q4=da*db;q31=off((da[:,None]*b+db[:,None]*a)/2);q22=off((da[:,None]*db[None,:]+db[:,None]*da[None,:])/6)
   self.pending.extend([(da,b),(db,a)])
  if self.birth:
   self.sources=[tuple(f.copy(x) if hasattr(x,'shape') else x for x in src) for src in recent_sources(v,100)]
   self.C=f.copy(v['C_off']);self.direction=f.copy(v['mu'])
   lam=v['lam_prev']*v['SSC_K']['lam_feed'][v['li']]
   self.dmats=[(f.ones(v['n'],dtype=f.float32),self.C*lam/3)]+[(x/6,m) for x,m in self.pending]
  if q4 is None:return orig
  a,b,c,d=orig;return a+q4,b+q22,c+q31.T,d+q31
 def post(self,v):
  self.pairs=[]
  if self.birth:
   start=time.monotonic();before=self.ctx.flops_used
   wa=v['W_all'];p=wa[:,4];w2=wa[:,6]
   e1=wa[:,5]-2*v['pk1v']*p;e2=wa[:,7]-2*v['pk1v']*w2
   op=PowerBirth(self.sources,self.dmats,self.C,p,w2,e1,e2)
   u,b=op.rank1(self.direction,1)
   # Residual of the normalized left singular direction; cheap local proxy, not dense SVD.
   self.pairs=[(v['li'],u,b)]
   self.birth_rows.append(dict(layer=v['li'],sources=len(self.sources),U=(self.ctx.flops_used-before)/2**31,wall=time.monotonic()-start))
   print(json.dumps(self.birth_rows[-1]),flush=True)
   self.sources=[];self.dmats=[];self.C=None;self.direction=None
  return v['g_prev']
def load(probe):
 source=(ROOT/'lab/cand_k4x/s11/base_b.py').read_text()
 edits=[('            # ---- WK slices ----','            _p.transport(locals())\n            # ---- WK slices ----'),('                if regen:\n                    # F68:','                D3, D21 = _p.k3(D3, D21)\n                if regen:\n                    # F68:'),('            WT = W_all.T','            g4row, wk4m, wk431, wk431r = _p.pre(dict(locals(), SSC_K=SSC_K))\n            WT = W_all.T'),('            # S21 = S_sep + Rres:','            _fd3, _fd21 = _p.feed(locals())\n            if _fd3 is not None:\n                S3c = S3c - _fd3\n                S21 = S21 - _fd21\n            # S21 = S_sep + Rres:'),('            K4_sigma = (fnp.sum(K4v)','            g_prev = _p.post(locals())\n            K4_sigma = (fnp.sum(K4v)')]
 for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
 mod=types.ModuleType('full_b');mod.__dict__['_p']=probe;exec(compile(source,'full_b','exec'),mod.__dict__);return mod.Estimator()
def main():
 if sys.platform=='linux':resource.setrlimit(resource.RLIMIT_AS,(int(10.5*1024**3),int(10.5*1024**3)))
 ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,default=0);ap.add_argument('k',type=int);ap.add_argument('index',type=int);ap.add_argument('out',type=Path);a=ap.parse_args();signal.alarm(900)
 truth=load_truth('mini');seed=truth['mlp_seed'][a.index];base=ROOT/'lab/cand_k4x/s11/base_b.py';budget=2**41
 from compressed_pair import CompressedPair
 probe=CompressedPair(a.k,a.rank) if a.rank else FullPair(a.k);est=load(probe) if a.k else load_est(str(base))
 start=time.monotonic();est.setup(SetupContext(width=1024,depth=16,flop_budget=budget,api_version='1',submission_dir=str(base.parent)));setup=time.monotonic()-start
 mlp=MLP(width=1024,depth=16,weights=[f.array(w,dtype=f.float32) for w in weights_from_seed(seed)],seed=estimator_seed(seed),name=str(truth['mlp_name'][a.index]))
 ctx=flops.BudgetContext(flop_budget=2**60,quiet=True);probe.ctx=ctx;start=time.monotonic()
 with ctx:pred=np.asarray(est.predict(mlp,budget)).astype(float)
 assert np.isfinite(pred).all();raw=float(np.mean((pred[-1]-truth['all_layer_means'][a.index,-1])**2));u=ctx.flops_used/2**31
 row=dict(k=a.k,rank=a.rank,index=a.index,raw=raw,U=u,adj=raw*max(.1,u/1024),setup=setup,wall=time.monotonic()-start,residual=ctx.residual_wall_time_s,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,births=probe.birth_rows,base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),scope='diagnostic upper arm; full B sources; rank1 one kernel power step; age1; full suffix K3 feedback; official prediction budget passed with enlarged meter limit')
 a.out.write_text(json.dumps(row,indent=2)+'\n');np.save(a.out.with_suffix('.npy'),pred);print(json.dumps(row),flush=True)
if __name__=='__main__':main()
