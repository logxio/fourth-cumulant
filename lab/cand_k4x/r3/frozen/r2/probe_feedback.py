"""Research-only K4 inputs to B; NumPy sidecar cost is separately itemized.
Truth is read only after prediction. No learned coefficients.
"""
import os,sys,json,time,types,resource,threading,argparse,math
from pathlib import Path
import numpy as np
import cross_math
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'tools'))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP,SetupContext
from mlpdata import load_truth,weights_from_seed,estimator_seed
from evalmlp import watchdog

def arr(x):return np.asarray(x,dtype=np.float32)
def off(x):x=x.copy();np.fill_diagonal(x,0);return x
def trace_inverse(A):
 n=len(A);G=6*A/(n+4);G.flat[::n+1]-=3*np.trace(A)/((n+4)*(n+2));return G

def radial(P,G):
 M=P@P.T;N=(P@G)@P.T;m=np.diag(M);h=np.diag(N)
 return m*h,(m[:,None]*N+h[:,None]*M)*.5,(m[:,None]*h[None,:]+h[:,None]*m[None,:]+4*M*N)/6

class Probe:
 def __init__(self,case):
  bits=case.split(':');self.kind=bits[0];self.age=int(bits[1]);self.route=bits[2].split('+')[0];self.rank=int(bits[3]) if len(bits)>3 else 4;self.sources=[];self.pending=None;self.keep=None;self.wa=None;self.cost=0.;self.norms=[];self.new=None;self.cross_new=None
 def pre(self,v):
  li,n=v['li'],v['n'];W=arr(v['W']);wa=arr(v['W_all']);self.wa=wa.copy();self.keep=None
  orig=tuple(v.get(k) for k in ('g4row','wk4m','wk431','wk431r'))
  if self.kind=='base':return orig
  q=np.zeros(n,np.float32);q31=np.zeros((n,n),np.float32);q22=q31.copy();keep=[q.copy(),q22.copy()];new=[]
  for src in self.sources:
   age=li-src['li'];limit=1 if self.kind=='mixshort' and src['kind']!='star' else self.age
   if age>limit:continue
   P=W if src['P'] is None else W@src['P'];self.cost+=float(src['P'] is not None)
   if src['kind']=='star':
    U=(P*src['p'])@src['C'].T;U2=U*U;H=P*src['h']
    a=4*np.sum(H*U2*U,axis=1)
    b=(U2*U*src['h'])@P.T+3*(H*U2)@U.T
    J=(H*U)@U2.T;c=2*(J+J.T);self.cost+=4
   elif src['kind']=='slice31':
    S=P*P;J=P@src['D'];a=4*np.sum(J*S*P,axis=1)
    b=(J@(S*P).T+3*P@(J*S).T).T
    Z=(P*J)@S.T;c=2*(Z+Z.T);self.cost+=4
   elif src['kind']=='pair':
    Q,e,d=src['Q'],src['e'],src['d'];S=P*P;SQ=S@Q;H=(SQ*e)@Q.T-S*d
    a=3*np.sum(H*S,axis=1);b=(3*P@(P*H).T).T
    c=(SQ*e)@SQ.T-3*(S*d)@S.T
    for k in range(len(e)):
     Z=(P*Q[:,k])@P.T;c+=2*e[k]*(Z*Z)
    self.cost+=2+len(e)+3*len(e)/n
   elif src['kind']=='cross':
    a,b,c=cross_math.contract(src,P,project=False);self.cost+=15+2*src['left'].shape[1]-3
   else:raise ValueError(src['kind'])
   ra,rb,rc=radial(P,src['G']);self.cost+=3
   a-=ra;b-=rb;c-=rc
   q+=a;q31+=b;q22+=c
   if age<limit and not v['last']:
    keep[0]+=a;keep[1]+=off(c);src['P']=arr(v['Phi'])[:,None]*P;new.append(src)
  if self.route=='allp' and li>0 and not v['last']:
   var=arr(v['var']);C=arr(v['C_off']);z=np.zeros_like(C);zv=np.zeros_like(q)
   templates=[(2*var*var,z,off((var[:,None]**2+var[None,:]**2)/3)),
              (zv,off(var[:,None]*C),z),(zv,z,off(2*C*C)),(zv,z,off(var[:,None]*var[None,:]))]
   def dot(x,y):return sum(k*np.sum(a.astype(float)*b) for k,a,b in zip((1,4,3),x,y))
   templates=[tuple(x/np.sqrt(dot(t,t)) for x in t) for t in templates]
   vec=(q,off(q31),off(q22));gram=np.array([[dot(a,b) for a in templates] for b in templates]);rhs=np.array([dot(a,vec) for a in templates]);beta=np.linalg.solve(gram,rhs)
   for coefficient,t in zip(beta,templates):q-=coefficient*t[0];q31-=coefficient*t[1];q22-=coefficient*t[2]
  self.sources=new;self.pending=(q,off(q31),off(q22));self.keep=keep
  self.norms.append(dict(layer=li,q4=float(np.linalg.norm(q)),q31=float(np.linalg.norm(q31)),q22=float(np.linalg.norm(q22))))
  self.new=None;self.cross_new=None
  if li>0 and not v['last'] and self.kind in ('star','combo','starcross','mix','mixshort'):
   C=arr(v['C_off']).copy();p=wa[:,4].copy();h=wa[:,8].copy();L=C*p[None,:];r=h*np.sum(L*L,axis=1);G=6*(r[:,None]*L+L.T*r[None,:])/(n+4)
   self.new=dict(kind='star',li=li,P=None,C=C,p=p,h=h,G=G)
  if li>0 and not v['last'] and self.kind in ('cross','starcross'):
   x=cross_math.birth(arr(v['C_off']),arr(v['D21']),wa[:,4],wa[:,6],rank=self.rank)
   self.cross_new={k:(a.astype(np.float32) if isinstance(a,np.ndarray) else a) for k,a in x.items()}
   self.cross_new.update(kind='cross',li=li,P=None);self.cost+=12
  if li==0 or self.route=='c':return orig
  a,b,c,d=orig
  if a is None:return orig
  a=fnp.array(arr(a)+q,dtype=fnp.float32)
  if self.route=='q4' or v['last']:return a,b,c,d
  return a,fnp.array(arr(b)+off(q22),dtype=fnp.float32),fnp.array(arr(c)+off(q31.T),dtype=fnp.float32),fnp.array(arr(d)+off(q31),dtype=fnp.float32)
 def cov(self,v):
  if self.route!='c' or self.pending is None:return v['C'],v['K2v']
  q,a,b=self.pending;wa=self.wa;p,h,w2=wa[:,4],wa[:,8],wa[:,6]
  d=(a*(h[:,None]*p[None,:])+a.T*(p[:,None]*h[None,:]))/6+b*(w2[:,None]*w2[None,:])/4
  # Wick columns (4,1)=12,(4,2)=13; exact first-order marginal variance term.
  dv=q*(wa[:,13]-2*arr(v['K1v'])*wa[:,12])/24
  np.fill_diagonal(d,dv)
  return fnp.array(arr(v['C'])+d,dtype=fnp.float32),fnp.array(arr(v['K2v'])+dv,dtype=fnp.float32)
 def post(self,v):
  li,n=v['li'],v['n'];g=v['g_prev']
  if self.kind=='base':return g
  if self.route in ('all','q4') and self.keep is not None:
   q,c=self.keep;p2=self.wa[:,4]**2;s=p2*p2*q
   if self.route=='all':s+=np.sum(p2[:,None]*p2[None,:]*c,axis=1)
   g=fnp.array(arr(g)-float(v['st']['cA'])*s-float(v['st']['cI'])*np.sum(s),dtype=fnp.float32)
  if self.new is not None:self.sources.append(self.new)
  if self.cross_new is not None:self.sources.append(self.cross_new)
  if not v['last'] and self.kind in ('slice31','pair','combo','mix','mixshort'):
   birth4=arr(v['K4v']).copy();birth22=arr(v['K22']).copy()
   if self.keep is not None and self.route in ('all','q4'):
    k4,k22=self.keep;p2=self.wa[:,4]**2;birth4-=p2*p2*k4
    if self.route=='all':birth22-=p2[:,None]*p2[None,:]*k22
   wa=self.wa;C=arr(v['C_off']);m=wa[:,0];var=wa[:,1]-m*m
   if self.kind in ('slice31','combo','mix','mixshort'):
    K31=np.zeros((n,n),np.float32);power=np.ones_like(C)
    for r,j in ((1,4),(2,6),(3,8)):
     power*=C;w3=(3*wa[:,1] if r==1 else 6*wa[:,0] if r==2 else wa[:,10])
     coef=w3-3*m*wa[:,j+1]+(3*m*m-3*var)*wa[:,j]
     K31+=(power/math.factorial(r))*(wa[:,j,None]*coef[None,:])
    if self.kind in ('combo','mix','mixshort'):K31-=self.wa[:,8,None]*(self.wa[:,4]**3)[None,:]*(C*C*C)
    D=off(K31);np.fill_diagonal(D,birth4/4);A=D+D.T;np.fill_diagonal(A,birth4)
    self.sources.append(dict(kind='slice31',li=li,P=None,D=D,G=trace_inverse(A)))
   if self.kind in ('pair','mix','mixshort'):
    rank=8 if self.kind=='pair' else self.rank
    R=off(birth22);e,Q=np.linalg.eigh(R);ix=np.argsort(abs(e))[-rank:];e=e[ix];Q=Q[:,ix];d=np.sum(Q*Q*e,axis=1)
    # Zero-diagonal rank-8 pair tensor; trace is diagonal row sums.
    tr=Q@(e*np.sum(Q,axis=0))-d;A=np.diag(tr);self.sources.append(dict(kind='pair',li=li,P=None,Q=Q,e=e,d=d,G=trace_inverse(A)))
    self.cost+=10 # Conservative dense symmetric eigensolver paper allowance, not billed.
  return g

def main():
 a=argparse.ArgumentParser();a.add_argument('case');a.add_argument('index',type=int);a.add_argument('out');a=a.parse_args();probe=Probe(a.case)
 threading.Thread(target=watchdog,args=(7.,),daemon=True).start()
 source=(Path(__file__).parents[1]/('estimator.py' if '+s22' in a.case else 'base_b.py')).read_text()
 source=source.replace('            WT = W_all.T','            g4row, wk4m, wk431, wk431r = _s9_pre(locals())\n            WT = W_all.T',1)
 source=source.replace('            C = flops.as_symmetric(C, symmetry=(0, 1))','            C, K2v = _s9_cov(locals())\n            C = flops.as_symmetric(C, symmetry=(0, 1))',1)
 source=source.replace('            K4_sigma = (fnp.sum(K4v)','            g_prev = _s9_post(locals())\n            K4_sigma = (fnp.sum(K4v)',1)
 mod=types.ModuleType('s9r2_b');mod.__dict__.update(_s9_pre=probe.pre,_s9_post=probe.post,_s9_cov=probe.cov);exec(compile(source,'s9r2_b','exec'),mod.__dict__)
 t=load_truth('mini');seed=t['mlp_seed'][a.index];budget=2**41;est=mod.Estimator();est.setup(SetupContext(width=1024,depth=16,flop_budget=budget,api_version='1',submission_dir=str(Path(__file__).parent)))
 mlp=MLP(width=1024,depth=16,weights=[fnp.array(w,dtype=fnp.float32) for w in weights_from_seed(seed)],seed=estimator_seed(seed),name=str(t['mlp_name'][a.index]));ctx=flops.BudgetContext(flop_budget=budget,quiet=True);ts=time.monotonic()
 with ctx:pred=np.asarray(est.predict(mlp,budget)).astype(float)
 assert pred.shape==(16,1024) and np.isfinite(pred).all()
 raw=float(np.mean((pred[-1]-t['all_layer_means'][a.index,-1])**2));r=dict(case=a.case,index=a.index,raw=raw,B_billed_U=ctx.flops_used/2**31,sidecar_matrix_U_upper=probe.cost,paper_U=ctx.flops_used/2**31+probe.cost+3,wall=time.monotonic()-ts,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,norms=probe.norms,note='Research hooks: raw and paper units only; not valid residual or full billing.')
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');np.save(out.with_suffix('.npy'),pred);print(json.dumps({k:v for k,v in r.items() if k!='norms'}),flush=True)
if __name__=='__main__':main()
