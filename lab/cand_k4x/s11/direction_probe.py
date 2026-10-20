"""n64 diagnostics only: fixed B states, dense birth and first-read references."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
import json,time,signal,resource,types,argparse
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from whestbench import MLP,SetupContext
from lean_pair import LeanPair
from lean_birth import recent_sources
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/research/k4s11/r4'

def load_probe(probe):
    source=(Path(__file__).with_name('base_b.py')).read_text().replace('riders = (n == 1024 and L == len(CORR_BETA))','riders = (L == len(CORR_BETA))')
    edits=[('            # ---- WK slices ----','            _p.transport(locals())\n            # ---- WK slices ----'),
    ('                if regen:\n                    # F68:','                D3, D21 = _p.k3(D3, D21)\n                if regen:\n                    # F68:'),
    ('            WT = W_all.T','            g4row, wk4m, wk431, wk431r = _p.pre(dict(locals(), SSC_K=SSC_K))\n            WT = W_all.T'),
    ('            # S21 = S_sep + Rres:','            _fd3, _fd21 = _p.feed(locals())\n            if _fd3 is not None:\n                S3c = S3c - _fd3\n                S21 = S21 - _fd21\n            # S21 = S_sep + Rres:'),
    ('            K4_sigma = (fnp.sum(K4v)','            g_prev = _p.post(locals())\n            K4_sigma = (fnp.sum(K4v)')]
    for old,new in edits:assert source.count(old)==1;source=source.replace(old,new)
    mod=types.ModuleType('probe_b');mod.__dict__['_p']=probe
    exec(compile(source,'probe_b','exec'),mod.__dict__);return mod.Estimator()

def nsources(v,count):
    return [tuple(np.array(x,copy=True).astype(float) if hasattr(x,'shape') else x for x in src) for src in recent_sources(v,count)]

def sym3(x,y,z):
    t=np.einsum('ir,jr,kr->ijk',x,y,z,optimize=True)
    return (t+t.transpose(0,2,1)+t.transpose(1,0,2)+t.transpose(1,2,0)+t.transpose(2,0,1)+t.transpose(2,1,0))/6

def source_tensor(src):
    A,P,Z,L,s,e,ww,zf,r1,r2,c1,c2,r=src
    M=P*s+3*A*e+Z@L.T
    X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T
    F=A*c1+P*c2;y=Z[:,r+1]
    ss=F@P.T;ss=ss+ss.T
    return sym3(X,P,Y)+sym3(M,P,P)+(ss[:,:,None]*y[None,None,:]+ss[:,None,:]*y[None,:,None]+ss[None,:,:]*y[:,None,None])/6

def kernel(T,dm,C,p,w2,e1,e2):
    v=T*(e1[:,None,None]*(p[None,:,None]*p[None,None,:]+w2[None,:,None]*w2[None,None,:]*C[None])+e2[:,None,None]*(p[None,:,None]*w2[None,None,:]*C[:,None,:]+w2[None,:,None]*p[None,None,:]*C[:,:,None]))
    v+=e2[:,None,None]*w2[None,:,None]*w2[None,None,:]*C[:,:,None]*C[:,None,:]*C[None]
    for x,m in dm:
        v+=(e2*.5-p*p)[:,None,None]*x[:,None,None]*m[None]*p[None,:,None]*p[None,None,:]
        v+=e1[:,None,None]*.5*(x[None,:,None]*m[:,None,:]*w2[None,:,None]*p[None,None,:]+x[None,None,:]*m[:,:,None]*p[None,:,None]*w2[None,None,:])
    ix=np.arange(len(p));v[ix,ix,:]=0;v[ix,:,ix]=0;v[:,ix,ix]=0
    return 3*(v+v.transpose(0,2,1))

def first_read(F,W):
    n=len(W);aa=np.einsum('ir,jr->rij',W,W);bb=(W[None]@F)@W.T
    da=np.diagonal(aa,axis1=1,axis2=2);db=np.diagonal(bb,axis1=1,axis2=2)
    d4=np.sum(da*db,axis=0)
    d31=np.sum(da[:,:,None]*bb+db[:,:,None]*aa,axis=0)/2
    d22=(da.T@db+db.T@da)/6
    d211=(np.einsum('ri,rjk->ijk',da,bb)+np.einsum('ri,rjk->ijk',db,aa))/6
    ix=np.arange(n);d31[ix,ix]=0;d22[ix,ix]=0
    d211[ix,ix,:]=0;d211[ix,:,ix]=0;d211[:,ix,ix]=0
    return [d4,d31,d22,d211]

def normed(u):return u/max(np.linalg.norm(u),1e-300)
def directions(F,mean,seed):
    n=len(F);flat=F.reshape(n,-1)
    rng=np.random.default_rng(seed);q=np.linalg.qr(flat@rng.standard_normal((n*n,5)))[0]
    small=np.linalg.svd(q.T@flat,full_matrices=False)[0][:,0]
    # Exact SVD diagnosis specifies the maximum rank-one energy, separate from rsvd.
    ev,vec=np.linalg.eigh(flat@flat.T)
    out={'rsvd':q@small,'svd':vec[:,-1],'mean':normed(mean)}
    u=out['mean']
    for step in (1,2):u=normed(flat@(u@flat));out[f'power{step}']=u
    return out

def rel2(x,y):return float(np.sum((x-y)**2)/max(np.sum(y*y),1e-300))
class Capture:
    def __init__(self):self.states={}
    def transport(self,v):pass
    def k3(self,a,b):return a,b
    def feed(self,v):return None,None
    def pre(self,v):
        if v['li'] in (11,12,13,14):
            C=np.array(v['C_off'],dtype=float)
            self.current=dict(sources=nsources(v,100),C=C,mean=np.array(v['mu'],dtype=float),dm=[(np.ones(v['n']),C*float(v['lam_prev'])*v['SSC_K']['lam_feed'][v['li']]/3)])
        return tuple(v.get(k) for k in ('g4row','wk4m','wk431','wk431r'))
    def post(self,v):
        if v['li'] in (11,12,13,14):
            wa=np.array(v['W_all'],dtype=float);p,w2=wa[:,4],wa[:,6];pk=np.array(v['pk1v'])
            self.current.update(p=p,w2=w2,e1=wa[:,5]-2*pk*p,e2=wa[:,7]-2*pk*w2)
            self.states[v['li']]=self.current
        return v['g_prev']

def run(probe,seed):
    n=64;budget=2**55;rng=np.random.default_rng(seed)
    weights=[(rng.standard_normal((n,n))*(2/n)**.5).astype('f') for _ in range(16)]
    est=load_probe(probe);est.setup(SetupContext(width=n,depth=16,flop_budget=budget,api_version='1',submission_dir=str(Path(__file__).parent)))
    mlp=MLP(width=n,depth=16,weights=[f.array(w) for w in weights],seed=seed,name=f'n64-seed{seed}')
    with flops.BudgetContext(flop_budget=budget,quiet=True):pred=np.asarray(est.predict(mlp,budget)).astype(float)
    return pred,weights

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=5);a=ap.parse_args();signal.alarm(280);start=time.monotonic()
    cap=Capture();pred,weights=run(cap,a.seed)
    old=np.load(ROOT/f'outputs/research/k4s11/r2/plain_b_s{a.seed}.npy');assert np.array_equal(old,pred)
    # Keep fixed states as the reusable diagnostic input, outside the billed candidate.
    import pickle
    (OUT/f'states_s{a.seed}.pkl').write_bytes(pickle.dumps(dict(states=cap.states,weights=weights,pred=pred)))
    rows=[]
    for li,st in cap.states.items():
        tensors=[source_tensor(x) for x in st['sources']];rest={key:st[key] for key in ('dm','C','p','w2','e1','e2')}
        full=kernel(sum(tensors),**rest);fullread=first_read(full,weights[li+1])
        for count in (0,1,2):
            F=full if count==0 else kernel(sum(tensors[-count:]),**rest)
            rr=first_read(F,weights[li+1]);flat=F.reshape(64,-1)
            dirs=directions(F,st['mean'],111+li*1009)
            for name,u in dict(exact=None,**dirs).items():
                cand=F if u is None else np.einsum('i,jk->ijk',u,(u@flat).reshape(64,64))
                cr=rr if u is None else first_read(cand,weights[li+1])
                rows.append(dict(seed=a.seed,layer=li,sources='all' if not count else count,direction=name,
                  birth_rel2_full=rel2(cand,full),birth_rel2_selected=rel2(cand,F),
                  read_rel2_full=[rel2(x,y) for x,y in zip(cr,fullread)],read_rel2_selected=[rel2(x,y) for x,y in zip(cr,rr)],
                  birth_energy=float(np.sum(full*full)),selected_energy=float(np.sum(F*F)),read_energy=[float(np.sum(x*x)) for x in fullread],
                  direction_cos2_svd=None if u is None else float((u@dirs['svd'])**2)))
    result=dict(seed=a.seed,rows=rows,base_exact=True,wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (OUT/f'fixed_s{a.seed}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
if __name__=='__main__':main()
