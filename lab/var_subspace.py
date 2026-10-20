"""Subspace oracle: exact directional cumulants, without materializing k^4.
Cloud only. MC halves are independent; all explained powers and final errors use cross-half products.
"""
import argparse, os, sys, time, types, json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('AW_DATA',str(ROOT/'data'))
for p in ('tools','refs/504aldo/lean','refs/ref_0709/src'): sys.path.insert(0,str(ROOT/p))
from mlpdata import weights_from_seed,load_truth
KS=(8,16,32,64,128)

def lean_module():
    path=ROOT/'refs/504aldo/lean/lean_k3_aug.py'
    src=path.read_text()
    edits=[('Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(dtype) for w in weights]',
            'Ws = [torch.from_numpy(w.astype(np.float64).T).contiguous().to(device=torch.get_default_device(), dtype=dtype) for w in weights]'),
           ('torch.set_num_threads(22)','pass'),
           ('            D3, D21 = K3.dslices()\n','            D3, D21 = K3.dslices()\n            D3c = D3\n            if HOOK is not None: D3 = HOOK(l, "k3", D3)\n'),
           ('            D3_w = D3 * w1 ** 3\n','            D3_w = D3c * w1 ** 3\n'),
           ('                wk4_row = metric_c * dG\n','                wk4_row = metric_c * dG\n                if HOOK is not None: wk4_row = HOOK(l, "k4", wk4_row)\n'),
           ('        C = 0.5 * (C + C.T)\n        if K3 is not None:', '        C = 0.5 * (C + C.T)\n        if HOOK is not None:\n            vnew = HOOK(l, "var", torch.diagonal(C))\n            C = C - torch.diag(torch.diagonal(C)) + torch.diag(vnew)\n        if K3 is not None:')]
    for old,new in edits:
        assert src.count(old)==1,old
        src=src.replace(old,new)
    src=src.replace('dict(C=C, mu=mu, var=torch.diagonal(C), K21=K[(2, 1)],', 'dict(carrier=K3, C=C, mu=mu, var=torch.diagonal(C), K21=K[(2, 1)],')
    m=types.ModuleType('sublean');m.__file__=str(path);exec(compile(src,str(path),'exec'),m.__dict__);m.HOOK=None
    return m

def init(a):
    m=lean_module();torch.set_default_device('cpu');torch.set_default_dtype(torch.float64)
    statics=m.build_static_coefs(1024);torch.set_default_dtype(torch.float32);torch.set_default_device('cuda')
    tr=load_truth('mini')
    for idx in a.idx:
        out=a.out/f'chain{idx}.npz'
        if out.exists(): continue
        rec={}; us=[];post4=[];post3=[]; feats=[]; pre4=[]
        def hook(l,name,x):
            if name=='k4': pre4.append((l,x.detach().cpu().numpy()))
            return x
        m.HOOK=hook
        def gh(l,G,state):
            C=state['C'];ev,U=torch.linalg.eigh(C);U=U[:,-128:].flip(1).contiguous()
            us.append(U.cpu().numpy());rec[f'C{l}']=C.cpu().numpy()
            post4.append(state['K4v'].cpu().numpy());post3.append(state['K3v'].cpu().numpy())
            off=C-torch.diag(torch.diagonal(C));D=state['K21']
            xs=[off*off,(D+D.T)*.5,G,off]
            feats.append(torch.stack([U.T@x@U for x in xs]).cpu().numpy())
            return G
        W=np.stack(weights_from_seed(tr['mlp_seed'][idx]));t0=time.time()
        p,mu,var,d3,*_=m.lean_k3_predict(W,dtype=torch.float32,statics=statics,k4_aug=True,return_diag=True,g_hook=gh)
        k4=np.zeros((16,1024));
        for l,x in pre4:k4[l]=x
        np.savez(out,P=p.cpu().numpy(),mu=mu.cpu().numpy(),var=var.cpu().numpy(),d3=d3.cpu().numpy(),k4=k4,U=np.stack(us),post4=np.stack(post4),post3=np.stack(post3),features=np.stack(feats),**rec)
        print('INIT',idx,round(time.time()-t0,2),flush=True)

def cumulants(M):
    a,b,c,d=np.moveaxis(M,-2,0)
    return c-3*a*b+2*a**3,d-4*a*c-3*b*b+12*b*a*a-6*a**4

def sums(x):
    x2=x*x
    return torch.stack([x.sum(0),x2.sum(0),(x2*x).sum(0),(x2*x2).sum(0)])

def mc(a):
    tr=load_truth('mini');chunk=a.chunk;nch=int(np.ceil(a.n/chunk/2))*2
    for idx in a.idx:
        out=a.out/f'mc{idx}.npz'
        if out.exists():continue
        ch=np.load(a.out/f'chain{idx}.npz');W=torch.tensor(np.stack(weights_from_seed(tr['mlp_seed'][idx])),device='cuda')
        T=torch.tensor(tr['all_layer_means'][idx],device='cuda',dtype=torch.float32)
        U=torch.tensor(ch['U'],device='cuda');A=[U[l].T@W[l+1] for l in range(15)]
        C=torch.zeros(16,1024,device='cuda');C[1:]=torch.stack([T[l]@W[l+1] for l in range(15)])
        M=torch.zeros(2,len(KS),16,4,1024,device='cuda',dtype=torch.float64)
        S=torch.zeros(2,6,16,1024,device='cuda',dtype=torch.float64)
        Q=torch.zeros(2,16,1024,1024,device='cuda',dtype=torch.float64)
        g=torch.Generator(device='cuda').manual_seed(1000003+int(tr['mlp_seed'][idx])%1000000007)
        x=torch.empty(chunk,1024,device='cuda');t0=time.time()
        for ci in range(nch):
            torch.randn(x.shape,generator=g,out=x);h=x;half=ci%2
            for l in range(16):
                z=h@W[l];d=z-C[l];h=z.clamp_min(0);hc=h-T[l]
                S[half,:4,l]+=sums(d).double();S[half,4,l]+=h.sum(0).double();S[half,5,l]+=(h*h).sum(0).double()
                Q[half,l]+=(hc.T@hc).double()
                if l<15:
                    proj=hc@U[l]
                    for ki,k in enumerate(KS):
                        y=proj[:,:k]@A[l][:k]
                        M[half,ki,l+1]+=sums(y).double()
            if (ci+1)%20==0: print('MC',idx,ci+1,nch,round(time.time()-t0,1),flush=True)
        nh=nch//2*chunk
        np.savez(out,S=S.cpu().numpy(),M=M.cpu().numpy()/nh,Q=(Q/nh).float().cpu().numpy(),n=nh,ks=KS)
        print('MC_DONE',idx,2*nh,round(time.time()-t0,1),flush=True)

def residual(x,y):
    xc=x-x.mean(-1,keepdims=True)
    b=(xc*y).sum(-1,keepdims=True)/np.maximum((xc*xc).sum(-1,keepdims=True),1e-30)
    return y-y.mean(-1,keepdims=True)-b*xc

def prepare(a,idx):
    ch=np.load(a.out/f'chain{idx}.npz');mc=np.load(a.out/f'mc{idx}.npz')
    raw=mc['S']/float(mc['n']);v=raw[:,1]-raw[:,0]**2
    k3,k4=cumulants(raw[:,:4].transpose(0,2,1,3));s3,s4=cumulants(mc['M'])
    tr=load_truth('mini');W=np.stack(weights_from_seed(tr['mlp_seed'][idx]));diag=np.zeros_like(ch['k4'])
    for l in range(1,16):diag[l]=ch['post4'][l-1]@(W[l]**4)
    r3=residual(ch['d3'],k3);r4=residual(ch['k4'],k4)
    # Remove the same chain-affine nuisance from each projected tensor contraction.
    q3=residual(ch['d3'][None,None],s3);q4=residual(ch['k4'][None,None],s4+diag[None,None])
    for x in (q3,q4):x[:,:,0]=0
    for x in (r3,r4):x[:,0]=0
    return ch,v,k3,k4,r3,r4,q3,q4

def oracle(a):
    m=lean_module();torch.set_default_device('cpu');torch.set_default_dtype(torch.float64);statics=m.build_static_coefs(1024)
    torch.set_default_dtype(torch.float32);torch.set_default_device('cuda');tr=load_truth('mini');tot={};base=0;rows=[]
    for idx in a.idx:
        ch,v,k3,k4,r3,r4,q3,q4=prepare(a,idx)
        W=np.stack(weights_from_seed(tr['mlp_seed'][idx]));gt=tr['all_layer_means'][idx].astype(float)
        err=ch['P'].astype(float)-gt;b=float(np.mean(err[-1]**2));base+=b
        for ki,k in enumerate(KS):
            for name,target,pred in [('k3',r3,q3[:,ki]),('k4',r4,q4[:,ki])]:
                den=np.mean(target[0]*target[1],axis=-1)
                eps=np.mean((target[0]-pred[0])*(target[1]-pred[1]),axis=-1)
                cross=.5*np.mean(target[0]*pred[1]+target[1]*pred[0],axis=-1)
                pp=np.mean(pred[0]*pred[1],axis=-1)
                rows.append(dict(idx=idx,k=k,name=name,share=(1-eps/np.maximum(den,1e-30)).tolist(),cross=(cross/np.maximum(den,1e-30)).tolist(),predpower=(pp/np.maximum(den,1e-30)).tolist()))
                print('SHARE',idx,k,name,' '.join(f'{1-eps[l]/max(den[l],1e-30):.3f}' for l in (3,6,9,12,15)),flush=True)
        modes=['base','k4true','k34true','r34true']+[f'{n}:{k}' for k in KS for n in ('s4','s34','v+s34')]
        for mode in modes:
            ps=[]
            for half in range(2):
                corr={};repl={}
                if mode=='k4true':repl['k4']=k4[half]
                elif mode=='k34true':repl.update(k3=k3[half],k4=k4[half])
                elif mode=='r34true':corr.update(k3=r3[half],k4=r4[half])
                elif ':' in mode:
                    typ,ks=mode.split(':');tok=set(typ.split('+'));ki=KS.index(int(ks));corr['k4']=q4[half,ki]
                    assert tok <= {'s4','s34','v'}
                    if 's34' in tok:corr['k3']=q3[half,ki]
                    if 'v' in tok:repl['var']=v[half]
                corr={k:torch.tensor(x,device='cuda',dtype=torch.float32) for k,x in corr.items()}
                repl={k:torch.tensor(x,device='cuda',dtype=torch.float32) for k,x in repl.items()}
                def hook(l,name,x):
                    if l==0:return x
                    if name in repl:return repl[name][l]
                    if name in corr:return x+corr[name][l]
                    return x
                m.HOOK=hook
                pred=m.lean_k3_predict(W,dtype=torch.float32,statics=statics,k4_aug=True)
                if isinstance(pred,tuple):pred=pred[0]
                ps.append(pred.cpu().numpy().astype(float)-gt)
            value=float(np.mean(ps[0][-1]*ps[1][-1]));tot[mode]=tot.get(mode,0)+value
            np.savez(a.out/f'oracle{idx}_{mode.replace(":","_")}.npz',A=ps[0],B=ps[1])
            print('ORACLE',idx,mode,value/b,flush=True)
    result=dict(base=base,ratios={k:v/base for k,v in tot.items()},shares=rows)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2));print('TOTAL',json.dumps(result['ratios']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','mc','oracle']);p.add_argument('--idx',default='0,1,2,3');p.add_argument('--out',type=Path,required=True);p.add_argument('--n',type=float,default=2e7);p.add_argument('--chunk',type=int,default=16384)
    a=p.parse_args();a.idx=[int(i) for i in a.idx.split(',')];a.out.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.set_num_threads(6);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    globals()[a.stage](a)
