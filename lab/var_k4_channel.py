"""Full-K4 teacher and restricted output-channel oracles.
No fitted truth enters the teacher or its image directions. Full replacements,
raw differences, and chain-affine residuals are separate named experiments.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ('tools', 'refs/ref_0709/src', 'refs/504aldo/lean'):
    sys.path.insert(0, str(ROOT / p))
os.environ.setdefault('AW_DATA', str(ROOT / 'data'))
import numpy as np
import torch
from mlpdata import load_truth, weights_from_seed
from var_sens import lean_c
from var_subspace import residual, cumulants


def reference(a):
    sys.path.insert(0,str(ROOT / "refs/mlp_cumulant_propagation/src"))
    import k4full as f
    tr = load_truth('mini')
    original = f.nonlin
    for idx in a.idx:
        dest = a.out / f'{a.tag}{idx}.npz'
        if dest.exists():
            continue
        rec = dict(C=[], k4=[], mu=[])
        if a.forward_rank:
            b=np.load(a.out/f'base{idx}.npz')
            _,Vs=maps(np.stack(weights_from_seed(tr['mlp_seed'][idx])),b['mu'],b['var'],a.device)
        def capture(K, **kw):
            l=len(rec['C'])
            if a.forward_rank and l and 4 in K:
                U=Vs[l][:a.forward_rank].T.to(dtype=torch.float32)
                # All four transport legs share the observable basis.  This is a
                # dense accuracy simulation; portable core costs are separate.
                old=K[4]
                K[4]=f.SK4(old.n,[g.moved(U@(U.T@g.Q)) for g in old.groups],
                            None,old.device,old.dtype)
            rec['C'].append(K[2].core.detach().cpu().numpy().copy())
            rec['mu'].append(K[1].core.detach().cpu().numpy().copy())
            k4 = K[4].get_dslice((4,)) if 4 in K else torch.zeros_like(K[1].core)
            rec['k4'].append(k4.detach().cpu().numpy().copy())
            print('REF_LAYER', idx, len(rec['C'])-1, f._mem(), flush=True)
            return original(K, **kw)
        f.nonlin = capture
        f.STATS.update(groups=0, birth_bytes=0, dslice_s=0.0)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        P = f.run_chain(weights_from_seed(tr['mlp_seed'][idx]), a.device, torch.float32)
        np.savez_compressed(dest, P=P, **{k: np.stack(v) for k,v in rec.items()})
        print('REF_DONE', idx, 'mse', np.mean((P[-1]-tr['all_layer_means'][idx,-1])**2),
              'seconds', time.time()-t0, f._mem(), flush=True)
        del rec, P
        torch.cuda.empty_cache()
    f.nonlin = original


def maps(W, mu, var, device, rank=64):
    W = torch.as_tensor(W, device=device, dtype=torch.float64)
    mu = torch.as_tensor(mu, device=device, dtype=torch.float64)
    var = torch.as_tensor(var, device=device, dtype=torch.float64)
    s = var.sqrt(); al = mu/s
    phi = torch.exp(-al.square()/2)/np.sqrt(2*np.pi)
    c4 = (al.square()-1)*phi/(24*s**3)
    Phi = torch.special.ndtr(al)
    J = torch.eye(W.shape[-1], device=device, dtype=torch.float64)
    Ms, Vs = {}, {}
    for l in range(len(W)-1, 0, -1):
        Ms[l] = J*c4[l][None,:]
        Vs[l] = torch.linalg.svd(Ms[l], full_matrices=False).Vh[:rank]
        J = J @ (Phi[l][:,None]*W[l].T)
    return Ms, Vs


def oracle(a):
    m = lean_c(); torch.set_default_device('cpu'); torch.set_default_dtype(torch.float64)
    st = m.build_static_coefs(1024)
    torch.set_default_device(a.device); torch.set_default_dtype(torch.float32)
    tr = load_truth('mini'); acc = {}; base_sum = 0.; rows = []
    modes = a.modes.split(',')
    valid = {'base','C','k4','k4res','proj','late'}
    for mode in modes:
        assert set(mode.split('+')) <= valid, mode
    for idx in a.idx:
        W = np.stack(weights_from_seed(tr['mlp_seed'][idx])); gt = tr['all_layer_means'][idx].astype(float)
        r = np.load(a.out/f'{a.teacher or a.tag}{idx}.npz'); rec = {}
        def capture(l, name, x):
            rec[name,l] = x.detach().cpu().numpy().copy()
            return x
        m.HOOK = capture
        out = m.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True, return_diag=True)
        P, mu, var = [x.detach().cpu().numpy() for x in out[:3]]
        k4base = np.stack([rec.get(('k4',l), np.zeros(1024)) for l in range(16)])
        np.savez_compressed(a.out/f'base{idx}.npz',P=P, mu=mu,var=var,k4=k4base)
        err = P.astype(float)-gt; b = np.mean(err[-1]**2); base_sum += b
        Ms, Vs = maps(W,mu,var,a.device,a.readout_rank)
        Cref = torch.as_tensor(r['C'],device=a.device)
        kr = torch.as_tensor(r['k4'],device=a.device)
        rr = residual(k4base,r['k4']); rr[0]=0
        rr = torch.as_tensor(rr,device=a.device,dtype=torch.float32)
        for mode in modes:
            dest=a.out/f'{a.tag}_oracle{idx}_{mode}.npz'
            tok=set(mode.split('+')); start=9 if 'late' in tok else 1
            def hook(l,name,x):
                if l < 1:return x
                if name=='C' and 'C' in tok:return Cref[l].to(x)
                if name=='k4' and l >= start:
                    if 'k4res' in tok: delta=rr[l].to(x)
                    elif 'k4' in tok: delta=kr[l].to(x)-x
                    else:return x
                    if 'proj' in tok:
                        V=Vs[l].to(x); delta=V.T@(V@delta)
                    return x+delta
                return x
            if mode=='base': pred=P
            else:
                m.HOOK=hook; t0=time.time()
                pred=m.lean_k3_predict(W,dtype=torch.float32,statics=st,k4_aug=True)
                pred=(pred[0] if isinstance(pred,tuple) else pred).cpu().numpy()
            e=pred.astype(float)-gt; val=float(np.mean(e[-1]**2)); acc[mode]=acc.get(mode,0.)+val
            np.savez_compressed(dest,P=pred,error=e)
            rows.append(dict(idx=idx,mode=mode,mse=val,base=float(b),ratio=val/b))
            print('ORACLE',idx,mode,'mse',val,'ratio',val/b,flush=True)
        # Dual test: exact SVD of sum_i v_i e_i^4 at the readout (pair flattening).
        # Rank-R is precisely the R largest |v_i| entries.  Sources remain oracle;
        # this measures compression alone, not a deployable source generator.
        for rank in ((1,2,4,8,16,64) if a.tag == "ref" else ()):
            dx=torch.zeros_like(rr); retained=[]; lower=0.
            for l in range(9,16):
                V=Vs[l]; pos=V.abs().topk(rank,dim=1).indices
                test=torch.zeros_like(V).scatter(1,pos,V.gather(1,pos))
                dx[l]=(V.T@(test@rr[l].double())).float()
                retained.append(float(test.square().sum()/V.square().sum()))
                lower+=l*pos.unique().numel()/1024
            u=sum(Ms[l]@dx[l].double() for l in range(9,16))
            ee=torch.as_tensor(err[-1],device=a.device)
            val=float((ee+u).square().mean());key=f'dual_r{rank}_linear'
            acc[key]=acc.get(key,0.)+val
            rows.append(dict(idx=idx,mode=key,ratio=val/b,test_power=retained,
                             transport_only_units=lower,source_cost_included=False))
            np.savez_compressed(a.out/f'dual{idx}_r{rank}.npz',delta=dx.cpu().numpy())
            print('DUAL_SCREEN',idx,rank,'ratio',val/b,'transport_floor_units',lower,flush=True)
        # Teacher's affine residual in exactly the old lin/img maps, no Monte Carlo re-run.
        mcpath=a.out/'mc_S_0_3.npz'
        if mcpath.exists() and a.tag == "ref":
            mc=np.load(mcpath); pos=list(mc['idx']).index(idx)
            raw=mc['S'][pos]/float(mc['n'][pos])
            _,mc4=cumulants(raw[:,:4].transpose(0,2,1,3))
            mr=residual(k4base,mc4);mr[:,0]=0
            ua=sum(Ms[l]@torch.as_tensor(mr[0,l],device=a.device) for l in Ms)
            ub=sum(Ms[l]@torch.as_tensor(mr[1,l],device=a.device) for l in Ms)
            up=sum(Ms[l]@rr[l].double() for l in Ms)
            share=float(1-((ua-up)*(ub-up)).sum()/(ua*ub).sum())
            rows.append(dict(idx=idx,teacher_mc_image_share=share))
            print('TEACHER_MC_IMAGE_SHARE',idx,share,flush=True)
            for late in (False,True):
                ps=[]
                for h in range(2):
                    def mchook(l,name,x):
                        if name=='k4' and l >= (9 if late else 1):
                            V=Vs[l].to(x); delta=torch.as_tensor(mr[h,l],device=a.device,dtype=x.dtype)
                            return x+V.T@(V@delta)
                        return x
                    m.HOOK=mchook
                    outmc=m.lean_k3_predict(W,dtype=torch.float32,statics=st,k4_aug=True)
                    pm=(outmc[0] if isinstance(outmc,tuple) else outmc).cpu().numpy().astype(float)
                    ps.append(pm-gt)
                val=float(np.mean(ps[0][-1]*ps[1][-1]));key=f'mc_res_proj_late{int(late)}'
                acc[key]=acc.get(key,0.)+val
                rows.append(dict(idx=idx,mode=key,mse=val,base=float(b),ratio=val/b))
                np.savez_compressed(a.out/f'oracle{idx}_{key}.npz',A=ps[0],B=ps[1])
                print('MATCHED_MC',idx,key,val/b,flush=True)
        for late in (False,True):
            for proj in (False,True):
                u=torch.zeros(1024,device=a.device,dtype=torch.float64)
                for l in range(9 if late else 1,16):
                    x=rr[l].double()
                    if proj:x=Vs[l].T@(Vs[l]@x)
                    u+=Ms[l]@x
                ee=torch.as_tensor(err[-1],device=a.device)
                key=f'linear_res_late{int(late)}_proj{int(proj)}'
                val=float((ee+u).square().mean());acc[key]=acc.get(key,0.)+val
                print('IMAGE',idx,key,val/b,flush=True)
        del Cref, kr, rr, rec, out
        torch.cuda.empty_cache()
    result=dict(readout_rank=a.readout_rank,teacher=a.teacher or a.tag,base_mse=base_sum/len(a.idx),ratios={k:v/base_sum for k,v in acc.items()},rows=rows)
    (a.out/f'{a.tag}_oracle_summary.json').write_text(json.dumps(result,indent=2))
    print('SUMMARY',json.dumps(result['ratios']),flush=True)


def dual(a):
    m=lean_c();torch.set_default_device('cpu');torch.set_default_dtype(torch.float64)
    st=m.build_static_coefs(1024);torch.set_default_device(a.device);torch.set_default_dtype(torch.float32)
    tr=load_truth('mini');rows=[];acc={};bsum=0.
    for idx in a.idx:
        W=np.stack(weights_from_seed(tr['mlp_seed'][idx]));gt=tr['all_layer_means'][idx].astype(float)
        b=np.load(a.out/f'base{idx}.npz');base=float(np.mean((b['P'][-1]-gt[-1])**2));bsum+=base
        C=torch.as_tensor(np.load(a.out/f'ref{idx}.npz')['C'],device=a.device)
        for rank in (4,8,64):
            delta=torch.as_tensor(np.load(a.out/f'dual{idx}_r{rank}.npz')['delta'],device=a.device)
            for useC in (False,True):
                def hook(l,name,x):
                    if not l:return x
                    if useC and name=='C':return C[l].to(x)
                    if name=='k4':return x+delta[l].to(x)
                    return x
                m.HOOK=hook
                pred=m.lean_k3_predict(W,dtype=torch.float32,statics=st,k4_aug=True)
                pred=(pred[0] if isinstance(pred,tuple) else pred).cpu().numpy().astype(float)
                val=float(np.mean((pred[-1]-gt[-1])**2));key=f'dual{rank}'+('+C' if useC else '')
                acc[key]=acc.get(key,0.)+val;rows.append(dict(idx=idx,mode=key,mse=val,base=base,ratio=val/base))
                np.savez_compressed(a.out/f'dual_run{idx}_{key}.npz',P=pred)
                print('DUAL_RERUN',idx,key,val/base,flush=True)
    result=dict(base_mse=bsum/len(a.idx),ratios={k:v/bsum for k,v in acc.items()},rows=rows)
    (a.out/'dual_summary.json').write_text(json.dumps(result,indent=2));print('DUAL_SUMMARY',result['ratios'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reference','oracle','dual']);p.add_argument('--idx',default='0,1,2,3')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--teacher',default=None);p.add_argument('--readout-rank',type=int,default=64)
    p.add_argument('--forward-rank',type=int,default=0);p.add_argument('--tag',default='ref')
    p.add_argument('--modes',default='base,k4+proj+late,k4res+proj+late,k4res+proj,k4,k4res,C,C+k4,C+k4res,C+k4+proj+late,C+k4res+proj+late')
    a=p.parse_args();a.idx=[int(x) for x in a.idx.split(',')];a.out.mkdir(parents=True,exist_ok=True)
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    globals()[a.stage](a)

if __name__=='__main__':main()
