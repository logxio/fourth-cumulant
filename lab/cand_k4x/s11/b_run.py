"""Actual cmp13 at n64 with suite K4 regeneration explicitly enabled."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,resource,signal,types,hashlib
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as f
from whestbench import MLP,SetupContext
from b_pair import BPair
ROOT=Path(__file__).resolve().parents[3]


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=5)
    p.add_argument('--k',type=int,default=0);p.add_argument('--age',type=int,default=1)
    p.add_argument('--plain',action='store_true',help='unhooked B with the same n64 regeneration override')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    signal.alarm(280)
    source=(ROOT/'lab/cand_cmp13/estimator.py').read_text()
    sha=hashlib.sha256(source.encode()).hexdigest()
    source=source.replace('riders = (n == 1024 and L == len(CORR_BETA))','riders = (L == len(CORR_BETA))')
    probe=BPair(a.k,a.age)
    replacements=[
      ('            # ---- WK slices ----','            _p.mark("transport", _p.transport, locals())\n            # ---- WK slices ----'),
      ('            WT = W_all.T','            g4row, wk4m, wk431, wk431r = _p.mark("pre", _p.pre, dict(locals(), SSC_K=SSC_K))\n            WT = W_all.T'),
      ('            S3c = K3v - D3_w if D3_w is not None else K3v','            _fd3, _fd21 = _p.mark("feed", _p.feed, locals())\n            S3c = K3v - D3_w if D3_w is not None else K3v'),
      ('            # S21 = S_sep + Rres:', '            S3c = S3c - _fd3\n            S21 = S21 - _fd21\n            # S21 = S_sep + Rres:'),
      ('            K4_sigma = (fnp.sum(K4v)','            g_prev = _p.mark("post", _p.post, locals())\n            K4_sigma = (fnp.sum(K4v)')]
    for old,new in ([] if a.plain else replacements):
        assert source.count(old)==1,old
        source=source.replace(old,new)
    mod=types.ModuleType('b_pair_probe');mod.__dict__['_p']=probe
    exec(compile(source,'b_pair_probe','exec'),mod.__dict__)
    original=mod.Estimator._dslices
    def ds(self,*args,**kwargs):
        return probe.mark('k3_read',probe.add_k3,original(self,*args,**kwargs))
    if not a.plain:mod.Estimator._dslices=ds
    n=64;budget=2**50;rng=np.random.default_rng(a.seed)
    weights=[(rng.standard_normal((n,n))*(2/n)**.5).astype('f') for _ in range(16)]
    est=mod.Estimator();est.setup(SetupContext(width=n,depth=16,flop_budget=budget,api_version='1',submission_dir=str(Path(__file__).parent)))
    model=MLP(width=n,depth=16,weights=[f.array(w,dtype=f.float32) for w in weights],seed=a.seed,name=f'n64-seed{a.seed}')
    ctx=flops.BudgetContext(flop_budget=budget,quiet=True);probe.ctx=ctx;start=time.monotonic()
    try:
        with ctx:pred=np.asarray(est.predict(model,budget)).astype(float)
        assert pred.shape==(16,n)
        if not np.isfinite(pred).all():raise FloatingPointError('non-finite final prediction')
    except Exception as exc:
        result=dict(n=n,seed=a.seed,k=a.k,age=a.age,status='failed',error=repr(exc),flops=ctx.flops_used,
                    width_units=ctx.flops_used/(2*n**3),sidecar_flops=probe.cost,rows=probe.rows,
                    wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    base_sha256=sha,cost_scope='partial bill through failure; lower bound on a completed chain')
        a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return
    tpath=ROOT/'outputs/research/k4s11'/('full_n64_mc_r1.npz' if a.seed==5 else f'full_n64_seed{a.seed}_mc_r1.npz')
    gt=np.load(tpath)['pred'];raw=float(np.mean((pred[-1]-gt[-1])**2))
    result=dict(n=n,seed=a.seed,k=a.k,age=a.age,status='ok',raw=raw,flops=ctx.flops_used,
                width_units=ctx.flops_used/(2*n**3),sidecar_flops=probe.cost,
                wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                rows=probe.rows,base_sha256=sha,
                changes_from_1024='suite regeneration enabled at n64; row pruning and old shared basis disabled by shape/rank guards; young atom cap remains enabled (Y_EPS=0.005); R_RES=8,R_FB=4 retained; frozen LAM/SSC retained')
    a.out.write_text(json.dumps(result,indent=2)+'\n');np.save(a.out.with_suffix('.npy'),pred)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
