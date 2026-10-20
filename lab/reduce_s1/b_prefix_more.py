"""reduce_s1 copy of lab/cand_k4x/b_prefix.py with the export layers as an argument (--layers 4,10,12,13).

Cloud research only: export actual B preactivation states without editing B.

Snapshots contain the implicit symmetric K3 factorization and its evaluated
repeated slices.  They are not permitted submission data or estimator code.
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'lab'/'cand_k4x'))
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext
from mlpdata import load_truth, weights_from_seed, estimator_seed
import base_b as b


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',default='4,6,8')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    layers=tuple(int(x) for x in a.layers.split(','))
    original=b.Estimator._dslices
    pending={}
    signature=inspect.signature(original)
    def capture(self,*args,**kw):
        caller=inspect.currentframe().f_back.f_locals
        result=original(self,*args,**kw)
        layer=caller['li']
        if layer in layers:
            values=signature.bind(self,*args,**kw).arguments
            def array(x): return np.asarray(x).copy()
            names=('A_st','P_st','Z_st','L_st','w2b_list','s_list','e_list',
                   'c1_list','c2_list','Zf_st','R1T_st','R2T_st')
            data={k:array(values[k]) if not isinstance(values[k],list)
                  else np.stack([array(x) for x in values[k]])
                  for k in names if values.get(k) is not None}
            # B's _zero_diag aliases C_pre; its saved var is the diagonal.
            covariance=array(caller['C_pre'])
            np.fill_diagonal(covariance,array(caller['var']))
            data.update(mu=array(caller['mu']),C=covariance,
                        D3=array(result[0]),D21=array(result[1]),
                        inv_perm=array(caller['inv_perm']) if caller['inv_perm'] is not None
                                 else np.arange(1024),
                        prefix=np.stack([array(x) for x in caller['rows']]),
                        rres=values['rres'],rfb=values['rfb'])
            np.savez(a.out/f'B_mini{a.index}_pre{layer}.npz',**data)
            pending[layer]=data
            print('CAPTURE',a.index,layer,data['A_st'].shape,flush=True)
        return result
    b.Estimator._dslices=capture
    original_pdf=b.flops.stats.norm.pdf
    def capture_pdf(*args,**kwargs):
        caller=inspect.currentframe().f_back.f_locals
        layer=caller.get('li')
        if layer in pending and caller.get('g4row') is not None:
            data=pending.pop(layer)
            for name in ('g4row','wk4m','wk431'):
                data[name]=np.asarray(caller[name]).copy()
            data['lam_feed']=float(caller['lam_prev'])*b.SSC_K['lam_feed'][layer]
            np.savez(a.out/f'B_mini{a.index}_pre{layer}.npz',**data)
            print('K4_SLICES',a.index,layer,flush=True)
        return original_pdf(*args,**kwargs)
    b.flops.stats.norm.pdf=capture_pdf
    tr=load_truth('mini'); budget=2**41
    est=b.Estimator()
    est.setup(SetupContext(width=1024,depth=16,flop_budget=budget,api_version='1',
                           submission_dir=str(ROOT/'lab'/'cand_k4x')))
    seed=tr['mlp_seed'][a.index]
    model=MLP(width=1024,depth=16,weights=[fnp.array(w,dtype=fnp.float32)
              for w in weights_from_seed(seed)],seed=estimator_seed(seed),
              name=str(tr['mlp_name'][a.index]))
    ctx=flops.BudgetContext(flop_budget=budget,quiet=True)
    with ctx:
        pred=np.asarray(est.predict(model,budget)).astype(float)
    np.save(a.out/f'B_mini{a.index}.npy',pred)
    result=dict(index=a.index,mse=float(np.mean((pred[-1]-tr['all_layer_means'][a.index,-1])**2)),
                units=ctx.flops_used/2**31,note='research export; residual and timing invalidated by snapshots')
    (a.out/f'B_mini{a.index}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('B_RESULT',result,flush=True)


if __name__=='__main__':main()
