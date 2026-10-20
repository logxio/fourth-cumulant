"""Recompute paired final errors and sourcewise conditional-energy statistics."""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--truth',type=Path,required=True);a=p.parse_args()
tr=np.load(a.truth);rows=[];energies={}
for index in (0,1):
    values={}
    for mode in ('exact','tree','annealed'):
        path=a.root/f'{mode}{index}.json'
        if not path.exists():continue
        j=json.loads(path.read_text());pred=np.load(path.with_suffix('.npy'))
        assert pred.shape==(16,1024) and np.isfinite(pred).all()
        raw=float(np.mean((pred[-1]-tr['all_layer_means'][index,-1])**2))
        assert math.isclose(raw,j['mse'],rel_tol=1e-10)
        values[mode]=raw
        if mode=='exact':
            agg=defaultdict(lambda:defaultdict(float))
            for r in j['paired']:
                for part,m in r['metrics'].items():
                    for group in (f"age{r['age']}/{part}",f"layer{r['layer']}/{part}",f'all/{part}'):
                        for key in ('exact_norm2','mean_norm2','error_norm2','dot'):agg[group][key]+=m[key]
                        agg[group]['sources']+=1
            result={}
            for group,m in agg.items():
                xx=m['exact_norm2'];yy=m['mean_norm2'];xy=m['dot']
                result[group]=dict(m,mean_energy_fraction=yy/max(xx,1e-300),
                    residual_energy_fraction=m['error_norm2']/max(xx,1e-300),
                    relative_error=(m['error_norm2']/max(xx,1e-300))**.5,
                    cosine=xy/max((xx*yy)**.5,1e-300))
            energies[str(index)]=result
    bp=a.root/'seed_states'/f'B_mini{index}.npy'
    if bp.exists():
        pred=np.load(bp);values['B']=float(np.mean((pred[-1]-tr['all_layer_means'][index,-1])**2))
        for mode in ('exact','tree','annealed'):
            path=a.root/f'{mode}{index}.npy'
            if path.exists():assert np.array_equal(np.load(path)[:4],pred[:4])
    if all(k in values for k in ('exact','tree','annealed','B')):
        values['non_tree_gain_recovered']=(values['tree']-values['annealed'])/(values['tree']-values['exact'])
        values['annealed_over_B']=values['annealed']/values['B']
        values['annealed_over_tree']=values['annealed']/values['tree']
    rows.append(dict(index=index,**values))
result=dict(rows=rows,sourcewise_energy=energies,
    energy_scope='sum of squared per-source contributions, with cross term reported; excludes interference between distinct sources',
    correlation_scope='cosine of paired sourcewise readouts; Pearson within one layer is undefined because conditional entries are constant')
(a.root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(rows,indent=2))
