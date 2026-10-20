"""Small chain reconstruction check: splitting births preserves exact chain."""
import argparse,json
from pathlib import Path
from probe import ConditionalProbe,f,torch,np
p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(123)
n=16;K=f.coerce_input({1:torch.zeros(n,dtype=torch.float64),2:torch.eye(n,dtype=torch.float64)},k_max=4,kind=f.Kind.SIMPLE)
f.SPEC=f.parse_spec('slice4+slice31,!slice22+slice211')
rec=ConditionalProbe(a.mode) if a.mode!='original' else None
if rec:rec.install()
pred=[]
for l in range(6):
    if rec:rec.layer=l
    W=torch.randn(n,n,dtype=torch.float64)*(2/n)**.5
    K=f.linear_kprop(K,W,k_max=4,set_metric=2.)
    if rec:rec.start_layer()
    K=f.nonlin(K,nonlin_wick_coef=f.WICK_COEF_D['relu'])
    pred.append(K[1].to_tensor().numpy().copy())
np.save(a.out,np.stack(pred));print(json.dumps(dict(mode=a.mode,finite=bool(np.isfinite(pred).all()))))
