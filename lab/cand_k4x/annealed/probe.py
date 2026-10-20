"""Cloud reference experiment: average only current-W non-tree readouts."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
for folder in ('tools','refs/mlp_cumulant_propagation/src','lab','lab/cand_k4x'):
    sys.path.insert(0,str(ROOT/folder))
os.environ.setdefault('AW_SPEC','star,three2,slice4+slice31,!slice22+slice211')
import b_suffix as base
from tree_birth import TreeBirth
from k4n3_young_direct_chain import tree_tensor
from wick import expected,add_expected
np,torch,f=base.np,base.torch,base.f
METER=None


class NonTreeGroup(f.Group):
    __slots__=('born','conditional','slope')
    def __init__(self,types,crossed,xb,yb,Bb,Q,born,conditional=None,slope=None):
        super().__init__(types,crossed,xb,yb,Bb,Q)
        self.born,self.conditional,self.slope=born,conditional,slope
    def moved(self,Q):
        return NonTreeGroup(self.types,self.crossed,self.xb,self.yb,self.Bb,Q,self.born,self.conditional,self.slope)


class ConditionalProbe(TreeBirth):
    def __init__(self,mode):
        super().__init__();self.mode=mode;self.layer=-1;self.rows=[];self.birth_rows=[]
        self.pending=None;self.trace_seconds=0.;self.trace_calls=0;self.zero_first_reads=0
    def split_birth(self,ds):
        total=ds.slices[(2,1,1)]
        tree=tree_tensor(self.current.trees,ds.n,total.device,total.dtype)
        residual=total-tree
        self.birth_rows.append(dict(layer=self.layer,total_norm=float(total.norm()),tree_norm=float(tree.norm()),non_tree_norm=float(residual.norm())))
        ds.slices[(2,1,1)]=tree
        self.pending=residual if self.mode!='tree' else None
    def install(self):
        super().install()
        rec=self;origfrom=self.saved_from
        def make(ds):
            out=origfrom(ds)
            if rec.pending is not None:
                B=6*rec.pending.permute(1,2,0)
                B=(B+B.transpose(0,1))/2
                Q=torch.eye(ds.n,device=ds.device,dtype=ds.dtype)
                out.groups.append(NonTreeGroup(('slice211',),False,None,None,f._store(B),Q,rec.layer))
                rec.pending=None
            return out
        self.saved_from=make
        oldcontract=f.SK4.contract_W
        def contract(sk,W):
            out=oldcontract(sk,W)
            for old,new in zip(sk.groups,out.groups):
                if isinstance(old,NonTreeGroup):
                    if rec.layer-old.born==1:
                        z=old.Q.new_zeros(())
                        new.conditional=(z,z,z,z)
                        rec.zero_first_reads+=1
                    else:
                        t0=time.monotonic()
                        new.conditional=expected(old.Q,old.Bb,2/W.shape[1],old.crossed)
                        rec.trace_seconds+=time.monotonic()-t0;rec.trace_calls+=1
                    new.slope=None
            return out
        f.SK4.contract_W=contract
        oldwick=f.SK4.contract_wick_
        def wick(sk,w):
            oldwick(sk,w)
            for g in sk.groups:
                if isinstance(g,NonTreeGroup) and g.conditional is not None:
                    g.slope=w if g.slope is None else g.slope*w
        f.SK4.contract_wick_=wick
        oldread=f.group_dslices
        def read(g,out):
            if not isinstance(g,NonTreeGroup) or g.conditional is None:
                return oldread(g,out)
            d4,d22,a,b=g.conditional
            if rec.mode=='annealed':
                add_expected(out,d4,d22,g.slope);return
            # Exact trajectory, with a paired conditional prediction at the
            # same quenched state. Record only the pre-nonlinear first read.
            tmp={p:torch.zeros_like(v) for p,v in out.items()}
            oldread(g,tmp)
            if g.slope is None:
                metrics={}
                for p,x in tmp.items():
                    x=f.zero_repeated(x)
                    if p==(4,): y=torch.full_like(x,d4)
                    elif p==(2,2):
                        y=torch.full_like(x,d22);y.fill_diagonal_(0)
                    else: y=None
                    xx=float(x.square().sum());xy=float((x*y).sum()) if y is not None else 0.
                    yy=float(y.square().sum()) if y is not None else 0.
                    metrics[str(p)]=dict(exact_norm2=xx,mean_norm2=yy,dot=xy,error_norm2=xx+yy-2*xy,
                        mean_energy_fraction=yy/max(xx,1e-300),relative_error=((xx+yy-2*xy)/max(xx,1e-300))**.5,
                        cosine=xy/max((xx*yy)**.5,1e-300))
                rec.rows.append(dict(layer=rec.layer,birth=g.born,age=rec.layer-g.born,
                    trace_a=float(a),trace_b=float(b),mean4=float(d4),mean22=float(d22),metrics=metrics))
            for p in out:out[p].add_(tmp[p])
        f.group_dslices=read


def seed(K,snapshot):
    inv=torch.as_tensor(snapshot['inv_perm'].astype('int64'),device='cuda')
    def unsort(name):
        v=torch.as_tensor(snapshot[name],device='cuda',dtype=torch.float32)
        for ax in range(v.ndim):v=v.index_select(ax,inv)
        return v
    C=unsort('C');C.fill_diagonal_(0);G=C*float(snapshot['lam_feed'])
    ds=f.DSTensor(d=4,n=1024,slices={(4,):unsort('g4row'),(3,1):unsort('wk431').T,
        (2,2):unsort('wk4m'),(2,1,1):G.unsqueeze(0).expand(1024,-1,-1).clone()/3},autozero=True,device=G.device,dtype=G.dtype)
    saved=f.SPEC;f.SPEC=f.parse_spec('slice4+slice31+slice22+slice211')
    K[4]=f.SK4.from_dstensor(ds);f.SPEC=saved


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0)
    p.add_argument('--mode',choices=('exact','tree','annealed'),required=True)
    p.add_argument('--snapshots',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--spec',default='slice4+slice31,!slice22+slice211')
    a=p.parse_args();torch.set_num_threads(4)
    tr=base.d.load_truth('mini');weights=base.d.weights_from_seed(tr['mlp_seed'][a.index])
    snapshot=np.load(a.snapshots/f'B_mini{a.index}_pre4.npz');t0=time.monotonic()
    mu,C,T,diag=base.expand(snapshot)
    K=f.coerce_input({1:mu,2:C,3:T},k_max=4,kind=f.Kind.SIMPLE);del mu,C,T
    f.SPEC=f.parse_spec(a.spec);seed(K,snapshot)
    rec=ConditionalProbe(a.mode);rec.install();means=list(snapshot['prefix']);layers=[]
    for layer in range(4,16):
        if METER is not None:METER.phase=f'layer{layer}'
        rec.layer=layer;f.LAST_LAYER=layer==15;t1=time.monotonic()
        if layer>4:
            w=torch.as_tensor(weights[layer].T.copy(),device='cuda',dtype=torch.float32)
            K=f.linear_kprop(K,w,k_max=4,set_metric=2.)
        if layer==15:f.SPEC=[]
        rec.start_layer();K=f.nonlin(K,nonlin_wick_coef=f.WICK_COEF_D['relu'])
        means.append(K[1].to_tensor().double().cpu().numpy())
        row=dict(layer=layer,seconds=time.monotonic()-t1,memory=f._mem(),
                 mse=float(np.mean((means[-1]-tr['all_layer_means'][a.index,layer])**2)))
        layers.append(row);print(json.dumps(row),flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.with_suffix('.partial.json').write_text(json.dumps(dict(layers=layers,paired=rec.rows,births=rec.birth_rows)))
    pred=np.stack(means);np.save(a.out.with_suffix('.npy'),pred)
    result=dict(index=a.index,mode=a.mode,spec=a.spec,start=4,bridge=diag,
        mse=layers[-1]['mse'],seconds=time.monotonic()-t0,layers=layers,paired=rec.rows,births=rec.birth_rows,
        trace_calls=rec.trace_calls,zero_first_reads=rec.zero_first_reads,trace_seconds=rec.trace_seconds,memory=f._mem(),
        semantics='actual B prefix and seed; average current W only in non-tree slice211-source contractions; all other operations quenched')
    if METER is not None:result['matrix_meter']=METER.result()
    a.out.write_text(json.dumps(result,indent=2)+'\n');print('RESULT',a.mode,a.index,result['mse'],flush=True)


if __name__=='__main__':
    from matrix_meter import MatrixMeter
    METER=MatrixMeter()
    with METER:main()
