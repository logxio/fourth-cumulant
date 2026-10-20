"""Continue actual B preactivation (mu,C,K3) with research K4 births.

K4 starts empty at the switch.  The exact B implicit K3 factors are expanded
on the cloud GPU for this accuracy probe, never inside a scoring artifact.
"""
import argparse
import json
import time
from pathlib import Path
import os
os.environ.setdefault('AW_SPEC','star,three2,slice4+slice31,!slice22+slice211')
import delayed_reference as d
from slice_age import SliceAges
np,torch,f=d.np,d.torch,d.f
METER=None


def expand(snapshot, device='cuda'):
    inv=torch.as_tensor(snapshot['inv_perm'].astype('int64'),device=device)
    def array(name):
        return torch.as_tensor(snapshot[name],device=device,dtype=torch.float32)
    A,P,Z,L=(array(k) for k in ('A_st','P_st','Z_st','L_st'))
    A,P,Z=(x.index_select(1,inv) for x in (A,P,Z))
    w2,s,e,c1,c2=(array(k) for k in ('w2b_list','s_list','e_list','c1_list','c2_list'))
    rres,rfb=int(snapshot['rres']),int(snapshot['rfb'])
    n=P.shape[1]
    T=torch.zeros((n,n,n),device=device,dtype=torch.float32)
    def add(x,y,z):
        raw=torch.einsum('ir,jr,kr->ijk',x,y,z)
        for order in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)):
            T.add_(raw.permute(order),alpha=1/6)
    for k in range(P.shape[0]):
        M=P[k]*s[k]+3*A[k]*e[k]+Z[k]@L[k].T
        X,Y=3*A[k],A[k]*w2[k]
        if rfb:
            Zf=array('Zf_st')[k].index_select(0,inv)
            X=X+Zf[:,:rfb]@array('R1T_st')[k].T
            Y=Y+Zf[:,rfb:]@array('R2T_st')[k].T
        add(X,P[k],Y)
        add(M,P[k],P[k])
        add(A[k]*c1[k]+P[k]*c2[k],P[k],Z[k,:,rres+1,None].expand_as(P[k]))
    D3=array('D3').index_select(0,inv)
    D21=array('D21').index_select(0,inv).index_select(1,inv)
    i=torch.arange(n,device=device)
    got3=T[i,i,i]
    got21=T[i,i,:].clone();got21[i,i]=0
    # The B dslice evaluator deliberately clips some D21 rows/columns, while
    # the factors define the all-distinct tensor. Preserve its actual interface.
    diagnostics=dict(d3_max_abs=float((got3-D3).abs().max()),
                     d3_relative_norm=float((got3-D3).norm()/D3.norm().clamp_min(1e-30)),
                     d21_max_abs_before_interface_clip=float((got21-D21).abs().max()))
    assert diagnostics['d3_relative_norm'] < 1e-3, diagnostics
    T[i,i,:]=D21;T[i,:,i]=D21;T[:,i,i]=D21.T;T[i,i,i]=D3
    return (array('mu').index_select(0,inv),
            array('C').index_select(0,inv).index_select(1,inv),T,diagnostics)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--start',type=int,required=True)
    p.add_argument('--snapshots',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--old-slice-rank',type=int,default=0)
    p.add_argument('--old-source-rank',type=int,default=0)
    p.add_argument('--max-slice-age',type=int,default=0,
                   help='drop slice groups before transport/read above this age; 0 = unlimited; B seed age is zero at switch')
    p.add_argument('--seed-b-slices',action='store_true')
    p.add_argument('--tree-only',action='store_true',
                   help='retain graph-factor trees in every slice211 birth, including pK-to-K conversion trees')
    p.add_argument('--non-tree-rank',type=int,default=0,
                   help='restore the non-tree birth residual projected on this covariance Krylov rank')
    p.add_argument('--spec',default='star,three2,slice4+slice31,!slice22+slice211')
    a=p.parse_args()
    if a.max_slice_age < 0:
        p.error('--max-slice-age must be nonnegative')
    torch.set_num_threads(4)
    tr=d.load_truth('mini');weights=d.weights_from_seed(tr['mlp_seed'][a.index])
    snapshot=np.load(a.snapshots/f'B_mini{a.index}_pre{a.start}.npz')
    t0=time.monotonic()
    if METER is not None: METER.phase='B_switch_expand'
    mu,C,T,diag=expand(snapshot)
    assert float(C.diag().min()) > 0, 'B snapshot covariance diagonal missing'
    print('BRIDGE',a.index,a.start,diag,flush=True)
    K=f.coerce_input({1:mu,2:C,3:T},k_max=4,kind=f.Kind.SIMPLE)
    del mu,C,T
    f.SPEC=f.parse_spec(a.spec)
    if a.seed_b_slices:
        if METER is not None: METER.phase='B_K4_seed'
        inv=torch.as_tensor(snapshot['inv_perm'].astype('int64'),device='cuda')
        def unsort(name):
            v=torch.as_tensor(snapshot[name],device='cuda',dtype=torch.float32)
            for axis in range(v.ndim):
                v=v.index_select(axis,inv)
            return v
        # Preserve B's actual K4 (4), (3,1), (2,2) consumers. Its independently
        # calibrated (2,1,1) feed is approximated by its lambda_feed radial core.
        # This is a specified switch seed, not an exact full K4 tensor of B.
        Cseed=unsort('C');Cseed.fill_diagonal_(0)
        G=Cseed*float(snapshot['lam_feed'])
        d211=G.unsqueeze(0).expand(1024,-1,-1).clone()/3
        ds=f.DSTensor(d=4,n=1024,slices={(4,):unsort('g4row'),
                      (3,1):unsort('wk431').T,(2,2):unsort('wk4m'),
                      (2,1,1):d211},autozero=True,device=G.device,dtype=torch.float32)
        saved=f.SPEC
        f.SPEC=f.parse_spec('slice4+slice31+slice22+slice211')
        K[4]=f.SK4.from_dstensor(ds)
        f.SPEC=saved
        del ds,d211,G,Cseed
    means=list(snapshot['prefix']);layers=[]
    ages=SliceAges(a.start,a.max_slice_age,K[4].groups)
    tree_rec=None
    if a.tree_only:
        from tree_birth import TreeBirth
        tree_rec=TreeBirth(a.non_tree_rank)
        tree_rec.install()
    for layer in range(a.start,16):
        if METER is not None: METER.phase=f'layer{layer}/transport'
        f.LAST_LAYER=layer==15
        dropped=ages.prune(layer,K[4])
        if layer>a.start:
            w=torch.as_tensor(weights[layer].T.copy(),device='cuda',dtype=torch.float32)
            K=f.linear_kprop(K,w,k_max=4,set_metric=2.)
        if layer==15:
            f.SPEC=[] # output mean has no dependency on final K4 births
        t1=time.monotonic()
        if METER is not None: METER.phase=f'layer{layer}/nonlinear'
        if tree_rec is not None:
            tree_rec.start_layer()
        K=f.nonlin(K,nonlin_wick_coef=f.WICK_COEF_D['relu'])
        ages.register(layer,K[4].groups)
        projection={}
        for kind,rank in (('slice',a.old_slice_rank),('source',a.old_source_rank)):
            # Project after the age-1 nonlinear step, before its age-2 read.
            old=[g for g in K[4].groups if layer-ages.births[id(g.Bb)]>=1
                 and g.types[0].startswith('slice')==(kind=='slice')]
            if rank and old:
                gram=sum(g.Q@g.Q.T for g in old)
                _,v=torch.linalg.eigh((gram+gram.T)*.5)
                q=v[:,-rank:].contiguous()
                lost=total=0.
                for g in old:
                    approx=q@(q.T@g.Q)
                    lost+=float((approx-g.Q).square().sum())
                    total+=float(g.Q.square().sum())
                    g.Q=approx
                projection[kind]=dict(rank=rank,members=len(old),
                                     relative_loss=(lost/max(total,1e-30))**.5)
            del old
        means.append(K[1].to_tensor().double().cpu().numpy())
        row=dict(layer=layer,seconds=time.monotonic()-t1,memory=f._mem(),projection=projection,
                 dropped_slices=dropped,slice_groups=sum(g.types[0].startswith('slice') for g in K[4].groups),
                 tree_birth=tree_rec.last if tree_rec is not None else None,
                 mse=float(np.mean((means[-1]-tr['all_layer_means'][a.index,layer])**2)))
        layers.append(row);print(json.dumps(row),flush=True)
    pred=np.stack(means)
    result=dict(index=a.index,start=a.start,spec=a.spec,bridge=diag,
                tree_only=a.tree_only,non_tree_rank=a.non_tree_rank,
                max_slice_age=a.max_slice_age,seed_age_at_switch=0,
                old_slice_rank=a.old_slice_rank,old_source_rank=a.old_source_rank,
                seed_b_slices=a.seed_b_slices,
                prefix='actual B preactivation mu,C,K3; '+('B K4 slices and radial feed seed' if a.seed_b_slices else 'K4 empty at switch'),
                mse=float(np.mean((pred[-1]-tr['all_layer_means'][a.index,-1])**2)),
                per_layer_mse=np.mean((pred-tr['all_layer_means'][a.index])**2,axis=1).tolist(),
                seconds=time.monotonic()-t0,memory=f._mem(),layers=layers)
    if METER is not None: result['matrix_meter']=METER.result()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.save(a.output.with_suffix('.npy'),pred)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',json.dumps(result),flush=True)


if __name__=='__main__':
    if os.environ.get('K4_MATRIX_METER') == '1':
        from matrix_meter import MatrixMeter
        METER=MatrixMeter()
        with METER: main()
    else:
        main()
