"""Small-width dense diagnostic chains; never a production FLOP claim."""
from direction_probe import *
from b_pair import diag,off
class DensePair(LeanPair):
    def __init__(self,count,mode,k=2):
        super().__init__(count);self.mode=mode;self.k=k
    def transport(self,v):
        W=v['W'];self.current=[];self.final4=None;self.final3=None
        for born,u,b in self.pairs:
            if v['li']-born!=1:continue
            if v['last']:
                d=f.sum(W*W*u[None,:],axis=1)*f.sum((W@b)*W,axis=1)
                self.final4=d if self.final4 is None else self.final4+d
            else:self.current.append((born,(W*u[None,:])@W.T,(W@b)@W.T))
        if v['last']:
            for x,m in self.fb:
                d=(W@x)*f.sum((W@m)*W,axis=1);self.final3=d if self.final3 is None else self.final3+d
        else:self.fb=[(W@x,(W@m)@W.T) for x,m in self.fb]
    def pre(self,v):
        orig=tuple(v.get(key) for key in ('g4row','wk4m','wk431','wk431r'));self.pending=[];self.birth=15-self.k<=v['li']<15
        if v['last']:return (orig[0]+self.final4,)+orig[1:] if self.final4 is not None else orig
        q4=q31=q22=None
        for born,a,b in self.current:
            da,db=diag(a),diag(b);a4=da*db;a31=off((da[:,None]*b+db[:,None]*a)/2);a22=off((da[:,None]*db[None,:]+db[:,None]*da[None,:])/6)
            q4=a4 if q4 is None else q4+a4;q31=a31 if q31 is None else q31+a31;q22=a22 if q22 is None else q22+a22
            self.pending.extend([(da,b),(db,a)])
        if self.birth:
            self.sources=nsources(v,self.count or 100);self.C=np.array(v['C_off'],dtype=float);self.mean=np.array(v['mu'],dtype=float)
            self.dm=[(np.ones(v['n']),self.C*float(v['lam_prev'])*v['SSC_K']['lam_feed'][v['li']]/3)]+[(np.array(x,dtype=float)/6,np.array(m,dtype=float)) for x,m in self.pending]
        if q4 is None:return orig
        a,b,c,d=orig;return a+q4,b+q22,c+q31.T,d+q31
    def post(self,v):
        self.pairs=[]
        if self.birth:
            T=sum(source_tensor(x) for x in self.sources)
            wa=np.array(v['W_all'],dtype=float);p,w2=wa[:,4],wa[:,6];pk=np.array(v['pk1v'])
            F=kernel(T,self.dm,self.C,p,w2,wa[:,5]-2*pk*p,wa[:,7]-2*pk*w2)
            if self.mode=='exact':us=np.eye(v['n']);bs=F
            else:
                u=directions(F,self.mean,111+1009*v['li'])[self.mode];us=u[:,None];bs=(u@F.reshape(v['n'],-1)).reshape(1,v['n'],v['n'])
            self.pairs=[(v['li'],f.array(us[:,i],dtype=f.float32),f.array(b,dtype=f.float32)) for i,b in enumerate(bs)]
        return v['g_prev']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=5);ap.add_argument('--k',type=int,default=2);a=ap.parse_args();signal.alarm(280);start=time.monotonic()
    gt=np.load(ROOT/'outputs/research/k4s11'/('full_n64_mc_r1.npz' if a.seed==5 else f'full_n64_seed{a.seed}_mc_r1.npz'))['pred']
    bp=np.load(ROOT/f'outputs/research/k4s11/r2/plain_b_s{a.seed}.npy');base=float(np.mean((bp[-1]-gt[-1])**2));rows=[]
    for count in (0,1,2):
        for mode in ('exact','rsvd','power1','power2','mean'):
            pred,_=run(DensePair(count,mode,a.k),a.seed)
            raw=float(np.mean((pred[-1]-gt[-1])**2));row=dict(seed=a.seed,k=a.k,sources='all' if not count else count,direction=mode,raw=raw,raw_ratio=raw/base,finite=bool(np.isfinite(pred).all()))
            rows.append(row);np.save(OUT/f'chain_s{a.seed}_k{a.k}_c{count}_{mode}.npy',pred);print(json.dumps(row),flush=True)
            (OUT/f'chain_s{a.seed}_k{a.k}.json').write_text(json.dumps(dict(seed=a.seed,k=a.k,base_raw=base,rows=rows,wall=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2)+'\n')
if __name__=='__main__':main()
