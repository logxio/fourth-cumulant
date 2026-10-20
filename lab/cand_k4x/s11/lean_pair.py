"""Recent-source, mean-direction matrix pairs attached to the real B chain."""
import flopscope.numpy as f
from b_pair import BPair,diag,off,feedback_slices
from lean_birth import project,recent_sources

class LeanPair(BPair):
    def __init__(self,count=1):
        self.k=2;self.age=1;self.count=count;self.pairs=[];self.fb=[];self.current=[]
        self.pending=[];self.birth=False;self.final4=None;self.final3=None

    def transport(self,v):
        W=v['W'];li=v['li'];self.current=[];self.final4=None;self.final3=None
        for born,u,b in self.pairs:
            if li-born!=1:continue
            if v['last']:
                da=f.sum(W*W*u[None,:],axis=1);db=f.sum((W@b)*W,axis=1)
                self.final4=da*db
            else:self.current.append((born,(W*u[None,:])@W.T,(W@b)@W.T))
        if v['last']:
            for x,m in self.fb:
                d=(W@x)*f.sum((W@m)*W,axis=1)
                self.final3=d if self.final3 is None else self.final3+d
        else:self.fb=[(W@x,(W@m)@W.T) for x,m in self.fb]

    def k3(self,d3,d21):
        if self.final3 is not None:return d3+self.final3,d21
        return self.add_k3((d3,d21))

    def pre(self,v):
        orig=tuple(v.get(key) for key in ('g4row','wk4m','wk431','wk431r'))
        self.pending=[];self.birth=13<=v['li']<15
        if v['last']:
            return (orig[0]+self.final4,)+orig[1:] if self.final4 is not None else orig
        q4=None;q31=None;q22=None
        for born,a,b in self.current:
            da,db=diag(a),diag(b)
            q4=da*db;q31=off((da[:,None]*b+db[:,None]*a)/2)
            q22=off((da[:,None]*db[None,:]+db[:,None]*da[None,:])/6)
            self.pending.extend([(da,b),(db,a)])
        if self.birth:
            self.sources=[tuple(f.copy(x) if hasattr(x,'shape') else x for x in src) for src in recent_sources(v,self.count)]
            self.C=f.copy(v['C_off']);self.direction=f.copy(v['mu'])
            lam=v['lam_prev']*v['SSC_K']['lam_feed'][v['li']]
            self.dmats=[(f.ones(v['n'],dtype=f.float32),self.C*lam/3)]+[(x/6,m) for x,m in self.pending]
        if q4 is None:return orig
        a,b,c,d=orig;return a+q4,b+q22,c+q31.T,d+q31

    def feed(self,v):
        # Early layers add no arithmetic to B. Hooks subtract only actual feed.
        if not self.pending:return None,None
        return super().feed(v)

    def post(self,v):
        self.pairs=[]
        if self.birth:
            wa=v['W_all'];p=wa[:,4];w2=wa[:,6]
            e1=wa[:,5]-2*v['pk1v']*p;e2=wa[:,7]-2*v['pk1v']*w2
            u,b=project(self.sources,self.dmats,self.C,p,w2,e1,e2,self.direction)
            self.pairs=[(v['li'],u,b)]
            self.sources=[];self.dmats=[];self.C=None;self.direction=None
        return v['g_prev']
