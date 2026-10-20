"""Factorized B-state non-tree birth and matrix-pair feedback.

All numerical work in this module uses flopscope. CP graph contractions
cost O(ell*n^2*R), with R proportional to depth*n, not O(n^4).
"""
import itertools
import flopscope.numpy as f


def diag(a): return f.einsum('ii->i',a)
def off(a):
    a=f.copy(a);f.fill_diagonal(a,0.);return a


def vm_cp(v,m,axis):
    n=len(v);eye=f.eye(n,dtype=f.float32)
    # v_r M_ab = sum_j v_r I_aj M_bj. Other axes are permutations.
    parts=[v[:,None]*f.ones((1,n),dtype=f.float32),eye,m.T]
    if axis==0:return tuple(parts)
    if axis==1:return parts[1],parts[0],parts[2]
    return parts[1],parts[2],parts[0]


class BirthOperator:
    def __init__(self,cp,dmats,C,w1,w2,e1,e2):
        self.n=len(w1);self.C=C;self.tri=(e2,w2)
        self.terms=[]
        X,Y,Z=cp
        self.add(X,Y,Z,e1,w1,w1,None)
        self.add(X,Y,Z,e1,w2,w2,12)
        self.add(X,Y,Z,e2,w1,w2,2)
        self.add(X,Y,Z,e2,w2,w1,1)
        # K4 core and the subtraction of its already transported Wick block.
        for v,m in dmats:
            X,Y,Z=vm_cp(v,m,0)
            self.add(X,Y,Z,e2*.5-w1*w1,w1,w1,None)
            X,Y,Z=vm_cp(v,m,1)
            self.add(X,Y,Z,e1*.5,w2,w1,None)
            X,Y,Z=vm_cp(v,m,2)
            self.add(X,Y,Z,e1*.5,w1,w2,None)
        # T_rrb, T_rar and T_raa; subtract these intersections exactly.
        n=self.n;self.d01=f.zeros((n,n),dtype=f.float32);self.d02=f.copy(self.d01);self.d12=f.copy(self.d01)
        for x,y,z,edge in self.terms:
            a=(x*y)@z.T if edge!=1 else None
            b=(x*z)@y.T if edge!=2 else None
            c=x@(y*z).T if edge!=12 else None
            if a is not None:self.d01=self.d01+(a if edge is None else a*C)
            if b is not None:self.d02=self.d02+(b if edge is None else b*C)
            if c is not None:self.d12=self.d12+(c if edge is None else c*C)
        self.dd=diag(self.d01)

    def add(self,x,y,z,a,b,c,edge):
        self.terms.append((x*a[:,None],y*b[:,None],z*c[:,None],edge))

    def right(self,omega):
        # omega has shape (ell,n,n); result n x ell.
        ans=[]
        for om in omega:
            value=f.zeros(self.n,dtype=f.float32)
            for x,y,z,edge in self.terms:
                if edge==1:
                    value=value+f.sum(x*(self.C@(y*(om@z))),axis=1)
                elif edge==2:
                    value=value+f.sum(x*(self.C@(z*(om.T@y))),axis=1)
                else:
                    oo=om*self.C if edge==12 else om
                    value=value+x@f.sum(y*(oo@z),axis=0)
            e,w=self.tri
            value=value+e*f.sum((self.C@(om*self.C*w[:,None]*w[None,:]))*self.C,axis=1)
            value=value-f.sum(self.d01*om,axis=1)-f.sum(self.d02*om.T,axis=1)-self.d12@diag(om)+2*self.dd*diag(om)
            ans.append(value)
        return f.stack(ans,axis=1)

    def left(self,u):
        ans=[]
        for s in range(u.shape[1]):
            us=u[:,s];value=f.zeros((self.n,self.n),dtype=f.float32)
            for x,y,z,edge in self.terms:
                if edge==1:
                    value=value+(y*(self.C.T@(x*us[:,None])))@z.T
                elif edge==2:
                    value=value+((z*(self.C.T@(x*us[:,None])))@y.T).T
                else:
                    v=(y*(us@x)[None,:])@z.T
                    value=value+(v*self.C if edge==12 else v)
            e,w=self.tri
            value=value+((self.C.T*(us*e)[None,:])@self.C)*self.C*w[:,None]*w[None,:]
            value=value-us[:,None]*self.d01-self.d02.T*us[None,:]
            value=value-f.diagflat(us@self.d12)+2*f.diagflat(us*self.dd)
            ans.append(value)
        return f.stack(ans)

    def rank1(self,rng):
        ell=min(5,self.n)
        omega=rng.standard_normal((ell,self.n,self.n),dtype=f.float32)
        basis,_=f.linalg.qr(self.right(omega),mode='reduced')
        core=self.left(basis).reshape(ell,self.n*self.n)
        small,_,_=f.linalg.svd(core,full_matrices=False)
        u=basis@small[:,:1]
        v=(small[:,:1].T@core).reshape(self.n,self.n)
        return f.diagflat(u[:,0]),3*(v+v.T)


def b_cp(v,feedback):
    xs=[];ys=[];zs=[]
    def triple(x,y,z):
        arr=[x,y,z]
        for perm in itertools.permutations((0,1,2)):
            xs.append(arr[perm[0]]/6);ys.append(arr[perm[1]]);zs.append(arr[perm[2]])
    for k in range(v['A_st'].shape[0]):
        A,P,Z,L=(v[name][k] for name in ('A_st','P_st','Z_st','L_st'))
        M=P*v['s_list'][k]+3*A*v['e_list'][k]+Z@L.T
        X,Y=3*A,A*v['w2b_list'][k]
        if v['rfb']:
            zf=v['Zf_st'][k];rr=v['rfb']
            X=X+zf[:,:rr]@v['R1T_st'][k].T
            Y=Y+zf[:,rr:]@v['R2T_st'][k].T
        triple(X,P,Y);triple(M,P,P)
        yy=Z[:,v['r']+1,None]*f.ones((1,v['n']),dtype=f.float32)
        triple(A*v['c1_list'][k]+P*v['c2_list'][k],P,yy)
    # Sym_3(v tensor M) is exactly a CP3 source with one identity leg.
    for vector,matrix in feedback:
        triple(*vm_cp(vector,matrix,0))
    return f.concatenate(xs,axis=1),f.concatenate(ys,axis=1),f.concatenate(zs,axis=1)


def feedback_slices(vector,matrix):
    dd=diag(matrix)
    d3=vector*dd
    d21=(2*vector[:,None]*matrix+dd[:,None]*vector[None,:])/3
    return d3,off(d21)


class BPair:
    def __init__(self,k,age):
        self.k,self.age=k,age;self.pairs=[];self.fb=[];self.current=[]
        self.pending=None;self.cp=None;self.C=None;self.rows=[];self.cost={}
        self.rng=f.random.default_rng(111)

    def mark(self,name,fn,*args):
        # Querying the budget is bookkeeping; every numerical operation is billed.
        before=self.ctx.flops_used
        out=fn(*args)
        self.cost[name]=self.cost.get(name,0)+(self.ctx.flops_used-before)
        return out

    def transport(self,v):
        if not self.k:return
        W=v['W'];li=v['li'];self.current=[]
        for born,a,b in self.pairs:
            if self.age and li-born>self.age:continue
            aa=(W@a)@W.T;bb=(W@b)@W.T
            self.current.append((born,aa,bb))
        self.fb=[(W@x,(W@m)@W.T) for x,m in self.fb]

    def add_k3(self,result):
        d3,d21=result
        for x,m in self.fb:
            a,b=feedback_slices(x,m)
            d3=d3+a
            if d21 is not None:d21=d21+b
        return d3,d21

    def pre(self,v):
        orig=tuple(v.get(key) for key in ('g4row','wk4m','wk431','wk431r'))
        if not self.k:return orig
        n=v['n'];self.pending=[]
        q4=f.zeros(n,dtype=f.float32);q31=f.zeros((n,n),dtype=f.float32);q22=f.copy(q31)
        for born,a,b in self.current:
            da,db=diag(a),diag(b)
            q4=q4+da*db
            if not v['last']:
                q31=q31+(da[:,None]*b+db[:,None]*a)/2
                q22=q22+(da[:,None]*db[None,:]+db[:,None]*da[None,:])/6
                self.pending.extend([(da,b),(db,a)])
        self.q=(q4,off(q31),off(q22))
        self.birth=15-self.k<=v['li']<15
        if self.birth:
            self.cp=b_cp(v,self.fb)
            self.C=f.copy(v['C_off'])
            lam=v['lam_prev']*v['SSC_K']['lam_feed'][v['li']] if 'SSC_K' in v else v['lam_prev']
            # B's specified radial K4 feed; it already lives in the B state.
            G=self.C*lam/3
            self.dmats=[(f.ones(n,dtype=f.float32),G)]
            self.dmats += [(x/6,f.copy(m)) for x,m in self.pending]
        a,b,c,d=orig
        if a is None:return orig
        if v['last']:return a+q4,b,c,d
        return a+q4,b+self.q[2],c+self.q[1].T,d+self.q[1]

    def feed(self,v):
        n=v['n'];z=f.zeros(n,dtype=f.float32);zm=f.zeros((n,n),dtype=f.float32)
        if not self.k:return z,zm
        p,w2=v['w1'],v['w2']
        # Wick the inherited factorized K3 sources, then append new feed.
        self.fb=[(p*x,p[:,None]*m*p[None,:]) for x,m in self.fb]
        for x,m in self.pending:
            xx=w2*x/4;mm=p[:,None]*m*p[None,:]
            self.fb.append((xx,mm))
            a,b=feedback_slices(xx,mm);z=z+a;zm=zm+b
        return z,zm

    def post(self,v):
        if not self.k:return v['g_prev']
        p=v['w1'];li=v['li'];g=v['g_prev'];keep=[]
        for born,a,b in self.current:
            if self.age and li-born>=self.age:continue
            aa=p[:,None]*a*p[None,:];bb=p[:,None]*b*p[None,:]
            da,db=diag(aa),diag(bb)
            q4=da*db;q22=off((da[:,None]*db[None,:]+db[:,None]*da[None,:])/6)
            trace=q4+f.sum(q22,axis=1)
            g=g-v['st']['cA']*trace-v['st']['cI']*f.sum(trace)
            keep.append((born,aa,bb))
        if self.birth:
            wa=v['W_all'];p=wa[:,4];w2=wa[:,6]
            e1=wa[:,5]-2*v['pk1v']*p;e2=wa[:,7]-2*v['pk1v']*w2
            op=BirthOperator(self.cp,self.dmats,self.C,p,w2,e1,e2)
            aa,bb=op.rank1(self.rng)
            keep.append((li,aa,bb))
        self.pairs=keep;self.cp=None;self.C=None;self.dmats=None
        self.rows.append(dict(layer=li,pairs=len(keep),feedback_sources=len(self.fb)))
        return g
