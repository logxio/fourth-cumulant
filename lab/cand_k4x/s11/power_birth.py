"""B-factor exact left/right kernel contractions, sharing direction-free work.

F is the symmetric non-tree birth flattened on its repeated-index axis.
One power step computes u <- F(F.T u); all arithmetic uses flopscope.
"""
import itertools
import flopscope.numpy as f
from b_pair import diag

class PowerBirth:
    def __init__(self,sources,dmats,C,p,w2,e1,e2):
        self.C,self.p,self.w2,self.e1,self.e2=C,p,w2,e1,e2
        self.dm=dmats;self.parts=[];d01=f.zeros_like(C);d12=f.zeros_like(C);dd=f.zeros_like(p)
        for A,P,Z,L,s,e,ww,zf,r1,r2,c1,c2,r in sources:
            M=P*s+3*A*e+Z@L.T;X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T
            G=A*c1+P*c2;y=Z[:,r+1];S=G@P.T;ss=S+S.T
            d=((X*P)@Y.T+(X*Y)@P.T+(P*Y)@X.T)/3
            d=d+((P*P)@M.T+2*((M*P)@P.T))/3
            d=d+(diag(S)[:,None]*y[None,:]+y[:,None]*ss)/3
            self.parts.append((X,P,Y,M,G,y,S,ss))
            d01=d01+d*((e1*p)[:,None]*p[None,:]+(e1*w2+e2*p)[:,None]*w2[None,:]*C)
            d12=d12+d.T*(e1[:,None]*(p*p)[None,:]+2*e2[:,None]*(p*w2)[None,:]*C)
            dd=dd+diag(d)*e1*p*p
        alpha=e2*.5-p*p;self.alpha=alpha
        for x,m in dmats:
            md=diag(m)
            d01=d01+(x*(alpha*p+.5*e1*w2))[:,None]*m*p[None,:]+(.5*e1*p*md)[:,None]*(w2*x)[None,:]
            d12=d12+(alpha*x)[:,None]*(p*p*md)[None,:]+e1[:,None]*m*(p*w2*x)[None,:]
            dd=dd+x*md*(alpha*p*p+e1*w2*p)
        self.d01,self.d12,self.dd=d01,d12,dd

    def left(self,u):
        C,p,w2,e1,e2=self.C,self.p,self.w2,self.e1,self.e2
        v=u*e1;w=u*e2;out=f.zeros_like(C)
        for X,P,Y,M,G,y,S,ss in self.parts:
            # Symmetric XYZ: merge terms before GEMM, not after.
            h=(P*(v@X)[None,:]+X*(v@P)[None,:])@Y.T+(X*(v@Y)[None,:])@P.T
            h=(h+h.T)/6
            cx=C.T@(w[:,None]*X);cp=C.T@(w[:,None]*P);cy=C.T@(w[:,None]*Y)
            j=((cx*P+cp*X)@Y.T+(cx*Y+cy*X)@P.T+(cp*Y+cy*P)@X.T)/6
            aa=(M*(v@P)[None,:])@P.T
            h=h+((P*(v@M)[None,:])@P.T+aa+aa.T)/3
            cm=C.T@(w[:,None]*M)
            j=j+((cm*P+cp*M)@P.T+(cp*P)@M.T)/3
            t=P@(v@G)+G@(v@P)
            h=h+((v@y)*ss+t[:,None]*y[None,:]+y[:,None]*t[None,:])/6
            cg=C.T@(w[:,None]*G);yc=C.T@(w*y)
            q=f.sum(cg*P+cp*G,axis=1)
            j=j+(q[:,None]*y[None,:]+y[:,None]*(cg@P.T+cp@G.T)+yc[:,None]*ss)/6
            z=p[:,None]*h*p[None,:]+w2[:,None]*h*w2[None,:]*C
            edge=w2[:,None]*j*p[None,:];out=out+z+edge+edge.T
        out=out+((C.T*w[None,:])@C)*C*w2[:,None]*w2[None,:]
        for x,m in self.dm:
            h=m.T@(u*e1*.5)
            out=out+f.sum(u*self.alpha*x)*(p[:,None]*m*p[None,:])+(x*w2)[:,None]*(p*h)[None,:]+(p*h)[:,None]*(x*w2)[None,:]
        out=out-u[:,None]*self.d01-self.d01.T*u[None,:]-f.diagflat(u@self.d12)+2*f.diagflat(u*self.dd)
        return 3*(out+out.T)

    def right(self,omega):
        # The caller supplies a symmetric matrix; left() guarantees this.
        C,p,w2,e1,e2=self.C,self.p,self.w2,self.e1,self.e2
        Q=omega*(p[:,None]*p[None,:]+C*w2[:,None]*w2[None,:]);R=omega*w2[:,None]*p[None,:]
        out=f.zeros_like(p)
        for X,P,Y,M,G,y,S,ss in self.parts:
            qP=Q@P;qY=Q@Y
            h=(X@f.sum(P*qY,axis=0)+P@f.sum(X*qY,axis=0)+Y@f.sum(X*qP,axis=0))/3
            xyz=(X,P,Y);rz=(R@X,R@P,R@Y);j=f.zeros_like(p)
            for i,a,b in itertools.permutations(range(3)):
                j=j+f.sum(xyz[i]*(C@(xyz[a]*rz[b])),axis=1)/6
            h=h+(M@f.sum(P*qP,axis=0)+2*P@f.sum(M*qP,axis=0))/3
            rM=R@M
            j=j+(f.sum(M*(C@(P*rz[1])),axis=1)+f.sum(P*(C@(M*rz[1])),axis=1)+f.sum(P*(C@(P*rM)),axis=1))/3
            qy=Q@y
            h=h+(y*f.sum(G*qP)+G@(P.T@qy)+P@(G.T@qy))/3
            j=j+(f.sum(C*ss*(R@y)[None,:],axis=1)+f.sum(((C*y[None,:])@R)*ss,axis=1)+y*(C@f.sum(R*ss,axis=1)))/6
            out=out+e1*h+2*e2*j
        out=out+e2*f.sum((C@(omega*C*w2[:,None]*w2[None,:]))*C,axis=1)
        for x,m in self.dm:
            out=out+self.alpha*x*f.sum(omega*p[:,None]*m*p[None,:])+e1*(m@(p*(omega@(x*w2))))
        out=out-2*f.sum(self.d01*omega,axis=1)-self.d12@diag(omega)+2*self.dd*diag(omega)
        return 6*out

    def rank1(self,direction,steps=1):
        u=direction/f.sqrt(f.sum(direction*direction)+1e-30)
        for _ in range(steps):
            b=self.left(u);u=self.right(b);u=u/f.sqrt(f.sum(u*u)+1e-30)
        return u,self.left(u)
