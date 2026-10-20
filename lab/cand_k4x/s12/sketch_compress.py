"""Source-factor covariance range finder, avoiding n-by-n eigendecomposition and Gram matrices."""
import flopscope.numpy as f
from compress import source_core

def sketch(sources,rank,p,w2,e1,e2,C):
 n=len(p);ell=min(rank+8,n);weight=f.sqrt(e1*e1+e2*e2*f.sum(C*C,axis=1)+p*p+w2*w2+1e-20)
 rng=f.random.default_rng(514);omega=rng.standard_normal((n,ell),dtype=f.float32);terms=[]
 for s in sources:
  A,P,Z,L,ss,e,ww,zf,r1,r2,c1,c2,r=s
  M=P*ss+3*A*e+Z@L.T;X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T;F=A*c1+P*c2;y=Z[:,r+1]
  for a,b,c in ((X,P,Y),(M,P,P)):
   na,nb,nc=[f.sum(z*z,axis=0) for z in (a,b,c)]
   terms.extend(((a,nb*nc),(b,na*nc),(c,na*nb)))
  # The common thin y leg is retained through a covariance-weighted rank-one term.
  so=(F@(P.T@omega)+P@(F.T@omega))/2
  ny=f.sum(y*y);nf=f.sum(F*F,axis=0);np_=f.sum(P*P,axis=0)
  terms.extend(((F,np_*ny),(P,nf*ny),(y[:,None],f.reshape(f.sum(so*so)/ell,(1,)))))
 def apply(R):
  R=weight[:,None]*R;out=f.zeros_like(R)
  for x,s in terms:out=out+x@(s[:,None]*(x.T@R))
  return weight[:,None]*out
 V,_=f.linalg.qr(apply(omega),mode='reduced')
 small=V.T@apply(V);_,U=f.linalg.eigh((small+small.T)/2)
 B,_=f.linalg.qr((V@U[:,-rank:])/weight[:,None],mode='reduced')
 G=f.zeros((rank,rank,rank),dtype=f.float32)
 for s in sources:G=G+source_core(tuple(B.T@x if i in (0,1,2,7) else x for i,x in enumerate(s)))
 return B,G
