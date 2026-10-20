"""Build the final joint core directly from sources; eliminate large intermediary core."""
import flopscope.numpy as f
from compress import source_core

def direct(sources,rank,p,w2,e1,e2,C):
 n=len(p);weight=f.sqrt(e1*e1+e2*e2*f.sum(C*C,axis=1)+p*p+w2*w2+1e-20);H=f.zeros_like(C)
 for s in sources:
  A,P,Z,L,ss,e,ww,zf,r1,r2,c1,c2,r=s
  M=P*ss+3*A*e+Z@L.T;X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T;F=A*c1+P*c2;y=Z[:,r+1]
  for a,b,c in ((X,P,Y),(M,P,P)):
   na,nb,nc=[f.sum(z*z,axis=0) for z in (a,b,c)]
   H=H+(a*(nb*nc)[None,:])@a.T+(b*(na*nc)[None,:])@b.T+(c*(na*nb)[None,:])@c.T
  S=F@P.T;S=(S+S.T)/2;H=H+(S@S.T)*f.sum(y*y)+y[:,None]*y[None,:]*f.sum(S*S)
 _,V=f.linalg.eigh(H*weight[:,None]*weight[None,:]);B,_=f.linalg.qr(V[:,-rank:]/weight[:,None],mode='reduced')
 G=f.zeros((rank,rank,rank),dtype=f.float32)
 for s in sources:G=G+source_core(tuple(B.T@x if i in (0,1,2,7) else x for i,x in enumerate(s)))
 return B,G
