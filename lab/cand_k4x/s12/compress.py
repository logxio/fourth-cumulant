"""Symmetric Tucker core and weighted joint neuron space; all algorithm arithmetic billable."""
import flopscope.numpy as f

def sym3(x,y,z):
 r,m=x.shape;t=((x[:,None,:]*y[None,:,:]).reshape(r*r,m)@z.T).reshape(r,r,r)
 return (t+t.transpose(0,2,1)+t.transpose(1,0,2)+t.transpose(1,2,0)+t.transpose(2,0,1)+t.transpose(2,1,0))/6

def source_core(s):
 A,P,Z,L,ss,e,ww,zf,r1,r2,c1,c2,r=s
 M=P*ss+3*A*e+Z@L.T;X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T
 G=A*c1+P*c2;y=Z[:,r+1];a=G@P.T;a=a+a.T
 return sym3(X,P,Y)+sym3(M,P,P)+(a[:,:,None]*y[None,None,:]+a[:,None,:]*y[None,:,None]+a[None,:,:]*y[:,None,None])/6

def rotate(G,V):
 r=G.shape[0];s=V.shape[1]
 t=(V.T@G.reshape(r,r*r)).reshape(s,r,r)
 t=(t.transpose(0,2,1).reshape(s*r,r)@V).reshape(s,r,s).transpose(0,2,1)
 return (t.reshape(s*s,r)@V).reshape(s,s,s)

def old_tucker(Q,G,rank):
 r=G.shape[0];flat=G.reshape(r,r*r);_,V=f.linalg.eigh(flat@flat.T);V=V[:,-rank:]
 return Q@V,rotate(G,V)

def joint(Q,G,sources,rank,p,w2,e1,e2,C):
 # Weighted factor covariance: each symmetric trilinear leg weighted by norms of its partners.
 # Neuron weights reflect the four first-read kernel coefficients and actual covariance row energy.
 weight=f.sqrt(e1*e1+e2*e2*f.sum(C*C,axis=1)+p*p+w2*w2+1e-20)
 flat=G.reshape(G.shape[0],-1);H=Q@(flat@flat.T)@Q.T
 for s in sources:
  A,P,Z,L,ss,e,ww,zf,r1,r2,c1,c2,r=s
  M=P*ss+3*A*e+Z@L.T;X=3*A+zf[:,:r1.shape[1]]@r1.T;Y=A*ww+zf[:,r1.shape[1]:]@r2.T;F=A*c1+P*c2;y=Z[:,r+1]
  for a,b,c in ((X,P,Y),(M,P,P)):
   na,nb,nc=[f.sum(z*z,axis=0) for z in (a,b,c)]
   H=H+(a*(nb*nc)[None,:])@a.T+(b*(na*nc)[None,:])@b.T+(c*(na*nb)[None,:])@c.T
  S=F@P.T;S=(S+S.T)/2
  H=H+(S@S.T)*f.sum(y*y)+y[:,None]*y[None,:]*f.sum(S*S)
 # Weighted PCA, mapped back then orthonormalized in physical neuron coordinates.
 _,V=f.linalg.eigh(H*weight[:,None]*weight[None,:]);B,_=f.linalg.qr(V[:,-rank:]/weight[:,None],mode='reduced')
 out=rotate(G,Q.T@B)
 for s in sources:
  coords=tuple(B.T@x if i in (0,1,2,7) else x for i,x in enumerate(s));out=out+source_core(coords)
 return B,out
