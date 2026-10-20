"""Exact symmetric third-order core of sources sharing a neuron-side basis.

A shared rank r basis yields an r^3 core, not r CP columns. Core assembly is
separate; this consumer bills even the optimistic case of a free existing core.
"""
import flopscope.numpy as f
from b_pair import diag
class CoreBirth:
 def __init__(self,Q,G,C,p,w2,e1,e2,block=32):
  self.Q,self.G=Q,G;self.C,self.p,self.w2,self.e1,self.e2=C,p,w2,e1,e2;self.block=block;self.r=Q.shape[1]
  D=self.pair(Q,Q)@Q.T
  self.d01=D*((e1*p)[:,None]*p[None,:]+(e1*w2+e2*p)[:,None]*w2[None,:]*C)
  self.d12=D.T*(e1[:,None]*(p*p)[None,:]+2*e2[:,None]*(p*w2)[None,:]*C)
  self.dd=diag(D)*e1*p*p
 def pair(self,A,B):
  # Contract G_{abc} A_{ia} B_{ib}, block rows to bound RSS.
  out=[];r=self.r
  for start in range(0,len(A),self.block):
   x=A[start:start+self.block];y=B[start:start+self.block]
   xy=(x[:,:,None]*y[:,None,:]).reshape(len(x),r*r)
   out.append(xy@self.G.reshape(r*r,r))
  return f.concatenate(out,axis=0)
 def left(self,u):
  Q,G,C,p,w2,e1,e2=self.Q,self.G,self.C,self.p,self.w2,self.e1,self.e2;r=self.r
  h=Q@((u*e1)@Q@G.reshape(r,r*r)).reshape(r,r)@Q.T
  j=self.pair(C.T@((u*e2)[:,None]*Q),Q)@Q.T
  value=p[:,None]*h*p[None,:]+(w2[:,None]*h*w2[None,:])*C
  edge=w2[:,None]*j*p[None,:];value=value+edge+edge.T
  value=value-u[:,None]*self.d01-self.d01.T*u[None,:]-f.diagflat(u@self.d12)+2*f.diagflat(u*self.dd)
  return 3*(value+value.T)
 def right(self,omega):
  Q,G,C,p,w2,e1,e2=self.Q,self.G,self.C,self.p,self.w2,self.e1,self.e2;r=self.r
  weighted=omega*(p[:,None]*p[None,:]+C*w2[:,None]*w2[None,:])
  h=Q@(G.reshape(r,r*r)@(Q.T@weighted@Q).reshape(r*r))
  R=omega*w2[:,None]*p[None,:]
  j=f.sum(Q*(C@self.pair(Q,R@Q)),axis=1)
  out=e1*h+2*e2*j
  out=out-2*f.sum(self.d01*omega,axis=1)-self.d12@diag(omega)+2*self.dd*diag(omega)
  return 6*out

def merged_power(old,young,direction):
 u=direction/f.sqrt(f.sum(direction*direction)+1e-30)
 b=old.left(u)+young.left(u);u=old.right(b)+young.right(b);u=u/f.sqrt(f.sum(u*u)+1e-30)
 return u,old.left(u)+young.left(u)
