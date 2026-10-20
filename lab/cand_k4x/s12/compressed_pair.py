"""All-source compressed non-tree matrix pair on the frozen B hooks."""
from upper_eval import FullPair,f,time
from sketch_compress import sketch
from shared_core import CoreBirth
from power_birth import PowerBirth
class CompressedPair(FullPair):
 def __init__(self,k,rank=4):super().__init__(k);self.rank=rank
 def post(self,v):
  self.pairs=[]
  if self.birth:
   start=time.monotonic();before=self.ctx.flops_used
   wa=v['W_all'];p=wa[:,4];w2=wa[:,6];e1=wa[:,5]-2*v['pk1v']*p;e2=wa[:,7]-2*v['pk1v']*w2
   Q,G=sketch(self.sources,self.rank,p,w2,e1,e2,self.C)
   op=CoreBirth(Q,G,self.C,p,w2,e1,e2);other=PowerBirth([],self.dmats,self.C,p,w2,e1,e2)
   u=self.direction/f.sqrt(f.sum(self.direction*self.direction)+1e-30)
   b=op.left(u)+other.left(u);u=op.right(b)+other.right(b);u=u/f.sqrt(f.sum(u*u)+1e-30);b=op.left(u)+other.left(u)
   self.pairs=[(v['li'],u,b)]
   self.birth_rows.append(dict(layer=v['li'],sources=len(self.sources),rank=self.rank,U=(self.ctx.flops_used-before)/2**31,wall=time.monotonic()-start))
   self.sources=[];self.dmats=[];self.C=None;self.direction=None
  return v['g_prev']
