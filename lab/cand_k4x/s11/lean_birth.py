"""Single-direction non-tree birth from the nearest B sources.

Contract symmetric source factors directly; keep a single n-by-n output.
Every numerical operation belongs to flopscope.
"""
import flopscope.numpy as f
from b_pair import diag,off


def xyz(x,y,z,v,w,C):
    """H=sum_r v_r T_rab, J=sum_r w_r C_ra T_rab, D=T_rrb."""
    h=(y*(v@x)[None,:])@z.T
    h=h+(x*(v@y)[None,:])@z.T+(x*(v@z)[None,:])@y.T
    h=(h+h.T)/6
    cx=C.T@(w[:,None]*x);cy=C.T@(w[:,None]*y);cz=C.T@(w[:,None]*z)
    j=((cx*y)@z.T+(cx*z)@y.T+(cy*x)@z.T+(cy*z)@x.T+(cz*x)@y.T+(cz*y)@x.T)/6
    d=((x*y)@z.T+(x*z)@y.T+(y*z)@x.T)/3
    return h,j,d


def mpp(m,p,v,w,C):
    a=(m*(v@p)[None,:])@p.T
    h=((p*(v@m)[None,:])@p.T+a+a.T)/3
    cm=C.T@(w[:,None]*m);cp=C.T@(w[:,None]*p)
    j=((cm*p)@p.T+(cp*m)@p.T+(cp*p)@m.T)/3
    d=((p*p)@m.T+2*((m*p)@p.T))/3
    return h,j,d


def fpy(x,p,y,v,w,C):
    s=x@p.T;ss=s+s.T
    vx=v@x;vp=v@p
    t=p@vx+x@vp
    h=((v@y)*ss+t[:,None]*y[None,:]+y[:,None]*t[None,:])/6
    cx=C.T@(w[:,None]*x);cp=C.T@(w[:,None]*p)
    cy=C.T@(w*y)
    q=f.sum(cx*p+cp*x,axis=1)
    j=(q[:,None]*y[None,:]+y[:,None]*((cx@p.T)+(cp@x.T))+cy[:,None]*ss)/6
    d=(diag(s)[:,None]*y[None,:]+y[:,None]*ss)/3
    return h,j,d


def source_parts(source,v,w,C):
    A,P,Z,L,s,e,w2,zf,r1,r2,c1,c2,r=source
    M=P*s+3*A*e+Z@L.T
    X=3*A+zf[:,:r1.shape[1]]@r1.T
    Y=A*w2+zf[:,r1.shape[1]:]@r2.T
    F=A*c1+P*c2;y=Z[:,r+1]
    h,j,d=xyz(X,P,Y,v,w,C)
    hh,jj,dd=mpp(M,P,v,w,C);h=h+hh;j=j+jj;d=d+dd
    hh,jj,dd=fpy(F,P,y,v,w,C)
    return h+hh,j+jj,d+dd


def repeated_sub(value,u,d01,d12,dd):
    return value-u[:,None]*d01-d01.T*u[None,:]-f.diagflat(u@d12)+2*f.diagflat(u*dd)


def project(sources,dmats,C,p,w2,e1,e2,direction):
    # B's existing mean direction; normalization is included in billing.
    u=direction/f.sqrt(f.sum(direction*direction)+1e-30)
    v=u*e1;w=u*e2
    total=f.zeros_like(C)
    for src in sources:
        h,j,d=source_parts(src,v,w,C)
        value=p[:,None]*h*p[None,:]+(w2[:,None]*h*w2[None,:])*C
        edge=w2[:,None]*j*p[None,:];value=value+edge+edge.T
        d01=d*((e1*p)[:,None]*p[None,:]+(e1*w2+e2*p)[:,None]*w2[None,:]*C)
        d12=d.T*(e1[:,None]*(p*p)[None,:]+2*e2[:,None]*(p*w2)[None,:]*C)
        total=total+repeated_sub(value,u,d01,d12,diag(d)*e1*p*p)
    total=total+((C.T*w[None,:])@C)*C*w2[:,None]*w2[None,:]
    alpha=e2*.5-p*p
    for vv,m in dmats:
        value=f.sum(u*alpha*vv)*(p[:,None]*m*p[None,:])
        h=m.T@(u*e1*.5)
        value=value+(vv*w2)[:,None]*(p*h)[None,:]+(p*h)[:,None]*(vv*w2)[None,:]
        md=diag(m)
        d01=(vv*(alpha*p+.5*e1*w2))[:,None]*m*p[None,:]+(.5*e1*p*md)[:,None]*(w2*vv)[None,:]
        d12=(alpha*vv)[:,None]*(p*p*md)[None,:]+e1[:,None]*m*(p*w2*vv)[None,:]
        dd=vv*md*(alpha*p*p+e1*w2*p)
        total=total+repeated_sub(value,u,d01,d12,dd)
    return u,3*(total+total.T)


def recent_sources(v,count):
    sources=[]
    for k in range(max(0,v['A_st'].shape[0]-count),v['A_st'].shape[0]):
        sources.append(tuple(v[name][k] for name in ('A_st','P_st','Z_st','L_st','s_list','e_list','w2b_list','Zf_st','R1T_st','R2T_st','c1_list','c2_list'))+(v['r'],))
    return sources
