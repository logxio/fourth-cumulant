"""Conditional Gaussian average of a fixed slice211 contraction.

B[a,b,r] is symmetric in a,b; P contains only previous, quenched layers.
The actual reference same-side convention is preserved (crossed=False).
"""
import torch


def traces(P, B, chunk=16):
    H = P.T @ P
    a = H.new_zeros(())
    b = H.new_zeros(())
    # Research sources live in pinned host memory. Transfer the existing
    # input once; strided per-r host transfers repeatedly gather n^3 data.
    B = B.to(P.device, non_blocking=True)
    # O(n^3) scalar arithmetic, O(n^2 * chunk) scratch. The dense B is
    # research input; its construction/storage is a separate cost.
    for r0 in range(0, B.shape[2], chunk):
        r1 = min(r0 + chunk, B.shape[2])
        bc = B[:, :, r0:r1]
        h = H[:, r0:r1]
        a += ((bc * H[:, :, None]).sum((0, 1)) * H.diag()[r0:r1]).sum()
        b += (bc * h[:, None, :] * h[None, :, :]).sum()
    return a, b


def expected(P, B, variance, crossed=False):
    a, b = traces(P, B)
    d4 = variance**2 * (a + 2*b)
    d22 = variance**2 * (a + (2*b if crossed else 0))/3
    return d4, d22, a, b


def add_expected(out, d4, d22, slope=None):
    n = out[(4,)].shape[0]
    s = torch.ones(n, device=out[(4,)].device, dtype=out[(4,)].dtype) if slope is None else slope
    out[(4,)].add_(d4 * s**4)
    out[(2, 2)].add_(d22 * (s*s)[:, None] * (s*s)[None, :])


def direct_samples(P, B, W, crossed=False):
    """Four representative entries from independent draws, no n^4 tensor."""
    q = W @ P
    x,y,z = q[:,0],q[:,1],q[:,2]
    def bil(u,v):
        return torch.einsum('sa,abr,sb->sr',u,B,v)
    xx,xy,yy,yz = bil(x,x),bil(x,y),bil(y,y),bil(y,z)
    d4=(x*x*xx).sum(1)
    d31=.5*(x*x*xy+x*y*xx).sum(1)
    d22=((x*x*yy+y*y*xx)/6).sum(1)
    d211=((x*x*yz+y*z*xx)/6).sum(1)
    if crossed:
        d22 += (2*x*y*xy/3).sum(1)
        d211 += (x*y*bil(x,z)+x*z*xy).sum(1)/3
    return torch.stack((d4,d31,d22,d211),1)
