"""Metered NumPy evaluation of the factored augmented K=3 diagrams."""

from __future__ import annotations

import math

import numpy as rawnp
import flopscope as flops
import flopscope.numpy as np

import k3_aug_tables as tables


TABLES = tables.HIGH
LOW = tables.LOW


class WickCache:
    def __init__(self, mean, variance):
        self.mean = mean
        self.sigma = np.sqrt(np.maximum(variance, 1e-10))
        self.alpha = mean / self.sigma
        self.phi = np.exp(-0.5 * self.alpha * self.alpha) / math.sqrt(2 * math.pi)
        self.Phi = flops.stats.norm.cdf(self.alpha).astype(mean.dtype)
        self.cache = {}

    def __call__(self, k, p=1):
        key = k, p
        if key in self.cache:
            return self.cache[key]
        if k < p:
            a, b = tables.WICK_POLYS[(p, k)]
            value = self.sigma ** (p - k) * (
                self.horner(a) * self.phi + self.horner(b) * self.Phi)
        elif p > 1:
            value = math.factorial(p) * self(k - p + 1, 1)
        elif k == 0:
            value = self.sigma * self.phi + self.mean * self.Phi
        elif k == 1:
            value = self.Phi
        else:
            value = ((-1) ** (k - 2)) * self.sigma ** (-(k - 1)) * (
                self.horner(tables.HE_COEFS[k - 2])) * self.phi
        self.cache[key] = value
        return value

    def horner(self, coefficients):
        value = np.zeros_like(self.alpha)
        for c in reversed(coefficients):
            value = value * self.alpha + c
        return value


def transport3(w, tensor):
    n = len(w)
    a = (w @ tensor.reshape(n, -1)).reshape(n, n, n)
    b = (w @ a.transpose(1, 0, 2).reshape(n, -1)).reshape(n, n, n).transpose(1, 0, 2)
    return (w @ b.transpose(2, 0, 1).reshape(n, -1)).reshape(n, n, n).transpose(1, 2, 0)


def slice_for(vec, arrays):
    active = [i for i, v in enumerate(vec) if v]
    part = tuple(vec[i] for i in active)
    degree = sum(vec)
    if degree == 2:
        result = arrays["var"] if len(active) == 1 else arrays["cov"]
    elif degree == 3:
        if arrays["k3"] is None:
            return None
        if len(active) == 1:
            result = arrays["d3"]
        elif len(active) == 2:
            result = arrays["d21"] if part == (2, 1) else arrays["d21"].T
        else:
            result = arrays["k3"]
    elif degree == 4:
        if arrays["g"] is None:
            return None
        if len(active) == 1:
            result = arrays["d4"]
        elif len(active) == 2:
            result = (arrays["d31"] if part == (3, 1) else
                      arrays["d31"].T if part == (1, 3) else arrays["d22"])
        elif len(active) == 3:
            repeated = part.index(2)
            active = [i for i in active if i != active[repeated]]
            result = arrays["d211"]
        else:
            return None
    else:
        raise ValueError(vec)
    shape = [1] * len(vec)
    for axis in active:
        shape[axis] = arrays["n"]
    return result.reshape(shape)


def evaluate(records, arrays, wick):
    n = arrays["n"]
    result = np.zeros((n,) * len(records[0][0]), dtype=arrays["var"].dtype)
    for part, factors, coef, wick_kp in records:
        term = coef
        for axis, (k, p) in enumerate(wick_kp):
            shape = [1] * len(part)
            shape[axis] = n
            term = term * np.asarray(wick(k, p)).reshape(shape)
        for degree, vec in factors:
            value = slice_for(vec, arrays)
            if value is None:
                term = None
                break
            term = term * value
        if term is not None:
            result = result + term
    return result


def k3_from_slices(p111, diag, d21):
    n = len(diag)
    idx = rawnp.arange(n)
    current_diag = p111[idx, idx, idx]
    current21 = p111[idx[:, None], idx[:, None], idx[None, :]]
    eye = np.eye(n, dtype=p111.dtype)
    delta21 = (d21 - current21) * (1 - eye)
    leg = eye * (diag - current_diag)[:, None] + 3 * delta21.T
    correction = (leg[:, :, None] * eye[None, :, :]
                  + leg.T[:, :, None] * eye[:, None, :]
                  + leg.T[:, None, :] * eye[:, :, None]) / 3
    return p111 + correction


def predict(weights, *, dtype=np.float64):
    n = 128
    assert weights.shape[1:] == (n, n)
    coefs = tables.STATIC_128
    mu = np.zeros(n, dtype=dtype)
    cov = np.eye(n, dtype=dtype)
    k3 = None
    g = None
    rows = []
    mask = 1 - np.eye(n, dtype=dtype)
    mask3 = mask[:, :, None] * mask[:, None, :] * mask[None, :, :]
    for raw in weights:
        w = raw.astype(dtype).T
        mu = w @ mu
        cov = w @ cov @ w.T
        cov = (cov + cov.T) / 2
        if k3 is not None:
            k3 = transport3(w, k3)
        if g is not None:
            g = w @ g @ w.T
            g = (g + g.T) / 2
        var = np.maximum(np.diag(cov), 1e-10)
        cov_off = cov * mask
        if k3 is None:
            d3 = d21 = None
            distinct = None
        else:
            idx = rawnp.arange(n)
            d3 = k3[idx, idx, idx]
            d21 = k3[idx[:, None], idx[:, None], idx[None, :]] * mask
            distinct = k3 * mask3
        if g is None:
            d4 = d31 = d22 = d211 = None
        else:
            dg = np.diag(g)
            d4 = 2 * dg
            d22 = (2 / 6) * (dg[:, None] + dg[None, :]) * mask
            d31 = g * mask
            d211 = (2 / 6) * g
        arrays = dict(n=n, var=var, cov=cov_off, k3=distinct, d3=d3,
                      d21=d21, g=g, d4=d4, d31=d31, d22=d22, d211=d211)
        wick = WickCache(mu, var)
        pk = {}
        for int_part, factors, coef, wick_kp in LOW:
            degree = len(int_part)
            term = coef
            for ax, (k, p) in enumerate(wick_kp):
                shape = [1] * degree
                shape[ax] = n
                term = term * np.asarray(wick(k, p)).reshape(shape)
            for _, vec in factors:
                value = slice_for(vec, arrays)
                if value is None:
                    term = None
                    break
                term = term * value
            if term is not None:
                pk[int_part] = pk.get(int_part, 0) + term
        for key in ((1, 1), (2, 2)):
            pk[key] = (pk[key] + pk[key].T) / 2
        for key in ((1, 1), (2, 1), (2, 2), (3, 1)):
            pk[key] = pk[key] * mask
        p111 = evaluate(TABLES[(1, 1, 1)], arrays, wick)
        p111 = sum(p111.transpose(perm) for perm in (
            (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0),
            (2, 0, 1), (2, 1, 0))) / 6
        p211 = evaluate(TABLES[(2, 1, 1)], arrays, wick)
        p211 = (p211 + p211.transpose(0, 2, 1)) / 2
        def placed(vec):
            active = [i for i, x in enumerate(vec) if x]
            part = tuple(vec[i] for i in active)
            base = pk[tuple(sorted(part, reverse=True))]
            if part == (1, 2):
                base = base.T
            shape = [1] * len(vec)
            for i in active:
                shape[i] = n
            return base.reshape(shape)
        cumul = {}
        for part, table in tables.PK2K_TABLE_AUG.items():
            val = np.zeros((n,) * len(part), dtype=dtype)
            for blocks, coef in table:
                product = coef
                for block in blocks:
                    product = product * placed(block)
                val = val + product
            cumul[part] = val
        for key in ((1, 1), (2, 1), (2, 2), (3, 1)):
            cumul[key] = cumul[key] * mask
        k211 = (p211 - 2 * pk[(1,)][:, None, None] * p111
                - 2 * pk[(1, 1)][:, :, None] * pk[(1, 1)][:, None, :])
        k211cap = (k211 * mask3).sum(axis=0)
        mu = cumul[(1,)]
        cov = np.diag(cumul[(2,)]) + (cumul[(1, 1)] + cumul[(1, 1)].T) / 2
        k3 = k3_from_slices(p111, cumul[(3,)], cumul[(2, 1)])
        k4v, k22, k31 = cumul[(4,)], cumul[(2, 2)], cumul[(3, 1)]
        g = (coefs["cA"] * (np.diag(k4v) + np.diag(k22.sum(axis=1)) + k211cap)
             + coefs["cS"] * (k31 + k31.T) / 2
             + coefs["cI"] * (k4v.sum() + k22.sum()) * np.eye(n, dtype=dtype))
        rows.append(mu.copy())
    return np.stack(rows)
