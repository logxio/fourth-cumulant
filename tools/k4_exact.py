"""Metered NumPy exact K=4 chain with two-leg sources and cumulative Q."""

from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab/cand_k4x"))
import reference_port as reference
from exact_k4_group import Group, slice_birth, slices, two_leg_birth

WEIGHT_HASH = {
    3: "eeaa2243c7c57d0155692f4e136fa7caa51d9807f37a5dc840f46bde1286b5e3",
    4: "6c01182958971fb6c63da14c64da20048e5f237b6c008e237c73bafaffad0fa3",
    5: "a920e669d7d3679d8cb3d533dabbd2a30ca45195a90fcbc43b7d56a24e625855",
}
TRUTH_HASH = {
    3: "033bda54ff4c8e7cbaa0e6cc95131d2456825b425735f99fe696307041378081",
    4: "3e66c6a6adb1ea5709a206aeb3822d2b7e08f915da4870e7f2b496a1f36bc23d",
    5: "135b93738cc4a1fba92c82286e208ed57300639eae02b14766f8db7840719112",
}


class ExactBank:
    def __init__(self, width, dtype=fnp.float64, **_):
        self.width = width
        self.dtype = dtype
        self.sources: list[Group] = []
        self.shared = SimpleNamespace(members=[])
        self.eval_layer = 0

    def transport(self, operator):
        for group in self.sources:
            group.linear(operator)

    def wick_scale(self, slope):
        for group in self.sources:
            group.wick(slope)

    def join_old(self, *_):
        # Exact sources retain their birth kernel and cumulative transport.
        pass

    def slices(self, cp_limit=None):
        assert cp_limit is None
        n = self.width
        a = fnp.zeros(n, dtype=self.dtype)
        b = fnp.zeros((n, n), dtype=self.dtype)
        c = fnp.zeros((n, n), dtype=self.dtype)
        d = fnp.zeros((n, n, n), dtype=self.dtype)
        for group in self.sources:
            read = slices(group)
            a, b, c, d = (a + read[4,], b + read[3, 1],
                          c + read[2, 2], d + read[2, 1, 1])
        return a, b, c, d

    def birth_from_slices(self, _layer, d4, d31, d22, d211):
        n = self.width
        mask = 1 - np.eye(n, dtype=d211.dtype)
        d211 = d211 * mask[:, :, None] * mask[:, None, :] * mask[None, :, :]
        self.sources.append(slice_birth(d4, d31, d22, d211))


def add_two_leg(bank, _layer, covariance, k3, wick, *_args, **_kwargs):
    n = bank.width
    mask = 1 - np.eye(n, dtype=covariance.dtype)
    cube = (k3.tensor * mask[:, :, None] * mask[:, None, :] * mask[None, :, :]
            if k3.tensor is not None else fnp.zeros((n, n, n), dtype=bank.dtype))
    bank.sources.append(two_leg_birth(covariance, cube, wick(1, 1),
                                      wick(2, 1), wick(3, 1)))


class DenseK3:
    def __init__(self, tensor=None):
        self.tensor = tensor
        self.terms = [] if tensor is None else [True]

    def transport(self, weight):
        if self.tensor is None:
            return DenseK3()
        n = weight.shape[0]
        a = fnp.matmul(weight, self.tensor.reshape(n, n * n)).reshape(n, n, n)
        b = fnp.matmul(weight, a.transpose(1, 0, 2).reshape(n, n * n))
        b = b.reshape(n, n, n).transpose(1, 0, 2)
        c = fnp.matmul(weight, b.transpose(2, 0, 1).reshape(n, n * n))
        return DenseK3(c.reshape(n, n, n).transpose(1, 2, 0))

    def repeated(self):
        n = self.tensor.shape[0]
        idx = np.arange(n)
        return self.tensor[idx, idx, idx], self.tensor[idx[:, None], idx[:, None], idx[None, :]]


def dense_slice_lookup(covariance, k3, bank_slices, has_k4=True):
    n = covariance.shape[0]
    d4, d31, d22, d211 = bank_slices
    c_off = reference._off(covariance)
    if k3.tensor is None:
        d3 = fnp.zeros(n, dtype=covariance.dtype)
        d21 = fnp.zeros((n, n), dtype=covariance.dtype)
    else:
        d3, d21 = k3.repeated()
        d21 = reference._off(d21)
    cache = {}

    def get(vec):
        if vec in cache:
            return cache[vec]
        degree = sum(vec)
        active = [i for i, v in enumerate(vec) if v]
        part = tuple(vec[i] for i in active)
        if degree == 2:
            value = fnp.diag(covariance) if len(part) == 1 else c_off
        elif degree == 3:
            if k3.tensor is None:
                value = None
            elif len(part) == 1:
                value = d3
            elif len(part) == 2:
                value = d21 if part == (2, 1) else d21.T
            else:
                value = k3.tensor
        elif degree == 4:
            if not has_k4:
                value = None
            elif len(part) == 1:
                value = d4
            elif len(part) == 2:
                value = d31 if part == (3, 1) else (d31.T if part == (1, 3) else d22)
            elif len(part) == 3:
                repeated = part.index(2)
                singles = [i for i in range(3) if i != repeated]
                order = tuple(0 if i == repeated else 1 + singles.index(i) for i in range(3))
                value = d211.transpose(order)
            else:
                raise ValueError(part)
        else:
            raise ValueError(degree)
        cache[vec] = value
        return value
    return get


def dense_power(part, terms, get_slice, wick, width, config, rank_cap):
    assert len(part) == 3
    n = width
    total = fnp.zeros((n, n, n), dtype=wick(0, part[0]).dtype)
    for vec_part, coefficient in terms:
        degrees = [sum(v[axis] for v in vec_part) for axis in range(3)]
        term = (coefficient * wick(degrees[0], part[0])[:, None, None]
                * wick(degrees[1], part[1])[None, :, None]
                * wick(degrees[2], part[2])[None, None, :])
        for vec in vec_part:
            active = [i for i, power in enumerate(vec) if power]
            value = get_slice(vec)
            if value is None:
                term = None
                break
            shape = [1, 1, 1]
            for axis in active:
                shape[axis] = n
            term = term * value.reshape(shape)
        if term is not None:
            total = total + term
    if part == (1, 1, 1):
        return sum(total.transpose(p) for p in (
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0))) / 6
    return (total + total.transpose(0, 2, 1)) / 2


def dense_k3_from_slices(p111, k3_diag, k3_21, width, dtype):
    n = width
    idx = np.arange(n)
    current_diag = p111[idx, idx, idx]
    current_21 = p111[idx[:, None], idx[:, None], idx[None, :]]
    delta_21 = reference._off(k3_21 - current_21)
    delta_diag = k3_diag - current_diag
    eye = fnp.eye(n, dtype=dtype)
    leg = eye * delta_diag[:, None] + 3 * delta_21.T
    correction = (leg[:, :, None] * eye[None, :, :]
                  + leg.T[:, :, None] * eye[:, None, :]
                  + leg.T[:, None, :] * eye[:, :, None]) / 3
    return DenseK3(p111 + correction)


def dense_k211_from_power(p211, p111, p1, p11):
    return (p211 - 2 * p1[:, None, None] * p111
            - 2 * p11[:, :, None] * p11[:, None, :])


def install_numpy_exact_step():
    """Keep the K4 (2,1,1) contribution dense until the slice birth.

    Compute three-axis diagram terms directly as dense NumPy arrays. The
    generic CP3 route expands a covariance triangle into n^2 columns and
    then reconstructs the same cube column by column.
    """
    source = inspect.getsource(reference.nonlinear_step)
    old = source[source.index("    if feedback_active:\n        k211 = _k211_from_power"):
                 source.index("    trace(\"k211\" if feedback_active")]
    new = ("    if feedback_active:\n"
           "        k211 = _k211_from_power(p211, p111, p1, p11)\n"
           "        old_211 = (old_211_pre * slope2[:, None, None]\n"
           "                   * slope[None, :, None] * slope[None, None, :])\n"
           "        k211 = k211 - old_211 - new_211\n"
           "    else:\n"
           "        k211 = fnp.zeros((n, n, n), dtype=C.dtype)\n")
    replacements = ((old, new),)
    for old, new in replacements:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    exec(compile(source, "exact_numpy_nonlinear", "exec"), reference.__dict__)


def run_exact(weights, *, dtype="f32"):
    """Return all layer means and the flopscope charge in 2**31 FLOP units."""
    reference.K4Bank = ExactBank
    reference._add_two_leg_birth = add_two_leg
    reference.SymmetricK3 = DenseK3
    reference._slice_lookup = dense_slice_lookup
    reference._three_axis_power = dense_power
    reference._k3_from_slices = dense_k3_from_slices
    reference._k211_from_power = dense_k211_from_power
    install_numpy_exact_step()
    config = reference.ChainConfig(
        triangle="exact", mixed_cycle="exact", m0=999,
        k3_rank_cap=None, k211_rank_cap=None, feedback_rank_cap=None,
        slice_feedback_rank_cap=None, k3_feedback_rank_cap=None,
        cp_source_limit=None, cp_compress_once=False,
        k211_feedback_layers=None, enable_three2_birth=True,
    )
    dtype = fnp.float32 if dtype == "f32" else fnp.float64
    with flops.BudgetContext(flop_budget=10**18, quiet=True) as budget:
        means = reference.run_chain(list(weights), dtype=dtype, config=config)
    return np.asarray(means, dtype=np.float64), budget.flops_used / 2**31
