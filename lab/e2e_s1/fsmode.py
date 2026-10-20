"""FlopscopeMode: run existing torch CPU code with every ATen op executed by flopscope.numpy.

    with flops.BudgetContext(flop_budget=10**15, quiet=True) as ctx, FlopscopeMode() as m:
        out = research_step(...)          # plain torch code, unchanged
    print(ctx.flops_used); print(m.format_report())

Each ATen op torch would run is executed by flopscope.numpy (fnp) instead, inside whatever
BudgetContext is active, so flopscope's own counter meters the run. Torch tensors go in and
come out: inputs are wrapped zero-copy as FlopscopeArray views of t.numpy() (what fnp.asarray
returns, 0 flops), outputs are torch.from_numpy(np.asarray(fnp_result)) sharing memory with the fnp
result, and in-place / out= ops write into the original tensor memory through fnp ufunc out=,
fnp.copyto and fnp.put. Out-of-place ufuncs, fills and casts whose torch layout is not plain C
order write straight into a torch.empty_strided buffer through out= / fnp.copyto (same bill).
Every output has exactly torch's dtype, shape and strides (meta-device inference; svd's Vh is
corrected to the CPU kernel's column-major layout).

Pricing follows the literal NumPy port that lab/cand_k4x/r4/trace_cost.py models: fused
alpha/beta ops run as their explicit numpy steps, pow stays fnp.power (16x), erf is
2*stats.norm.cdf(sqrt(2)*x)-1, He_n is the explicit probabilists' recurrence, index_put_ is
fnp.put with charged int64 flat-index arithmetic, view/_unsafe_view are billed as fnp.reshape,
metadata-only views are free. Unknown ops, overloads, dtypes or devices raise
NotImplementedError naming the op; nothing falls back to unmetered torch arithmetic.

Work that is not arithmetic and is not billed, all visible per op in report():
  * views (permute, slice, expand, ...) run natively in torch; zeros/empty allocations are native
    (fnp.zeros/fnp.empty are free as well);
  * layout copies: when a result's numpy layout still differs from torch's (flip's negative
    strides, column-major LAPACK outputs) or the result would alias an input (He_1(x) = x), the
    values are copied with plain numpy into torch.empty_strided, so later as_strided/view calls
    behave exactly as in torch. Counted as unmetered_copy_elems (373 elements in the whole n=16
    K=4 chain, all from flip of polynomial coefficient vectors).
  * index bounds validation for index_put_ (min/max of the index tensors, plain numpy).
  * normal_/uniform_ draw their seed from torch's generator, then sample with fnp.random (metered);
    the values therefore differ from torch's stream.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import flopscope as flops
import flopscope.numpy as fnp
from flopscope._budget import get_active_budget

__all__ = ["FlopscopeMode", "HANDLED_OPS"]

NP_DTYPE = {torch.float32: np.dtype(np.float32), torch.float64: np.dtype(np.float64),
            torch.int64: np.dtype(np.int64), torch.bool: np.dtype(np.bool_)}
SCALAR = (bool, int, float)
_FSA = flops.FlopscopeArray

# ---------------------------------------------------------------- chunked threads for big pointwise ops
# numpy ufunc loops release the GIL, so splitting one large elementwise op along axis 0 of its output
# into FS_THREADS chunks runs them in parallel. Each chunk is an ordinary fnp call billed for its own
# elements (pointwise and copy prices are per element), so the sum equals the bill of the single call.
# Worker threads bill the caller's BudgetContext (flopscope keeps the active budget thread-local and
# charges under the context's RLock).
import os as _os
from concurrent.futures import ThreadPoolExecutor as _TPE
import flopscope._budget as _fsb
PAR_THREADS = int(_os.environ.get("FS_THREADS", "1"))
PAR_MIN = int(_os.environ.get("FS_PAR_MIN", str(1 << 22)))
_POOL = _TPE(PAR_THREADS) if PAR_THREADS > 1 else None


def _pcall(fn, ops, out, **kw):
    """fn(*ops, out=out, **kw), split along axis 0 of out across threads when out is large."""
    if _POOL is None or out.ndim == 0 or out.size < PAR_MIN or out.shape[0] < 2:
        return fn(*ops, out=out, **kw)
    n0 = out.shape[0]
    k = min(PAR_THREADS, n0)
    bounds = [(n0 * j // k, n0 * (j + 1) // k) for j in range(k)]
    budget = get_active_budget()

    def piece(lo, hi):
        _fsb._thread_local.active_budget = budget
        try:
            sub = []
            for o in ops:
                if isinstance(o, np.ndarray) and o.ndim == out.ndim and o.shape[0] == n0:
                    sub.append(o[lo:hi])
                else:
                    sub.append(o)       # scalar, lower rank or broadcast axis 0: same for every chunk
            fn(*sub, out=out[lo:hi], **kw)
        finally:
            _fsb._thread_local.active_budget = None

    for f in [_POOL.submit(piece, lo, hi) for lo, hi in bounds]:
        f.result()
    return out


def _pcopyto(dst, src, **kw):
    """fnp.copyto(dst, src) split like _pcall (copyto takes dst first, no out=)."""
    if _POOL is None or dst.ndim == 0 or dst.size < PAR_MIN or dst.shape[0] < 2:
        return fnp.copyto(dst, src, **kw)
    n0 = dst.shape[0]
    k = min(PAR_THREADS, n0)
    bounds = [(n0 * j // k, n0 * (j + 1) // k) for j in range(k)]
    budget = get_active_budget()

    def piece(lo, hi):
        _fsb._thread_local.active_budget = budget
        try:
            s = src[lo:hi] if isinstance(src, np.ndarray) and src.ndim == dst.ndim and src.shape[0] == n0 else src
            fnp.copyto(dst[lo:hi], s, **kw)
        finally:
            _fsb._thread_local.active_budget = None

    for f in [_POOL.submit(piece, lo, hi) for lo, hi in bounds]:
        f.result()
_UNMETERED = [0]        # elements copied with plain numpy (layout/alias only) during the current op
_KEY: dict = {}
_SCHEMA: dict = {}
_H: dict = {}           # 'add.Tensor' -> handler(call)


class _Call:
    __slots__ = ("func", "key", "args", "kwargs", "a", "tensors", "inplace")

    def __init__(self, func, key, args, kwargs):
        self.func, self.key, self.args, self.kwargs = func, key, args, kwargs
        self.a = _bind(func, args, kwargs)
        self.tensors = [t for t in _walk(args)] + [t for t in _walk(tuple(kwargs.values()))]
        self.inplace = func._schema.name.endswith("_")


def _fail(c, why):
    raise NotImplementedError(f"FlopscopeMode: aten.{c.key}: {why}")


def _key(func):
    k = _KEY.get(func)
    if k is None:
        k = _KEY[func] = str(func).removeprefix("aten.")
    return k


def _bind(func, args, kwargs):
    spec = _SCHEMA.get(func)
    if spec is None:
        spec = _SCHEMA[func] = [(a.name, a.kwarg_only, a.default_value if a.has_default_value() else None)
                                for a in func._schema.arguments]
    out, pos = {}, 0
    for name, kwonly, default in spec:
        if not kwonly and pos < len(args):
            out[name] = args[pos]
            pos += 1
        else:
            out[name] = kwargs.get(name, default)
    return out


def _walk(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _walk(y)


def _validate(c):
    for t in c.tensors:
        if t.device.type != "cpu":
            _fail(c, f"device {t.device} (CPU only)")
        if t.dtype not in NP_DTYPE:
            _fail(c, f"dtype {t.dtype} (float32/float64/int64/bool only)")
        if t.layout is not torch.strided or t.is_conj() or t.is_neg():
            _fail(c, "non-strided, conj or neg-bit tensor")
    dt = c.a.get("dtype")
    if dt is not None and dt not in NP_DTYPE:
        _fail(c, f"dtype {dt} (float32/float64/int64/bool only)")
    dev = c.a.get("device")
    if dev is not None and torch.device(dev).type != "cpu":
        _fail(c, f"device {dev} (CPU only)")


# ---------------------------------------------------------------- torch <-> fnp
def _np(t):
    if t.requires_grad:
        t = t.detach()
    return t.numpy()


def A(t):
    """torch tensor -> FlopscopeArray on the same memory (free).

Same result and same 0-flop bill as fnp.asarray(t.numpy()), built the way flopscope wraps its own
outputs (ndarray.view(FlopscopeArray), the class carries no state), which skips asarray's per-call
bookkeeping (~100 us, a quarter of the shim's wall time)."""
    return _np(t).view(_FSA)


def _ops(vals):
    """Operands with torch's scalar semantics: when a dimensioned tensor takes part, 0-dim tensors
    become Python scalars (weak in NumPy 2, like 0-dim tensors in torch promotion)."""
    dimmed = any(isinstance(v, torch.Tensor) and v.dim() > 0 for v in vals)
    out = []
    for v in vals:
        if isinstance(v, torch.Tensor):
            out.append(v.item() if dimmed and v.dim() == 0 else A(v))
        elif v is None or isinstance(v, SCALAR):
            out.append(v)
        else:
            raise NotImplementedError(f"FlopscopeMode: operand of type {type(v).__name__}")
    return out


def _to_meta(x):
    if isinstance(x, torch.Tensor):
        return torch.empty_strided(tuple(x.shape), x.stride(), dtype=x.dtype, device="meta")
    if isinstance(x, (list, tuple)):
        return type(x)(_to_meta(y) for y in x)
    return x


def _meta(c):
    """torch's own output metadata (dtype, shape, strides) for this call, from the meta device."""
    kw = {k: _to_meta(v) for k, v in c.kwargs.items() if k not in ("generator", "pin_memory")}
    if "device" in c.a:
        kw["device"] = torch.device("meta")
    return c.func(*_to_meta(c.args), **kw)


def _contig(shape):
    st, acc = [], 1
    for n in reversed(shape):
        st.append(acc)
        acc *= max(n, 1)
    return tuple(reversed(st))


def _spec(m):
    return m.dtype, tuple(m.shape), m.stride()


def _layout_ok(arr, shape, stride):
    if arr.size == 0:
        return True
    isz = arr.itemsize
    return all(n <= 1 or s == t * isz for n, s, t in zip(shape, arr.strides, stride))


def _T(r, spec, inputs=()):
    """fnp result -> torch tensor with torch's dtype, shape and strides.

A dtype difference is fixed with a charged fnp astype. A layout difference or aliasing of an
input is fixed with an unmetered numpy copy (counted)."""
    if isinstance(spec, torch.Tensor):
        spec = _spec(spec)
    dtype, shape, stride = spec
    arr = np.asarray(r)
    want = NP_DTYPE[dtype]
    if arr.dtype != want:
        arr = np.asarray(fnp.asarray(arr).astype(want))
    if arr.shape != tuple(shape):
        raise RuntimeError(f"FlopscopeMode: shape {arr.shape} != torch's {tuple(shape)}")
    ok = arr.flags.writeable and _layout_ok(arr, shape, stride)
    if ok and arr.size and inputs:
        lo, hi = np.lib.array_utils.byte_bounds(arr)
        for t in inputs:
            s = t.untyped_storage()
            a = s.data_ptr()
            if lo < a + s.nbytes() and a < hi:
                ok = False
                break
    if ok:
        return torch.from_numpy(arr)
    out = torch.empty_strided(tuple(shape), tuple(stride), dtype=dtype)
    if arr.size:
        np.copyto(out.numpy(), arr)
        _UNMETERED[0] += arr.size
    return out


def _out(c, r):
    return _T(r, _meta(c), c.tensors)


def _buffer(meta):
    return torch.empty_strided(tuple(meta.shape), meta.stride(), dtype=meta.dtype)


def _np_dtype_of(o):
    if isinstance(o, bool):
        return np.dtype(np.bool_)               # NEP 50: Python bool is not a weak scalar
    if isinstance(o, (int, float)):
        return type(o)                          # weak Python scalar
    return o.dtype


def _ufunc_out(c, fn, ops):
    """Out-of-place ufunc. When numpy's own result dtype already equals torch's, write straight into
a buffer with torch's strides via out= (same bill as fn(*ops), no layout copy); else compute and _T."""
    meta = _meta(c)
    u = getattr(np, getattr(fn, "__name__", ""), None)
    if isinstance(u, np.ufunc) and u.nout == 1:
        if u.resolve_dtypes(tuple(_np_dtype_of(o) for o in ops) + (None,))[-1] == NP_DTYPE[meta.dtype]:
            buf = _buffer(meta)
            _pcall(fn, ops, A(buf))
            return buf
    return _T(fn(*ops), meta, c.tensors)


def reg(*keys):
    def deco(fn):
        for k in keys:
            assert k not in _H, k
            _H[k] = fn
        return fn
    return deco


# ---------------------------------------------------------------- free: views, metadata, allocation
@reg("permute.default", "transpose.int", "t.default", "slice.Tensor", "select.int", "expand.default",
     "unsqueeze.default", "squeeze.dim", "squeeze.dims", "squeeze.default", "diagonal.default",
     "as_strided.default", "unbind.int", "alias.default", "detach.default", "lift_fresh.default",
     "split.Tensor", "split_with_sizes.default", "_local_scalar_dense.default")
def _native_free(c):
    return c.func(*c.args, **c.kwargs)


@reg("zeros.default", "zeros_like.default", "new_zeros.default", "empty.memory_format", "empty_like.default",
     "empty_strided.default", "new_empty.default", "new_empty_strided.default")
def _native_alloc(c):
    out = c.func(*c.args, **c.kwargs)       # fnp.zeros / fnp.empty are free; allocation, no arithmetic
    if out.dtype not in NP_DTYPE:
        _fail(c, f"dtype {out.dtype}")
    return out


@reg("view.default", "_unsafe_view.default", "_reshape_alias.default")
def _view(c):
    out = c.func(*c.args, **c.kwargs)       # torch's own view (raises exactly where torch raises)
    fnp.reshape(A(c.a["self"]), tuple(out.shape))   # literal fnp.reshape bill, 1x per element
    return out


# ---------------------------------------------------------------- pointwise
def _scale(v, alpha):
    if alpha == 1:
        return v
    if isinstance(v, SCALAR):
        return v * alpha
    if _POOL is not None and v.ndim and v.size >= PAR_MIN and v.shape[0] >= 2:
        dt = np.result_type(v.dtype, np.asarray(alpha).dtype) if not isinstance(alpha, (int, float)) else \
            np.multiply.resolve_dtypes((v.dtype, type(alpha), None))[-1]
        buf = np.empty(v.shape, dtype=dt).view(_FSA)
        return _pcall(fnp.multiply, [v, alpha], buf)      # same bill as fnp.multiply(v, alpha): N
    return fnp.multiply(v, alpha)


def _check_out(c, out, *operands):
    shape = np.broadcast_shapes(*(tuple(t.shape) for t in operands if isinstance(t, torch.Tensor)))
    if tuple(out.shape) != tuple(shape):
        _fail(c, f"out= of shape {tuple(out.shape)} needs a resize to {tuple(shape)}")


def _binary(fn, alpha=False, rsub=False, names=("self", "other")):
    def h(c):
        x, y = c.a[names[0]], c.a[names[1]]
        ox, oy = _ops([x, y])
        if rsub:                                    # rsub(self, other, alpha) = other - alpha*self
            ox, oy = oy, ox
        if alpha:
            oy = _scale(oy, c.a["alpha"])
        if c.inplace:
            _pcall(fn, [ox, oy], A(x))
            return x
        out = c.a.get("out")
        if out is not None:
            _check_out(c, out, x, y)
            _pcall(fn, [ox, oy], A(out))
            return out
        return _ufunc_out(c, fn, [ox, oy])
    return h


def _overloads(name, *ovs):
    return [f"{name}.{o}" for o in ovs]


for _n, _f, _al in (("add", fnp.add, True), ("sub", fnp.subtract, True), ("mul", fnp.multiply, False)):
    reg(*_overloads(_n, "Tensor", "Scalar", "out"), *_overloads(_n + "_", "Tensor", "Scalar"))(_binary(_f, alpha=_al))
reg("rsub.Tensor", "rsub.Scalar")(_binary(fnp.subtract, alpha=True, rsub=True))
for _n, _f in (("maximum", fnp.maximum), ("minimum", fnp.minimum), ("fmax", fnp.fmax), ("fmin", fnp.fmin)):
    reg(f"{_n}.default", f"{_n}.out")(_binary(_f))
for _n, _f in (("eq", fnp.equal), ("ne", fnp.not_equal), ("lt", fnp.less), ("le", fnp.less_equal),
               ("gt", fnp.greater), ("ge", fnp.greater_equal), ("bitwise_and", fnp.bitwise_and),
               ("bitwise_or", fnp.bitwise_or), ("bitwise_xor", fnp.bitwise_xor),
               ("remainder", fnp.remainder), ("floor_divide", fnp.floor_divide)):
    reg(*_overloads(_n, "Tensor", "Scalar"), *_overloads(_n + "_", "Tensor", "Scalar"))(_binary(_f))
for _n, _f in (("logical_and", fnp.logical_and), ("logical_or", fnp.logical_or), ("logical_xor", fnp.logical_xor),
               ("atan2", fnp.arctan2)):
    reg(f"{_n}.default", f"{_n}_.default")(_binary(_f))
reg("pow.Tensor_Scalar", "pow.Tensor_Tensor", "pow.Scalar", "pow_.Scalar", "pow_.Tensor")(
    _binary(fnp.power, names=("self", "exponent")))


def _div_mode(a, b, mode, out=None):
    if mode is None:
        return fnp.divide(a, b, out=out)
    if mode == "floor":
        return fnp.floor_divide(a, b, out=out)
    if mode == "trunc":
        return fnp.trunc(fnp.divide(a, b), out=out)
    raise NotImplementedError(f"FlopscopeMode: div rounding_mode={mode!r}")


@reg("div.Tensor", "div.Scalar", "div_.Tensor", "div_.Scalar", "div.out",
     "div.Tensor_mode", "div.Scalar_mode", "div_.Tensor_mode", "div_.Scalar_mode", "div.out_mode")
def _div(c):
    x, y = c.a["self"], c.a["other"]
    mode = c.a.get("rounding_mode")
    ox, oy = _ops([x, y])
    if c.inplace:
        if mode is None:
            _pcall(fnp.divide, [ox, oy], A(x))
        else:
            _div_mode(ox, oy, mode, out=A(x))
        return x
    out = c.a.get("out")
    if out is not None:
        _check_out(c, out, x, y)
        if mode is None:
            _pcall(fnp.divide, [ox, oy], A(out))
        else:
            _div_mode(ox, oy, mode, out=A(out))
        return out
    if mode is None:
        return _ufunc_out(c, fnp.divide, [ox, oy])
    if mode == "floor":
        return _ufunc_out(c, fnp.floor_divide, [ox, oy])
    return _out(c, _div_mode(ox, oy, mode))


def _rsqrt(x, out=None):
    return fnp.reciprocal(fnp.sqrt(x), out=out)          # literal sqrt then reciprocal


def _relu(x, out=None):
    return fnp.maximum(x, 0, out=out)


_UNARY = {"abs": fnp.abs, "neg": fnp.negative, "sign": fnp.sign, "sqrt": fnp.sqrt, "rsqrt": _rsqrt,
          "exp": fnp.exp, "log": fnp.log, "log1p": fnp.log1p, "expm1": fnp.expm1, "sin": fnp.sin,
          "cos": fnp.cos, "tanh": fnp.tanh, "floor": fnp.floor, "ceil": fnp.ceil, "trunc": fnp.trunc,
          "reciprocal": fnp.reciprocal, "relu": _relu, "logical_not": fnp.logical_not,
          "bitwise_not": fnp.invert, "isnan": fnp.isnan, "isinf": fnp.isinf, "isfinite": fnp.isfinite}
_INT_TO_FLOAT_FIRST = {"reciprocal", "rsqrt"}   # numpy would run these as integer loops


def _unary(name, fn):
    def h(c):
        x = c.a["self"]
        if c.inplace:
            _pcall(fn, [A(x)], A(x))
            return x
        xa = A(x)
        if name in _INT_TO_FLOAT_FIRST and not x.dtype.is_floating_point:
            xa = xa.astype(NP_DTYPE[_meta(c).dtype])
        return _ufunc_out(c, fn, [xa])
    return h


for _n, _f in _UNARY.items():
    _keys = [f"{_n}.default"]
    if _n not in ("isnan", "isinf", "isfinite", "logical_not", "bitwise_not"):
        _keys.append(f"{_n}_.default")
    reg(*_keys)(_unary(_n, _f))


@reg("erf.default")
def _erf(c):
    """Public erf port: 2*stats.norm.cdf(sqrt(2)*x)-1, cast back to torch's dtype (charged)."""
    x = A(c.a["self"])
    with warnings.catch_warnings():                   # float32 -> float64 promotion is the priced port
        warnings.simplefilter("ignore", flops.FlopscopeWarning)
        y = 2.0 * flops.stats.norm.cdf(x * math.sqrt(2.0)) - 1.0
    return _out(c, y)


@reg("special_hermite_polynomial_he.n_scalar")
def _hermite(c):
    """Explicit probabilists' Hermite He_n(x), scalar nonnegative integer degree."""
    n = c.a["n"]
    if isinstance(n, float) and n.is_integer():
        n = int(n)
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        _fail(c, f"degree {n!r} (scalar nonnegative integer only)")
    meta = _meta(c)
    x = A(c.a["x"])
    if n == 0:
        r = fnp.ones_like(x, dtype=NP_DTYPE[meta.dtype])
    elif n == 1:
        r = x                                         # free; _T copies (unmetered) so output never aliases x
    elif n == 2:
        r = x * x - 1.0
    elif n == 3:
        r = x * (x * x - 3.0)
    else:
        prev, r = x, x * x - 1.0
        for k in range(3, n + 1):
            prev, r = r, x * r - (k - 1) * prev
    return _T(r, meta, c.tensors)


@reg("clamp.default", "clamp.Tensor", "clamp_.default", "clamp_.Tensor", "clamp_min.default", "clamp_min.Tensor",
     "clamp_min_.default", "clamp_min_.Tensor", "clamp_max.default", "clamp_max.Tensor", "clamp_max_.default",
     "clamp_max_.Tensor")
def _clamp(c):
    x = c.a["self"]
    ox, lo, hi = _ops([x, c.a.get("min"), c.a.get("max")])
    steps = [(f, v) for f, v in ((fnp.maximum, lo), (fnp.minimum, hi)) if v is not None]
    if not steps:
        _fail(c, "clamp without bounds")
    r = ox
    for f, v in steps[:-1]:
        r = f(r, v)
    f, v = steps[-1]
    if c.inplace:
        f(r, v, out=A(x))
        return x
    return _ufunc_out(c, f, [r, v])


@reg("where.self", "where.ScalarSelf", "where.ScalarOther", "where.Scalar")
def _where(c):
    return _out(c, fnp.where(*_ops([c.a["condition"], c.a["self"], c.a["other"]])))


# ---------------------------------------------------------------- copies, casts, fills, factories
def _copy_into_layout(meta, src):
    """torch's output layout is not C order: one charged fnp.copyto per element into a torch-layout
buffer (the same per-element price as astype/copy/fill, without a layout copy afterwards)."""
    buf = _buffer(meta)
    _pcopyto(A(buf), src, casting="unsafe")
    return buf


@reg("_to_copy.default")
def _to_copy(c):
    meta = _meta(c)
    x = A(c.a["self"])
    if not meta.is_contiguous() or (_POOL is not None and x.size >= PAR_MIN):
        return _copy_into_layout(meta, x)              # charged copyto per element = astype's bill
    return _T(x.astype(NP_DTYPE[meta.dtype], order="C"), meta, c.tensors)


@reg("clone.default", "lift_fresh_copy.default")
def _clone(c):
    meta = _meta(c)
    x = A(c.a["self"])
    if not meta.is_contiguous() or (_POOL is not None and x.size >= PAR_MIN):
        return _copy_into_layout(meta, x)              # charged copyto per element = fnp.copy's bill
    return _T(fnp.copy(x, order="C"), meta, c.tensors)


@reg("copy_.default")
def _copy_(c):
    dst, src = c.a["self"], c.a["src"]
    if isinstance(src, torch.Tensor) and src.dim() == dst.dim() and src.shape[0] == dst.shape[0]:
        _pcopyto(A(dst), A(src), casting="unsafe")
    else:
        fnp.copyto(A(dst), A(src) if isinstance(src, torch.Tensor) else src, casting="unsafe")
    return dst


@reg("fill_.Scalar", "fill_.Tensor", "zero_.default")
def _fill_(c):
    x = c.a["self"]
    v = 0 if c.key == "zero_.default" else c.a["value"]
    if isinstance(v, torch.Tensor):
        if v.dim() != 0:
            _fail(c, "fill_ value must be 0-dim")
        v = v.item()
    _pcopyto(A(x), np.asarray(v, dtype=NP_DTYPE[x.dtype]))   # one charged write per element
    return x


@reg("ones.default", "new_ones.default")
def _ones(c):
    meta = _meta(c)
    return _T(fnp.ones(tuple(meta.shape), dtype=NP_DTYPE[meta.dtype]), meta)


@reg("ones_like.default")
def _ones_like(c):
    meta = _meta(c)
    if not meta.is_contiguous():
        return _copy_into_layout(meta, np.asarray(1, dtype=NP_DTYPE[meta.dtype]))
    return _T(fnp.ones(tuple(meta.shape), dtype=NP_DTYPE[meta.dtype]), meta)


@reg("full.default", "new_full.default")
def _full(c):
    meta = _meta(c)
    return _T(fnp.full(tuple(meta.shape), c.a["fill_value"], dtype=NP_DTYPE[meta.dtype]), meta)


@reg("full_like.default")
def _full_like(c):
    meta = _meta(c)
    if not meta.is_contiguous():
        return _copy_into_layout(meta, np.asarray(c.a["fill_value"], dtype=NP_DTYPE[meta.dtype]))
    return _T(fnp.full(tuple(meta.shape), c.a["fill_value"], dtype=NP_DTYPE[meta.dtype]), meta)


@reg("scalar_tensor.default")
def _scalar_tensor(c):
    meta = _meta(c)
    return _T(fnp.full((), c.a["s"], dtype=NP_DTYPE[meta.dtype]), meta)


@reg("eye.default", "eye.m")
def _eye(c):
    meta = _meta(c)
    n = c.a["n"]
    m = c.a.get("m")
    return _T(fnp.eye(n, n if m is None else m, dtype=NP_DTYPE[meta.dtype]), meta)


@reg("arange.default", "arange.start", "arange.start_step")
def _arange(c):
    meta = _meta(c)
    start, end, step = (0, c.a["end"], 1) if c.key == "arange.default" else (c.a["start"], c.a["end"], c.a.get("step", 1))
    return _T(fnp.arange(start, end, 1 if step is None else step, dtype=NP_DTYPE[meta.dtype]), meta)


def _rng(c):
    seed = int(torch.randint(0, 2**62, (1,), generator=c.a.get("generator")).item())
    return fnp.random.default_rng(seed)


@reg("normal_.default")
def _normal_(c):
    x = c.a["self"]
    if not x.dtype.is_floating_point:
        _fail(c, f"dtype {x.dtype}")
    o = A(x)
    _rng(c).standard_normal(dtype=NP_DTYPE[x.dtype], out=o)
    if c.a["std"] != 1:
        fnp.multiply(o, c.a["std"], out=o)
    if c.a["mean"] != 0:
        fnp.add(o, c.a["mean"], out=o)
    return x


@reg("uniform_.default")
def _uniform_(c):
    x = c.a["self"]
    if not x.dtype.is_floating_point:
        _fail(c, f"dtype {x.dtype}")
    lo, hi = c.a["from"], c.a["to"]
    o = A(x)
    _rng(c).random(dtype=NP_DTYPE[x.dtype], out=o)
    if hi - lo != 1:
        fnp.multiply(o, hi - lo, out=o)
    if lo != 0:
        fnp.add(o, lo, out=o)
    return x


# ---------------------------------------------------------------- matmul family
@reg("mm.default", "bmm.default", "matmul.default", "mv.default", "mm.out", "bmm.out")
def _mm(c):
    a, b = c.a["self"], c.a.get("mat2", c.a.get("other", c.a.get("vec")))
    out = c.a.get("out")
    if out is not None:
        fnp.matmul(A(a), A(b), out=A(out))
        return out
    return _out(c, fnp.matmul(A(a), A(b)))


@reg("dot.default")
def _dot(c):
    return _out(c, fnp.dot(A(c.a["self"]), A(c.a["tensor"])))


@reg("addmm.default", "addmm_.default", "baddbmm.default", "baddbmm_.default", "addmv.default", "addmv_.default")
def _addmm(c):
    """beta*self + alpha*(m1 @ m2) as the explicit numpy steps (trace_cost.py mapping)."""
    names = {"addmm": ("mat1", "mat2"), "baddbmm": ("batch1", "batch2"), "addmv": ("mat", "vec")}
    m1, m2 = (c.a[k] for k in names[c.key.split(".")[0].rstrip("_")])
    inp, beta, alpha = c.a["self"], c.a["beta"], c.a["alpha"]
    if c.inplace:
        o = A(inp)
        if beta == 0:
            fnp.matmul(A(m1), A(m2), out=o)
            if alpha != 1:
                fnp.multiply(o, alpha, out=o)
            return inp
        t = fnp.matmul(A(m1), A(m2))
        if alpha != 1:
            fnp.multiply(t, alpha, out=t)
        if beta != 1:
            fnp.multiply(o, beta, out=o)
        fnp.add(o, t, out=o)
        return inp
    t = fnp.matmul(A(m1), A(m2))
    if alpha != 1:
        fnp.multiply(t, alpha, out=t)
    if beta != 0:
        b = A(inp) if beta == 1 else fnp.multiply(A(inp), beta)
        fnp.add(t, b, out=t)
    return _out(c, t)


# ---------------------------------------------------------------- linear algebra
@reg("_linalg_eigh.default", "linalg_eigh.default")
def _eigh(c):
    a = A(c.a.get("A", c.a.get("self")))
    uplo = c.a.get("UPLO") or "L"
    mL, mQ = _meta(c)
    if c.a.get("compute_v", True) is False:
        w = fnp.linalg.eigvalsh(a, UPLO=uplo)
        return _T(w, mL, c.tensors), torch.empty_strided(tuple(mQ.shape), mQ.stride(), dtype=mQ.dtype)
    w, v = fnp.linalg.eigh(a, UPLO=uplo)
    return _T(w, mL, c.tensors), _T(v, mQ, c.tensors)


@reg("linalg_qr.default")
def _qr(c):
    a, mode = A(c.a["A"]), c.a["mode"]
    mQ, mR = _meta(c)
    if mode == "r":
        r = fnp.linalg.qr(a, mode="r")
        return torch.empty_strided(tuple(mQ.shape), mQ.stride(), dtype=mQ.dtype), _T(r, mR, c.tensors)
    if mode not in ("reduced", "complete"):
        _fail(c, f"mode {mode!r}")
    q, r = fnp.linalg.qr(a, mode=mode)
    return _T(q, mQ, c.tensors), _T(r, mR, c.tensors)


@reg("_linalg_svd.default", "linalg_svd.default")
def _svd(c):
    a = A(c.a["A"])
    full, uv = bool(c.a["full_matrices"]), c.a.get("compute_uv", True)
    mU, mS, mV = _meta(c)
    if not uv:
        s = fnp.linalg.svd(a, compute_uv=False)
        e = lambda m: torch.empty_strided(tuple(m.shape), m.stride(), dtype=m.dtype)
        return e(mU), _T(s, mS, c.tensors), e(mV)
    u, s, vh = fnp.linalg.svd(a, full_matrices=full, compute_uv=True)
    # torch 2.14's meta kernel reports Vh row-major; the CPU kernel returns it column-major per matrix
    sh = tuple(mV.shape)
    v_stride = _contig(sh[:-2] + (sh[-1], sh[-2]))
    v_stride = v_stride[:-2] + (v_stride[-1], v_stride[-2])
    return _T(u, mU, c.tensors), _T(s, mS, c.tensors), _T(vh, (mV.dtype, sh, v_stride), c.tensors)


@reg("trace.default")
def _trace(c):
    return _out(c, fnp.trace(A(c.a["self"])))


# ---------------------------------------------------------------- reductions
def _axis(dim):
    if dim is None:
        return None
    if isinstance(dim, int):
        return dim
    dim = tuple(dim)
    return None if not dim else dim


def _npdt(dt):
    return None if dt is None else NP_DTYPE[dt]


@reg("sum.default", "sum.dim_IntList", "mean.default", "mean.dim", "prod.default", "prod.dim_int")
def _sum(c):
    fn = {"sum": fnp.sum, "mean": fnp.mean, "prod": fnp.prod}[c.key.split(".")[0]]
    kw = dict(axis=_axis(c.a.get("dim")), dtype=_npdt(c.a.get("dtype")))
    if c.a.get("keepdim"):
        kw["keepdims"] = True
    return _out(c, fn(A(c.a["self"]), **kw))


@reg("max.default", "min.default", "amax.default", "amin.default", "any.default", "any.dim", "any.dims",
     "all.default", "all.dim", "all.dims", "argmax.default", "argmin.default")
def _reduce(c):
    name = c.key.split(".")[0]
    fn = {"max": fnp.max, "min": fnp.min, "amax": fnp.max, "amin": fnp.min, "any": fnp.any, "all": fnp.all,
          "argmax": fnp.argmax, "argmin": fnp.argmin}[name]
    kw = dict(axis=_axis(c.a.get("dim")))
    if c.a.get("keepdim"):
        kw["keepdims"] = True
    return _out(c, fn(A(c.a["self"]), **kw))


@reg("max.dim", "min.dim")
def _max_dim(c):
    vf, if_ = (fnp.max, fnp.argmax) if c.key.startswith("max") else (fnp.min, fnp.argmin)
    x, dim, keep = A(c.a["self"]), c.a["dim"], bool(c.a["keepdim"])
    mv, mi = _meta(c)
    return _T(vf(x, axis=dim, keepdims=keep), mv, c.tensors), _T(if_(x, axis=dim, keepdims=keep), mi, c.tensors)


@reg("cumsum.default", "cumprod.default")
def _cumsum(c):
    fn = fnp.cumsum if c.key.startswith("cumsum") else fnp.cumprod
    return _out(c, fn(A(c.a["self"]), axis=c.a["dim"], dtype=_npdt(c.a.get("dtype"))))


@reg("linalg_vector_norm.default")
def _vector_norm(c):
    x = A(c.a["self"])
    if c.a.get("dtype") is not None:
        x = x.astype(NP_DTYPE[c.a["dtype"]])
    r = fnp.linalg.vector_norm(x, ord=c.a["ord"], axis=_axis(c.a["dim"]), keepdims=bool(c.a["keepdim"]))
    return _out(c, r)


@reg("allclose.default")
def _allclose(c):
    return bool(fnp.allclose(A(c.a["self"]), A(c.a["other"]), rtol=c.a["rtol"], atol=c.a["atol"],
                             equal_nan=c.a["equal_nan"]))


@reg("equal.default")
def _equal(c):
    return bool(fnp.array_equal(A(c.a["self"]), A(c.a["other"])))


# ---------------------------------------------------------------- data movement and indexing
@reg("cat.default")
def _cat(c):
    ts = list(c.a["tensors"])
    if any(t.dim() > 1 for t in ts):                 # torch's legacy skip of 1-D empty tensors
        ts = [t for t in ts if not (t.dim() == 1 and t.shape[0] == 0)]
    return _out(c, fnp.concatenate([A(t) for t in ts], axis=c.a["dim"]))


@reg("stack.default")
def _stack(c):
    return _out(c, fnp.stack([A(t) for t in c.a["tensors"]], axis=c.a["dim"]))


@reg("flip.default")
def _flip(c):
    return _out(c, fnp.flip(A(c.a["self"]), axis=tuple(c.a["dims"])))   # free view; torch layout via _T


@reg("diagonal_copy.default")
def _diagonal_copy(c):
    d = fnp.diagonal(A(c.a["self"]), c.a["offset"], c.a["dim1"], c.a["dim2"])
    return _out(c, d.copy())                          # free diagonal view, charged copy


@reg("diag_embed.default")
def _diag_embed(c):
    x, off, d1, d2 = c.a["self"], c.a["offset"], c.a["dim1"], c.a["dim2"]
    meta = _meta(c)
    if x.dim() == 1 and off == 0:
        return _T(fnp.diag(A(x)), meta, c.tensors)  # zero allocation + one charged write per element
    out = torch.zeros(tuple(meta.shape), dtype=meta.dtype)
    fnp.copyto(A(out.diagonal(off, d1, d2)), A(x))  # same price: one charged write per input element
    return out


@reg("index_select.default")
def _index_select(c):
    idx = c.a["index"]
    if idx.dim() == 0:
        idx = idx.reshape(1)
    return _out(c, fnp.take(A(c.a["self"]), A(idx), axis=c.a["dim"]))


@reg("index.Tensor")
def _index(c):
    x, indices = c.a["self"], c.a["indices"]
    if all(i is None for i in indices):
        _fail(c, "no index tensors")
    for i in indices:
        if i is not None and i.dtype not in (torch.int64, torch.bool):
            _fail(c, f"index dtype {i.dtype}")
    r = A(x)[tuple(slice(None) if i is None else A(i) for i in indices)]
    if any(i is not None and i.dtype == torch.bool for i in indices):
        return _T(r, (x.dtype, r.shape, _contig(r.shape)), c.tensors)   # data-dependent shape: no meta
    return _out(c, r)


def _flat_index(c, x, indices):
    """C-order flat positions of x[indices] (numpy/torch advanced indexing) with charged fnp int64 arithmetic.

Axes indexed by the same index tensor are combined as index*(sum of their strides) (the B-switch
i=arange(n) diagonal contract of trace_cost.diagonal_put_port); every free axis adds a charged
arange(size)*stride coordinate. Indices are bounds-checked (unmetered, validation only)."""
    s = tuple(x.shape)
    nd = len(s)
    cs = [math.prod(s[j + 1:]) for j in range(nd)]
    per = []
    for i in indices:
        if i is None:
            per.append(None)
        elif i.dtype == torch.bool:
            for j, arr in enumerate(fnp.nonzero(A(i))):
                per.append((("nz", id(i), j), arr))
        elif i.dtype == torch.int64:
            per.append(((i.untyped_storage().data_ptr(), i.storage_offset(), tuple(i.shape), i.stride()), A(i)))
        else:
            _fail(c, f"index dtype {i.dtype}")
    if len(per) > nd:
        _fail(c, "too many indices")
    per += [None] * (nd - len(per))
    adv = [j for j, p in enumerate(per) if p is not None]
    if not adv:
        _fail(c, "no index tensors")
    for j in adv:
        raw = np.asarray(per[j][1])
        if raw.size and (raw.min() < 0 or raw.max() >= s[j]):
            _fail(c, f"index out of [0, {s[j]}) on axis {j} (negative-index wrap not ported)")
    B = np.broadcast_shapes(*(per[j][1].shape for j in adv))
    groups = {}
    for j in adv:
        k, arr = per[j]
        if k in groups:
            groups[k][1] += cs[j]
        else:
            groups[k] = [arr, cs[j]]
    flat = None
    for arr, st in groups.values():
        term = arr if st == 1 else fnp.multiply(arr, st)
        flat = term if flat is None else fnp.add(flat, term)
    free = [j for j in range(nd) if per[j] is None]
    if adv == list(range(adv[0], adv[-1] + 1)):      # adjacent advanced axes keep their place
        order = [j for j in free if j < adv[0]] + ["B"] + [j for j in free if j > adv[-1]]
    else:                                             # separated advanced axes move to the front
        order = ["B"] + free
    res_shape = []
    for o in order:
        res_shape.extend(B if o == "B" else (s[o],))
    if free:
        lead = order.index("B")
        flat = flat[(None,) * lead + (Ellipsis,) + (None,) * (len(order) - lead - 1)]   # free view
        d = 0
        for o in order:
            if o == "B":
                d += len(B)
                continue
            coord = fnp.arange(s[o], dtype=np.int64)
            if cs[o] != 1:
                coord = fnp.multiply(coord, cs[o])
            sel = [None] * len(res_shape)
            sel[d] = slice(None)
            flat = fnp.add(flat, coord[tuple(sel)])
            d += 1
    return flat, tuple(res_shape)


def _put(c, target, x):
    if c.a["accumulate"]:
        _fail(c, "accumulate=True")
    v = c.a["values"]
    if v.dtype != x.dtype:
        _fail(c, f"values dtype {v.dtype} != self dtype {x.dtype}")
    flat, res_shape = _flat_index(c, x, c.a["indices"])
    fnp.put(target, flat, fnp.broadcast_to(A(v), res_shape))


@reg("index_put_.default")
def _index_put_(c):
    x = c.a["self"]
    _put(c, A(x), x)
    return x


@reg("index_put.default")
def _index_put(c):
    x = c.a["self"]
    meta = _meta(c)
    r = fnp.copy(A(x), order="C" if meta.is_contiguous() else "K")
    _put(c, r, x)
    return _T(r, meta, c.tensors)


HANDLED_OPS = tuple(sorted(_H))


# ---------------------------------------------------------------- the mode
class FlopscopeMode(TorchDispatchMode):
    """Executes every ATen op through flopscope.numpy inside the active BudgetContext."""

    def __init__(self):
        super().__init__()
        self.stats = defaultdict(lambda: [0, 0, 0])   # op -> [calls, flops, unmetered_copy_elems]

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        key = _key(func)
        h = _H.get(key)
        if h is None:
            raise NotImplementedError(f"FlopscopeMode: no flopscope port for aten.{key}")
        c = _Call(func, key, args, kwargs)
        _validate(c)
        ctx = get_active_budget()
        f0 = ctx.flops_used if ctx is not None else 0
        _UNMETERED[0] = 0
        with np.errstate(all="ignore"):              # torch is silent on inf/nan; keep numpy silent too
            out = h(c)
        ctx1 = get_active_budget()
        if ctx1 is not ctx:                           # flopscope's implicit global default appeared
            f0 = 0
        rec = self.stats[key]
        rec[0] += 1
        rec[1] += (ctx1.flops_used if ctx1 is not None else 0) - f0
        rec[2] += _UNMETERED[0]
        return out

    # ------------------------------------------------------------ report helpers
    def report(self):
        """Per-op rows sorted by flops: op, calls, flops, unmetered_copy_elems."""
        rows = [dict(op=k, calls=v[0], flops=v[1], unmetered_copy_elems=v[2]) for k, v in self.stats.items()]
        rows.sort(key=lambda r: (-r["flops"], -r["calls"], r["op"]))
        return rows

    def total_flops(self):
        return sum(v[1] for v in self.stats.values())

    def reset_report(self):
        self.stats.clear()

    def format_report(self, top=None):
        rows = self.report()
        total = self.total_flops() or 1
        lines = [f"{'op':34s} {'calls':>8s} {'flops':>16s} {'share':>7s} {'unmetered_copy':>15s}"]
        for r in rows[:top]:
            lines.append(f"{r['op']:34s} {r['calls']:8d} {r['flops']:16,d} {100 * r['flops'] / total:6.2f}% "
                         f"{r['unmetered_copy_elems']:15,d}")
        lines.append(f"{'total':34s} {sum(r['calls'] for r in rows):8d} {self.total_flops():16,d}")
        return "\n".join(lines)
