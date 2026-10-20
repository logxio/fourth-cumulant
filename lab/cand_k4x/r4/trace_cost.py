"""Model actual ATen traces with public flopscope accounting prices.

This is a shape/operation cost MODEL, not a BudgetContext run of the chain.
The model specifies a literal NumPy/flopscope port, without symmetry tags.
Torch fused alpha/beta operations are priced as their explicit NumPy steps.
Unknown operators, unsupported dtypes and missing metadata stay explicit;
the complete modeled total is null until every record has a mapping.

Run --check for small real BudgetContext checks, or pass an aten-shapes-v1
JSON followed by --output. No large tensors are allocated by trace costing.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import platform
import warnings
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import flopscope.accounting as accounting

UNIT = 2**31


def tensors(x):
    if isinstance(x, dict):
        if x.get('tensor'):
            yield x
        elif 'repeat' in x and 'item' in x:
            yield from tensors(x['item'])
        else:
            for v in x.values():
                yield from tensors(v)
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from tensors(v)


def shape(t):
    return tuple(int(x) for x in t['shape'])


def numel(s):
    return math.prod(s)


def pw(name, s):
    return accounting.pointwise_cost(name, shape=tuple(s))


@lru_cache(None)
def dtype_rate(dtype):
    """Public BudgetContext calibration, six elements, never private APIs."""
    dtype = str(dtype).removeprefix('torch.')
    if dtype.startswith('complex'):
        raise ValueError('complex operator-specific multiplier requires explicit mapping')
    try:
        dt = np.dtype(dtype)
        a = fnp.array(np.arange(6).astype(dt).reshape(2, 3))
        b = fnp.array(np.arange(6, 12).astype(dt).reshape(2, 3))
        ctx = flops.BudgetContext(flop_budget=2**41, quiet=True)
        with ctx:
            fnp.add(a, b)
        return ctx.flops_used / pw('add', (2, 3))
    except Exception as exc:
        raise ValueError(f'unsupported billing dtype {dtype}: {exc}') from exc


def matmul_cost(a, b):
    """Closed matmul specialization of accounting.einsum_cost, no allocation.

The public helper allocates shape-sized np.empty operands, so using it on
n-by-n^2 trace shapes would violate the local memory limit. The n32 check
checks this specialization against both that public helper and runtime.
"""
    a, b = tuple(a), tuple(b)
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            raise ValueError('dot dimensions')
        return int(max(2*a[0]-1, 0)*accounting.get_weight('matmul'))
    av, bv = len(a) == 1, len(b) == 1
    aa = (1, a[0]) if av else a
    bb = (b[0], 1) if bv else b
    if aa[-1] != bb[-2]:
        raise ValueError('matmul dimensions')
    batch = numel(np.broadcast_shapes(aa[:-2], bb[:-2]))
    return int(batch * aa[-2] * bb[-1] * max(2*aa[-1]-1, 0)*accounting.get_weight('matmul'))


def erf_port(x):
    y = 2.0 * flops.stats.norm.cdf(x * math.sqrt(2.0)) - 1.0
    return y.astype(x.dtype) if y.dtype != x.dtype else y


def hermite_port(x, degree):
    """Explicit probabilists' Hermite polynomial, scalar nonnegative degree."""
    if degree == 0:
        return fnp.ones_like(x)
    if degree == 1:
        return x
    if degree == 2:
        return x*x-1.0
    if degree == 3:
        return x*(x*x-3.0)
    prev, cur = x, x*x-1.0
    for k in range(3, degree+1):
        prev, cur = cur, x*cur-(k-1)*prev
    return cur


def diagonal_put_port(array, index, value, indexed_axes):
    """B-switch repeated-index replacement using public mutable fnp.put.

The source expand() uses i=arange(n) on equal-length axes. Flat-index
arithmetic is part of this literal port and is charged, including int64.
One remaining free axis is allowed; no large dense boolean mask is built.
"""
    s = array.shape
    strides = tuple(math.prod(s[j+1:]) for j in range(len(s)))
    ds = sum(strides[j] for j in indexed_axes)
    diagonal = index*ds if ds != 1 else index
    free = [j for j in range(len(s)) if j not in indexed_axes]
    if free:
        if len(free) != 1:
            raise ValueError('diagonal_put_port supports at most one free axis')
        fs = strides[free[0]]
        other = index*fs if fs != 1 else index
        # Adjacent advanced indices after an initial basic slice retain the
        # free leading axis; separated advanced indices move to the front.
        basic_first = free[0] == 0 and tuple(indexed_axes) == tuple(range(1,len(s)))
        flat = other[:,None]+diagonal[None,:] if basic_first else diagonal[:,None]+other[None,:]
    else:
        flat = diagonal
    fnp.put(array, flat, value)


@lru_cache(None)
def erf_unit_cost(dtype):
    x = fnp.array(np.linspace(-1, 1, 6).astype(dtype))
    ctx = flops.BudgetContext(flop_budget=2**41, quiet=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with ctx:
            erf_port(x)
    return ctx.flops_used / 6


POINTWISE = {
    'add': 'add', 'sub': 'subtract', 'rsub': 'subtract', 'mul': 'multiply',
    'div': 'divide', 'true_divide': 'divide', 'neg': 'negative', 'abs': 'absolute',
    'pow': 'power', 'exp': 'exp', 'log': 'log', 'log1p': 'log1p',
    'sqrt': 'sqrt', 'square': 'square', 'reciprocal': 'reciprocal',
    'sin': 'sin', 'cos': 'cos', 'tanh': 'tanh',
    'maximum': 'maximum', 'minimum': 'minimum',
    'eq': 'equal', 'ne': 'not_equal', 'lt': 'less', 'le': 'less_equal',
    'gt': 'greater', 'ge': 'greater_equal', 'isfinite': 'isfinite',
    'isinf': 'isinf', 'isnan': 'isnan',
    'logical_and': 'logical_and', 'logical_or': 'logical_or',
    'logical_not': 'logical_not', 'bitwise_and': 'bitwise_and',
    'bitwise_or': 'bitwise_or', 'bitwise_not': 'invert',
}
ZERO = {
    'detach', 'alias', 't', 'transpose', 'permute', 'slice', 'select',
    'unsqueeze', 'squeeze', 'expand', 'as_strided', 'diagonal',
    'empty', 'empty_like', 'empty_strided', 'new_empty', 'new_empty_strided',
    'zeros', 'zeros_like', 'new_zeros',
    'resolve_conj', 'resolve_neg', '_conj',
    'unbind', 'flip', '_local_scalar_dense',
}


def record_cost(rec):
    op = rec['op'].split('.')
    if len(op) < 2 or op[0] != 'aten':
        raise ValueError('not an ATen operator')
    name = op[1].rstrip('_')
    args, kw = rec.get('args', []), rec.get('kwargs', {})
    ts = list(tensors(args))
    ots = list(tensors(rec.get('outputs')))
    if name in ZERO:
        return 0, 'metadata or uninitialized/zero allocation; no numeric arithmetic'
    out = shape(ots[0]) if ots else (shape(ts[0]) if ts else ())
    # widest participating numeric buffer, including cast destinations.
    dtypes = [t['dtype'].removeprefix('torch.') for t in ts + ots]
    # Index tensors do not determine floating arithmetic width.
    floats = [d for d in dtypes if d.startswith(('float', 'bfloat', 'complex'))]
    billed = max(floats or dtypes or ['float32'], key=lambda d: np.dtype(d).itemsize)
    rate = dtype_rate(billed)
    why = ''
    if name in POINTWISE:
        cost = pw(POINTWISE[name], out)
        alpha = kw.get('alpha', 1)
        if name in ('add', 'sub') and alpha != 1:
            cost += pw('multiply', shape(ts[1]) if len(ts) > 1 else out)
        why = 'public pointwise_cost; explicit alpha multiply when present'
    elif name in ('mm', 'bmm', 'matmul', 'mv', 'dot'):
        cost = matmul_cost(shape(ts[0]), shape(ts[1]))
        why = 'matmul specialization, checked against public einsum_cost'
    elif name in ('addmm', 'baddbmm', 'addmv'):
        cost = matmul_cost(shape(ts[1]), shape(ts[2]))
        alpha, beta = kw.get('alpha', 1), kw.get('beta', 1)
        if alpha != 1:
            cost += pw('multiply', out)
        if beta != 0:
            cost += pw('add', out)
            if beta != 1:
                cost += pw('multiply', shape(ts[0]))
        why = 'literal matmul plus alpha/beta scaling and addition'
    elif name in ('sum', 'mean', 'prod', 'amax', 'amin', 'max', 'min', 'all', 'any'):
        if len(ts) != 1:
            raise ValueError('binary/tuple reduction overload needs explicit mapping')
        axis = kw.get('dim', args[1] if len(args) > 1 else None)
        axis = tuple(axis) if isinstance(axis, list) else axis
        axis = None if axis == () else axis
        reduce_name = {'amax': 'max', 'amin': 'min'}.get(name, name)
        cost = accounting.reduction_cost(reduce_name, input_shape=shape(ts[0]), axis=axis)
        if name == 'mean':
            cost += pw('divide', out)
        why = 'public reduction_cost with traced axes'
    elif name in ('linalg_qr',):
        mode = kw.get('mode', args[1] if len(args) > 1 else 'reduced')
        s = shape(ts[0]);cost = numel(s[:-2])*accounting.qr_cost(*s[-2:], mode=mode)
        why = 'public qr_cost including Q formation'
    elif name in ('_linalg_eigh', 'linalg_eigh', 'linalg_eigvalsh'):
        s = shape(ts[0])
        vectors = name != 'linalg_eigvalsh'
        if name == '_linalg_eigh':
            vectors = kw.get('compute_v', args[2] if len(args) > 2 else True)
        fn = accounting.eigh_cost if vectors else accounting.eigvalsh_cost
        cost = numel(s[:-2])*fn(s[-1]);why = 'public symmetric eigendecomposition cost'
    elif name in ('_linalg_svd', 'linalg_svd', 'svd'):
        s = shape(ts[0])
        full = kw.get('full_matrices', args[1] if len(args) > 1 else False)
        vectors = kw.get('compute_uv', args[2] if len(args) > 2 else True)
        cost = numel(s[:-2])*accounting.svd_cost(*s[-2:], with_vectors=vectors, full_matrices=full)
        why = 'public SVD cost, traced vector/full-matrix modes'
    elif name in ('linalg_vector_norm', 'norm'):
        order = kw.get('ord', args[1] if len(args) > 1 else 2)
        cost = accounting.vector_norm_cost(shape(ts[0]), ord=order)
        why = 'public vector_norm_cost'
    elif name == 'trace':
        cost = accounting.trace_cost(min(shape(ts[0])[-2:]));why = 'public trace_cost'
    elif name == 'allclose':
        s = np.broadcast_shapes(shape(ts[0]),shape(ts[1]))
        cost = (pw('subtract',s)+2*pw('absolute',s)+pw('multiply',s)
                +pw('add',s)+pw('less_equal',s)
                +accounting.reduction_cost('all',input_shape=s))
        why = 'literal public allclose price: six pointwise operations plus all reduction; reference check retained'
    elif name == 'diag_embed':
        cost = pw('diag',shape(ts[0]));why = 'zero allocation plus one charged diagonal write per input element'
    elif name == 'diagonal_copy':
        cost = pw('copy',out);why = 'free diagonal view followed by charged copy'
    elif name == 'eye':
        cost = pw('eye',(min(out[-2:]),));why = 'zero allocation plus charged diagonal ones'
    elif name == 'arange':
        cost = 2*pw('arange',out);why = 'public arange start+i*step, two operations per element including integer dtype rate'
    elif name == 'special_hermite_polynomial_he':
        degree = args[1]
        if not isinstance(degree,int) or degree < 0:
            raise ValueError('Hermite port requires scalar nonnegative integer degree')
        if degree == 0:
            cost = pw('ones_like',out)
        elif degree == 1:
            cost = 0
        elif degree == 2:
            cost = pw('multiply',out)+pw('subtract',out)
        elif degree == 3:
            cost = 2*pw('multiply',out)+pw('subtract',out)
        else:
            cost = pw('multiply',out)+pw('subtract',out)+(degree-2)*(2*pw('multiply',out)+pw('subtract',out))
        why = 'explicit scalar-degree probabilists Hermite polynomial port; small runtime checked'
    elif name in ('clone', 'copy', 'contiguous', '_to_copy', 'to'):
        cost = pw('astype' if name in ('_to_copy', 'to') else 'copy', out)
        why = 'materialization priced per output element, widest buffer dtype'
    elif name in ('view', '_unsafe_view', 'reshape', 'reshape_alias', '_reshape_alias', 'flatten'):
        cost = pw('reshape', out);why = 'literal fnp.reshape mapping, per output element'
    elif name in ('cat', 'stack'):
        cost = pw('concatenate' if name == 'cat' else 'stack', out)
        why = 'materialization priced per output element'
    elif name in ('fill', 'zero', 'full', 'full_like', 'new_full', 'ones', 'ones_like', 'new_ones', 'scalar_tensor', 'lift_fresh'):
        cost = pw('fill', out);why = 'numeric fill priced per output element'
    elif name in ('clamp_min', 'clamp_max', 'relu'):
        cost = pw('maximum' if name != 'clamp_max' else 'minimum', out)
        why = 'one public pointwise min/max'
    elif name == 'clamp':
        lo = kw.get('min', args[1] if len(args) > 1 else None)
        hi = kw.get('max', args[2] if len(args) > 2 else None)
        cost = int(lo is not None)*pw('maximum', out)+int(hi is not None)*pw('minimum', out)
        why = 'one comparison per supplied bound'
    elif name == 'index_select':
        cost = pw('take', out);why = 'public gather/materialization price per selected element'
    elif name == 'index':
        indices = list(tensors(args[1]))
        scan = sum(numel(shape(t)) for t in indices if t['dtype']=='bool')
        # getitem's select-tier weight applies to gathered elements; the
        # documented boolean scan is separately one operation per mask cell.
        cost = 4*pw('getitem',out)+scan
        why = 'public advanced getitem gather; integer indices carry no extra scan'
    elif name == 'index_put':
        if kw.get('accumulate',args[3] if len(args)>3 else False):
            raise ValueError('accumulating index_put requires separate collision semantics')
        if rec.get('category') != 'B_switch_expand':
            raise ValueError('scatter mapping currently validated for B-switch diagonal contract')
        dest = shape(args[0]); idx = args[1]
        axes = tuple(j for j,t in enumerate(idx) if t is not None)
        ids = [idx[j] for j in axes]
        if not ids or len({t.get('alias') for t in ids}) != 1:
            raise ValueError('scatter expects one repeated diagonal index tensor')
        ni = shape(ids[0])
        if len(ni)!=1 or any(d!=ni[0] for d in dest):
            raise ValueError('scatter expects equal-width axes and one arange(n) index')
        if len(dest)-len(axes)>1:
            raise ValueError('scatter has more than one free axis')
        strides=tuple(math.prod(dest[j+1:]) for j in range(len(dest)))
        index_dtype=ids[0]['dtype']
        index_rate=dtype_rate(index_dtype)
        arithmetic=pw('multiply',ni) if sum(strides[j]for j in axes)!=1 else 0
        free=[j for j in range(len(dest)) if j not in axes]
        selected=ni if not free else (ni[0],ni[0])
        if free:
            if strides[free[0]]!=1:
                arithmetic+=pw('multiply',ni)
            arithmetic+=pw('add',selected)
        # Cost already includes separate integer-coordinate and value rates.
        cost = pw('put',selected)+arithmetic*index_rate/rate
        why = 'B-switch i=arange(n) diagonal scatter via public fnp.put; flat-index int64 arithmetic included'
    elif name in ('where',):
        cost = pw('where', out);why = 'public selection price per output element'
    elif name == 'erf':
        cost = erf_unit_cost(billed)*numel(out)/rate
        why = 'public erf port 2*stats.norm.cdf(sqrt(2)*x)-1, cast back; dtype promotion included'
    elif name == 'rsqrt':
        cost = pw('sqrt',out)+pw('reciprocal',out);why='literal sqrt then reciprocal'
    else:
        raise ValueError('operator has no checked mapping')
    return int(cost*rate), why


def cost_trace(data):
    records = data if isinstance(data, list) else data['records']
    known, unknown = [], []
    phases, categories = defaultdict(int), defaultdict(int)
    for rec in records:
        try:
            per, why = record_cost(rec)
        except Exception as exc:
            unknown.append(dict(record=rec, reason=str(exc)))
            continue
        total = per*int(rec.get('calls', 1))
        phases[str(rec.get('phase', 'unspecified'))] += total
        categories[str(rec.get('category', 'unspecified'))] += total
        known.append(dict(**rec, modeled_flops_per_call=per, modeled_flops=total, mapping=why))
    subtotal = sum(x['modeled_flops'] for x in known)
    return dict(schema='flopscope-trace-model-v1', mode='modeled_not_BudgetContext',
                flopscope_version=flops.__version__, complete_mapping=not unknown,
                complete_modeled_flops=None if unknown else subtotal,
                complete_modeled_units=None if unknown else subtotal/UNIT,
                known_subtotal_flops=subtotal, known_subtotal_units=subtotal/UNIT,
                total_calls=sum(int(x.get('calls', 1)) for x in records),
                mapped_calls=sum(int(x.get('calls', 1)) for x in known),
                by_phase_flops=dict(phases), by_category_flops=dict(categories),
                allclose_check_flops=sum(x['modeled_flops'] for x in known if x['op']=='aten.allclose.default'),
                unknown_ops=unknown, records=known,
                limitations=['literal NumPy port cost model; no symmetry annotations',
                             'GPU wall/RSS come from accuracy run; no scoring residual measured',
                             'B prefix cost is separate unless present in this trace',
                             'actual reference allclose checks are included and separately subtotaled; other research diagnostics follow traced phases',
                             'copy and reshape are charged; metadata-only transforms explicitly mapped'])


def check(path):
    rng = np.random.default_rng(284)
    x = fnp.array(rng.normal(size=(32, 32)).astype('float32'))
    y = fnp.array(rng.normal(size=(32, 32)).astype('float32'))
    sym = fnp.array(np.asarray(y)+np.asarray(y).T)
    vector=fnp.array(rng.normal(size=32).astype('float32'))
    boolean=fnp.array(np.arange(32)%2==0)
    cube=fnp.array(rng.normal(size=(32,32,32)).astype('float32'))
    index=fnp.array(np.arange(32,dtype='int64'))
    cases = [
        ('matmul', lambda: x@y, matmul_cost((32,32),(32,32))),
        ('add', lambda: fnp.add(x,y), pw('add',(32,32))),
        ('multiply', lambda: fnp.multiply(x,y), pw('multiply',(32,32))),
        ('power', lambda: x**2, pw('power',(32,32))),
        ('sum_axis1', lambda: fnp.sum(x,axis=1), accounting.reduction_cost('sum',input_shape=(32,32),axis=1)),
        ('mean_axis1', lambda: fnp.mean(x,axis=1), accounting.reduction_cost('mean',input_shape=(32,32),axis=1)+pw('divide',(32,))),
        ('copy', lambda: x.copy(), pw('copy',(32,32))),
        ('reshape', lambda: x.reshape(1024), pw('reshape',(1024,))),
        ('qr', lambda: fnp.linalg.qr(y,mode='reduced'), accounting.qr_cost(32,32)),
        ('eigh', lambda: fnp.linalg.eigh(sym), accounting.eigh_cost(32)),
        ('norm', lambda: fnp.linalg.norm(x), accounting.norm_cost((32,32))),
        ('erf_public_port', lambda: erf_port(x), int(erf_unit_cost('float32')*1024)),
        ('allclose',lambda:fnp.allclose(x,y),6*1024+accounting.reduction_cost('all',input_shape=(32,32))),
        ('diag_embed',lambda:fnp.diag(vector),pw('diag',(32,))),
        ('diagonal_copy',lambda:fnp.diagonal(x).copy(),pw('copy',(32,))),
        ('eye',lambda:fnp.eye(32,dtype=fnp.float32),pw('eye',(32,))),
        ('arange_int64',lambda:fnp.arange(32,dtype=fnp.int64),int(2*pw('arange',(32,))*dtype_rate('int64'))),
        ('flip',lambda:fnp.flip(vector),0),
        ('unbind',lambda:tuple(vector[i] for i in range(32)),0),
        ('scalar_item',lambda:vector[0].item(),0),
        ('any_bool',lambda:fnp.any(boolean),accounting.reduction_cost('any',input_shape=(32,))),
        ('bitwise_and_bool',lambda:fnp.bitwise_and(boolean,boolean),pw('bitwise_and',(32,))),
        ('advanced_index_iii',lambda:cube[index,index,index],4*pw('getitem',(32,))),
        ('advanced_index_iik',lambda:cube[index,index,:],4*pw('getitem',(32,32))),
        ('put_diag_matrix',lambda:diagonal_put_port(x,index,0,(0,1)),3*32),
        ('put_diag_cube_iii',lambda:diagonal_put_port(cube,index,vector,(0,1,2)),3*32),
        ('put_diag_cube_iik',lambda:diagonal_put_port(cube,index,y,(0,1)),2*32+3*32**2),
        ('put_diag_cube_iki',lambda:diagonal_put_port(cube,index,y,(0,2)),4*32+3*32**2),
        ('put_diag_cube_kii',lambda:diagonal_put_port(cube,index,y,(1,2)),4*32+3*32**2),
    ]
    for degree,operations in ((0,1),(1,0),(2,2),(3,3),(4,8)):
        cases.append((f'hermite_{degree}',lambda d=degree:hermite_port(vector,d),operations*32))
    rows=[]
    for name, fn, expected in cases:
        ctx=flops.BudgetContext(flop_budget=2**41,quiet=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with ctx:
                fn()
        rows.append(dict(op=name, modeled_flops=expected, actual_BudgetContext_flops=ctx.flops_used,
                         equal=expected==ctx.flops_used))
    oracle=accounting.einsum_cost('ik,kj->ij',[(32,32),(32,32)])
    scatter_checks=[]
    for dims,axes in ((2,(0,1)),(3,(0,1,2)),(3,(0,1)),(3,(0,2)),(3,(1,2))):
        raw=rng.normal(size=(32,)*dims).astype('float32')
        tensor=fnp.array(raw.copy())
        key=tuple(np.arange(32) if j in axes else slice(None) for j in range(dims))
        value=rng.normal(size=raw[key].shape).astype('float32')
        wrapped=fnp.array(value)
        raw[key]=value
        ctx=flops.BudgetContext(flop_budget=2**41,quiet=True)
        with ctx:
            diagonal_put_port(tensor,index,wrapped,axes)
        equal=np.array_equal(np.asarray(tensor),raw)
        scatter_checks.append(dict(dims=dims,indexed_axes=axes,equal=equal,actual_flops=ctx.flops_used))
    result=dict(kind='real_small_BudgetContext_checks',width=32,dtype='float32',
                python=platform.python_version(),flopscope_version=flops.__version__,
                matmul_public_einsum_cost=oracle,matmul_specialization_matches=oracle==cases[0][2],
                dtype_rate_float32=dtype_rate('float32'),dtype_rate_float64=dtype_rate('float64'),
                rows=rows,scatter_numeric_checks=scatter_checks,
                all_equal=all(r['equal'] for r in rows) and oracle==cases[0][2]
                          and all(r['equal']for r in scatter_checks))
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['all_equal']:
        raise SystemExit('cost-model check failed; inspect receipt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path,nargs='?')
    p.add_argument('--output',type=Path)
    p.add_argument('--check',action='store_true')
    a=p.parse_args()
    if a.check:
        check(Path(__file__).with_name('trace_cost_check.json'))
    if a.trace:
        result=cost_trace(json.loads(a.trace.read_text()))
        target=a.output or a.trace.with_name('trace_cost.json')
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','unknown_ops')},indent=2))
        print(f"unknown records: {len(result['unknown_ops'])}; receipt: {target}")
    if not a.trace and not a.check:
        p.error('supply trace or --check')


if __name__=='__main__':
    main()
