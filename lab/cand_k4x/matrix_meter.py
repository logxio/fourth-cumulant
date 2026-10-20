"""Matrix arithmetic lower bound for the GPU reference, using actual shapes.

The unit is 2*1024**3. Elementwise work, decompositions and data movement are
listed as excluded, so this is never presented as a flopscope scoring bill.
"""
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode


class MatrixMeter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.phase = 'setup'
        self.totals = defaultdict(float)
        self.calls = defaultdict(int)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        cost = None
        if name == 'aten.mm.default':
            a,b = args[:2]
            m,k = a.shape; _,n = b.shape
            cost = m*n*(2*k-1)
        elif name == 'aten.bmm.default':
            a,b = args[:2]
            batch,m,k = a.shape; n = b.shape[-1]
            cost = batch*m*n*(2*k-1)
        elif name in ('aten.addmm.default','aten.baddbmm.default'):
            a,b = args[1:3]
            m,k = a.shape[-2:]; n = b.shape[-1]
            batch = a.numel()//(m*k)
            cost = batch*m*n*2*k
        elif name == 'aten.mv.default':
            m,k = args[0].shape
            cost = m*(2*k-1)
        if cost is not None:
            key = self.phase+'/'+name
            self.totals[key] += cost/(2*1024**3)
            self.calls[key] += 1
        return func(*args, **(kwargs or {}))

    def result(self):
        return dict(matrix_units=sum(self.totals.values()),
                    by_phase_op=dict(self.totals),calls=dict(self.calls),
                    excluded=['elementwise','decompositions','copies','reductions'],
                    meaning='actual reference matrix arithmetic lower bound; not a flopscope bill')
