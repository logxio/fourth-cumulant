"""Keep the previously validated graph-factor trees at K4 birth.

The B switch seed is installed before these hooks. The seed retains its
specified radial feed. All subsequent slice211 births are replaced by the
tree sum; dense k3/k4 hyperedges, triangles and inherited-core subtractions
are excluded. Other repeated slices and optional star groups are unchanged.
This module is a GPU accuracy probe, not a flopscope estimator.
"""
from collections import Counter
from k4n3_young_direct_chain import Recorder, tree_tensor, Products, tree_parts


class TreeBirth(Recorder):
    def __init__(self, non_tree_rank=0):
        super().__init__(light=True, birth_mode='tree_only')
        self.non_tree_rank = non_tree_rank
        self.last = None

    def split_birth(self, ds):
        total = ds.slices[(2, 1, 1)]
        tree = tree_tensor(self.current.trees, ds.n, total.device, total.dtype)
        prod = Products()
        # Count the validated compact read schedule at identity. This costs
        # n^2 storage and supplies an exact shape count, not a whole-net bill.
        import torch
        q = torch.eye(ds.n, device=total.device, dtype=total.dtype)
        tree_parts(q, self.current.trees, ds.slices[(2, 2)], prod, None)
        self.last = dict(classes=dict(Counter(t.cls for t in self.current.trees)),
                         compact_tree_products=dict(prod.count),
                         total_norm=float(total.norm()), tree_norm=float(tree.norm()))
        if self.non_tree_rank:
            # A fixed network-derived covariance Krylov subspace. No truth is
            # used. Forming the residual densely belongs only to this probe;
            # a scoring implementation must contract its graph factors.
            from k4n3_tucker_age import mode_product, reconstruct
            cov = self.K[2].core
            # Fixed sketch reused for all calls with the same width/rank.
            rng = torch.Generator(device=total.device).manual_seed(4808)
            omega = torch.randn((ds.n, self.non_tree_rank), generator=rng,
                                device=total.device, dtype=total.dtype)
            q, _ = torch.linalg.qr(cov @ omega, mode='reduced')
            q, _ = torch.linalg.qr(cov @ q, mode='reduced')
            residual = total - tree
            core = residual
            for axis in range(3):
                core = mode_product(core, q, axis)
            for axis in range(3):
                core = reconstruct(core, q, axis)
            self.last['non_tree_rank'] = self.non_tree_rank
            self.last['non_tree_norm'] = float(residual.norm())
            self.last['non_tree_retained_norm'] = float(core.norm())
            from mlp_kprop.diagslice import _zero_repeated
            tree.add_(core)
            _zero_repeated(tree)
        ds.slices[(2, 1, 1)] = tree
