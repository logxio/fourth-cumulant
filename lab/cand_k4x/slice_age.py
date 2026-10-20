"""Lifetime policy for research K4 slice groups (no tensor dependencies)."""


class SliceAges:
    def __init__(self, start, limit, groups=()):
        if limit < 0:
            raise ValueError('slice age limit must be nonnegative (0 = unlimited)')
        self.limit = limit
        # B's preactivation seed is present at the switch, age zero.
        self.births = {id(g.Bb): start for g in groups}

    def register(self, layer, groups):
        # Remove dead identities so a reused Python id never inherits an age.
        self.births = {id(g.Bb): self.births.get(id(g.Bb), layer) for g in groups}

    def prune(self, layer, k4):
        removed = []
        kept = []
        for g in k4.groups:
            birth = self.births[id(g.Bb)]
            if self.limit and g.types[0].startswith('slice') and layer-birth > self.limit:
                removed.append({'birth': birth, 'age': layer-birth, 'types': list(g.types)})
            else:
                kept.append(g)
        if removed:
            k4.groups = kept
            # Cached slices contain the expired groups too.
            k4.clear_repeated()
            self.register(layer, kept)
        return removed
