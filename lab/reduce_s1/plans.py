"""Plans for select_run.py: full schedules, and states kept by read weight under a unit budget.

Read-weight rule ("a state is worth carrying while later layers still read it"): every
(state, read) pair of the instrumented full run carries a read weight, the norm of that read's
first-order effect on the final mean. Pairs are taken in descending weight; taking a pair keeps
its state alive from birth through that read (its earlier reads come with it) and is skipped when
that closure does not fit the remaining units. 'state' and 'stateunit' keep whole states at full
life, ranked by their summed read weight, alone or per counted unit. 'early' uses no read weight: whole
tree, non-tree and star states in birth order, no C.k3 births (the late-start ablation order). The B seed and the tree slices born at s0 travel
together as one source (Table 7a) and are always kept. 'reverse' takes pairs in ascending weight,
a control at the same units.

usage: plans.py full <s0> <out.json> [--last-birth 14]
       plans.py window <s0> <last birth> <with C.k3 0|1> <out.json> [--tree-all]   (every family of births s0..e at full life)
       plans.py select <instrumented.json> <rule> <budget> <out.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger


def full(s0, last_birth=13):
    life = {fam: {str(b): 14 - b for b in range(s0, last_birth + 1)} for fam in ('star', 'ck3', 'tree', 'nontree')}
    return dict(name=f'full_s{s0}' + ('' if last_birth == 13 else f'_b{last_birth}'), start=s0,
                last_birth=last_birth, life=life)


def window(s0, last, ck3, tree_all=False):
    fams = ('star', 'tree', 'nontree') + (('ck3',) if ck3 else ())
    life = {fam: ({str(b): 14 - b for b in range(s0, last + 1)} if fam in fams else {})
            for fam in ('star', 'ck3', 'tree', 'nontree')}
    if tree_all:  # tree slices (33 units a read) born at every layer, the gain families only in the window
        life['tree'] = {str(b): 14 - b for b in range(s0, 14)}
    return dict(name=f"window_s{s0}_b{last}" + ('_ck3' if ck3 else '') + ('_treeall' if tree_all else ''),
                start=s0, last_birth=13, life=life)


def select(instrumented, rule, budget):
    r = json.loads(Path(instrumented).read_text())
    s0 = r['start']
    weight = {}
    for x in r['reads']:
        fam, b, l = x['family'], x['birth'], x['layer']
        if fam == 'seed' or b > 13 or l > ledger.LAST_READ or l <= b:
            continue
        weight[(fam, b, l)] = x['w']
    life = {fam: {} for fam in ('star', 'ck3', 'tree', 'nontree')}
    life['tree'][s0] = 14 - s0  # seed + layer-s0 slice births, one source in Table 7a

    def cost(lf):
        return ledger.units({k: {str(b): a for b, a in v.items()} for k, v in lf.items()}, s0)[0]

    assert rule in ('weight', 'reverse', 'state', 'stateunit', 'early')
    taken = []
    base = cost(life)
    assert base <= budget, (base, budget)
    if rule.startswith('state'):
        # whole states at full life: total read weight, alone or per counted unit
        total_w = {}
        for (fam, b, l), w in weight.items():
            total_w[(fam, b)] = total_w.get((fam, b), 0) + w
        def marginal(key):
            trial = {k: dict(v) for k, v in life.items()}
            trial[key[0]][key[1]] = 14 - key[1]
            return cost(trial) - cost(life)
        pool = [k for k in total_w if life[k[0]].get(k[1], 0) < 14 - k[1]]
        while pool:
            score = {k: total_w[k] / (marginal(k) if rule == 'stateunit' else 1) for k in pool if marginal(k) > 0}
            fits = [k for k in sorted(score, key=score.get, reverse=True) if cost(life) + marginal(k) <= budget]
            if not fits:
                break
            k = fits[0]
            life[k[0]][k[1]] = 14 - k[1]
            taken.append([k[0], k[1], 14, total_w[k]])
            pool.remove(k)
        order = []
    elif rule == 'early':
        # from the measured late-start ablations, no read weights: whole tree, non-tree and star states at
        # full life in birth order, C.k3 births left out (they cost 27,390 of the 60,549)
        for b in range(s0, 14):
            for fam in ('tree', 'nontree', 'star'):
                if life[fam].get(b, 0) >= 14 - b:
                    continue
                trial = {k: dict(v) for k, v in life.items()}
                trial[fam][b] = 14 - b
                if cost(trial) <= budget:
                    life = trial
                    taken.append([fam, b, 14, None])
        order = []
    else:
        order = sorted(weight, key=lambda k: weight[k], reverse=(rule == 'weight'))
    for fam, b, l in order:
        age = l - b
        if life[fam].get(b, 0) >= age:
            continue
        trial = {k: dict(v) for k, v in life.items()}
        trial[fam][b] = age
        if cost(trial) <= budget:
            life = trial
            taken.append([fam, b, l, weight[(fam, b, l)]])
    total, rows = ledger.units({k: {str(b): a for b, a in v.items()} for k, v in life.items()}, s0)
    kept = sum(weight.get((fam, b, l), 0) for fam, births in life.items() for b, a in births.items()
               for l in range(b + 1, b + a + 1) if (fam, b, l) in weight)
    return dict(name=f'{rule}_{int(budget)}', start=s0, last_birth=13, rule=rule, budget=budget,
                units=total, unit_rows=rows, read_weight_kept=kept, read_weight_all=sum(weight.values()),
                taken=taken, life={k: {str(b): a for b, a in sorted(v.items())} for k, v in life.items()})


def main():
    if sys.argv[1] == 'window':
        plan = window(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), '--tree-all' in sys.argv)
        out = sys.argv[5]
    elif sys.argv[1] == 'full':
        lb = int(sys.argv[sys.argv.index('--last-birth') + 1]) if '--last-birth' in sys.argv else 13
        plan = full(int(sys.argv[2]), lb)
        out = sys.argv[3]
    else:
        plan = select(sys.argv[2], sys.argv[3], float(sys.argv[4]))
        out = sys.argv[5]
    Path(out).write_text(json.dumps(plan, indent=2) + '\n')
    print(out, plan['name'], round(ledger.units(plan['life'], plan['start'])[0]),
          {k: len(v) for k, v in plan['life'].items()})


if __name__ == '__main__':
    main()
