"""reduce_s1 readout: bottleneck reduction table and accuracy-units curve, every number with its source.

usage: python3.13 lab/reduce_s1/make_readout.py recovered_run_root
writes runs/cloud/reduce_s1/tables.md (the two tables) and runs/cloud/reduce_s1/curve.csv (one row per point).
Raw MSEs from this batch are recomputed from the saved predictions against data/mini_truth.npz.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger

OUT = ROOT / 'runs/cloud/reduce_s1'
if len(sys.argv) != 2:
    raise SystemExit("usage: python3 lab/reduce_s1/make_readout.py recovered_run_root")
BK = Path(sys.argv[1])
TRUTH = np.load(ROOT / 'data/mini_truth.npz')
N = 1024


def rel(p):
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def raw_of(npy, net):
    p = np.load(npy).astype(float)
    return float(np.mean((p[-1] - TRUTH['all_layer_means'][net, -1]) ** 2))


def adj(raw, units):
    return raw * max(0.1, units / N) if units <= N else float('nan')


def numbers():
    rows = {}
    with open(ROOT / 'data/numbers.csv') as fh:
        for r in csv.reader(fh):
            if len(r) >= 5 and r[0] != 'key':
                try:
                    rows[r[0]] = (float(r[1]), r[4])
                except ValueError:
                    pass
    return rows


NUM = numbers()


def num(key):
    v, src = NUM[key]
    return v, f'data/numbers.csv:{key} ({src})'


def batch(net, name):
    j = BK / f'outputs/net{net}/{name}.json'
    if not j.exists():
        return None
    r = json.loads(j.read_text())
    if 'plan' in r:
        raw = raw_of(j.with_suffix('.npy'), net)
        assert math.isclose(raw, r['mse'], rel_tol=1e-9), (j, raw, r['mse'])
        return dict(raw=raw, units=r['units'], rows=r['unit_rows'], src=rel(j), plan=r['plan'], result=r)
    raw = raw_of(j.with_suffix('.npy'), net)
    assert math.isclose(raw, r['raw'], rel_tol=1e-9), (j, raw, r['raw'])
    return dict(raw=raw, units=r['paper_U'], src=rel(j), result=r)


def b_alone(net):
    j = BK / f'snapshots/B_mini{net}.json'
    r = json.loads(j.read_text())
    raw = raw_of(j.with_suffix('.npy'), net)
    return dict(raw=raw, units=r['units'], src=rel(j))


def fmt(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ''
    if isinstance(x, float) and (abs(x) < 1e-3 or abs(x) >= 1e5):
        return f'{x:.{d}e}'.replace('e-0', 'e-')
    return f'{x:,.{d}f}' if isinstance(x, float) else f'{x:,}'


def read_weight_shares(net, tag='full_s4_b14'):
    j = BK / f'outputs/net{net}/{tag}.json'
    if not j.exists():
        return None
    r = json.loads(j.read_text())
    tot, counted = {}, {}
    for x in r['reads']:
        fam = x['family']
        tot[fam] = tot.get(fam, 0) + x['w']
        if fam != 'seed' and x['birth'] <= 13 and x['birth'] < x['layer'] <= 14:
            counted[fam] = counted.get(fam, 0) + x['w']
    return dict(all=tot, counted=counted, src=rel(j), raw=r['mse'])


def main():
    lines = []
    out_rows = []
    B0 = b_alone(0)
    adjB0 = adj(B0['raw'], B0['units'])
    # ---------- Table A: reduction of the s0 = 4 column ----------
    total4, rows4 = ledger.units(ledger.full_life(4), 4)
    rw = read_weight_shares(0)
    rw2 = read_weight_shares(0, 'full_s4_b14_v2')
    rw12 = read_weight_shares(1, 'full_s4_b14_v2')
    late = {k: num(k) for k in ('late_B_alone_net0', 'late_start4_net0', 'late_start4_noCk3_star_net0',
                                'late_start4_noCk3_slices_net0', 'late4_tree_star0_age0')}
    lv = {k: v[0] for k, v in late.items()}
    fi = {k: num(k)[0] for k in ('abl01_full', 'abl01_no_k3c', 'abl01_no_path', 'abl01_no_slicex', 'abl01_portable')}
    fz = {(i, a): num(f'fz1024_mini{i}_{a}')[0] for i in (5, 6, 7) for a in ('full', 'no_k3c', 'no_path', 'no_slicex')}
    nostar = (num('c45_1024_nostar_min')[0], num('c45_1024_nostar_max')[0])
    nontree_c = num('drop_C_kappa3_nontree_net0')[0] / num('portable_net0')[0]

    def share(fam, src):
        if src is None or fam is None:
            return ''
        t = sum(src['counted'].values())
        return f"{100 * src['counted'].get(fam, 0) / t:.1f}%"

    full4 = batch(0, 'full_s4_b14')
    nt_kept = batch(0, 'n0_weight_33611')
    ts = batch(0, 'n0_early_3010')
    ck3_u = rows4['C.k3 births and crossing']
    nt_u = rows4['non-tree first reads']
    tree_u = rows4['tree slice reads'] + rows4['feedback']
    star_u = rows4['K=4 sources'] + rows4['young-leg transport']
    fam_rows = [
        ('C·κ3 births (real κ3 width + rank-16 crossing)', ck3_u, 'ck3',
         f"{lv['late_start4_noCk3_star_net0'] / lv['late_start4_net0']:.2f}× (without them 1.129e-8 against 5.418e-9)",
         f"{fi['abl01_no_k3c'] / fi['abl01_full']:.2f}× nets 0–1; " + ', '.join(f"{fz[(i, 'no_k3c')] / fz[(i, 'full')]:.2f}×" for i in (5, 6, 7)) + ' frozen 5–7'),
        ('non-tree births ((2,1,1) graph terms with no shared-edge factorization)', nt_u, 'nontree',
         (f"{nt_kept['raw'] / full4['raw']:.2f}× with C·κ3 and star kept ({nt_kept['raw']:.3e} against {full4['raw']:.3e}, this batch; non-tree kept only at layer 13); " if nt_kept and full4 else '') +
         f"{lv['late4_tree_star0_age0'] / lv['late_start4_noCk3_slices_net0']:.2f}× with C·κ3 and star off (1.245e-8 against tree-only 2.415e-8)",
         f"{nontree_c:.2f}× for the one class C_jk κ3_ijk (portable set, net 0)"),
        ('tree slices + (2,1,1)→K3 feedback (incl. chain B seed)', tree_u, 'tree',
         f"{lv['late_B_alone_net0'] / lv['late4_tree_star0_age0']:.2f}× against chain B alone (tree slices alone 2.415e-8, chain B 2.262e-8)" +
         (f"; with star births too {ts['raw'] / B0['raw']:.2f}× worse than chain B ({ts['raw']:.3e} at {ts['units']:,.0f} units, this batch)" if ts else ''),
         f"crossings in them {fi['abl01_no_slicex'] / fi['abl01_full']:.2f}× nets 0–1; " + ', '.join(f"{fz[(i, 'no_slicex')] / fz[(i, 'full')]:.2f}×" for i in (5, 6, 7)) + ' frozen'),
        ('star births + K=4 source machinery + young legs', star_u, 'star',
         f"{lv['late_start4_noCk3_slices_net0'] / lv['late_start4_noCk3_star_net0']:.2f}× (star + slices 1.129e-8, slices 1.245e-8)",
         f"{nostar[0]:.2f}–{nostar[1]:.2f}× frozen 5–7"),
        ('chain B, metered (prefix and K=3 state)', rows4['chain B (metered)'], None, 'the base: 2.262e-8', ''),
    ]
    lines.append('**Table R1.** The s₀ = 4 column of Table 7a (60,549 units, official network 0) split by state family. '
                 'Units: the Table 7a ledger, row for row. Read-weight share: of all counted (state, read) pairs of the '
                 'instrumented full run, the share of summed read weight (first-order effect of the read on the final mean). '
                 'Gain carried: final MSE without the family ÷ with it, measured.')
    lines.append('')
    lines.append('| state family | units | share of 60,549 | read weight v1 (mean, variance), net 0 | v2 (+ κ3), net 0 | v2, net 1 | gain carried, start at layer 4 (net 0) | gain carried, full K=4 from the input (1024) |')
    lines.append('|:--|--:|--:|--:|--:|--:|:--|:--|')
    for label, u, fam, g_late, g_in in fam_rows:
        lines.append(f"| {label} | {u:,.0f} | {100 * u / total4:.1f}% | {share(fam, rw)} | {share(fam, rw2)} | {share(fam, rw12)} | {g_late} | {g_in} |")
    lines.append(f"| **total** | **{total4:,.0f}** | 100% | | | | full chain from layer 4: 5.418e-9, {lv['late_B_alone_net0'] / lv['late_start4_net0']:.2f}× below chain B | full K=4: {fi['abl01_full']:.3e} nets 0–1 |")
    lines.append('')
    srcA = ['ledger: lab/reduce_s1/ledger.py (rows from runs/cloud/k4s5/cost.json; reproduces Table 7a 60,549 / 47,287 / 33,611)']
    srcA += [f'{k}: {v[1]}' for k, v in late.items()]
    srcA += [f'{k}: {num(k)[1]}' for k in ('abl01_full', 'abl01_no_k3c', 'abl01_no_path', 'abl01_no_slicex',
                                          'drop_C_kappa3_nontree_net0', 'portable_net0', 'c45_1024_nostar_min')]
    srcA += [f'fz1024_mini5..7_*: ' + num('fz1024_mini5_full')[1]]
    if rw:
        srcA.append('read weights: ' + ', '.join(x['src'] for x in (rw, rw2, rw12) if x))
    if nt_kept:
        srcA.append(f"non-tree off, C·κ3 kept: {nt_kept['src']}; full from layer 4 split: {full4['src']}")
    lines.append('Sources: ' + '; '.join(srcA))
    lines.append('')
    # ---------- the (3,1) arm ----------
    k31 = json.loads((ROOT / 'runs/kaggle/reduce_s1/output-v1/collected.json').read_text())
    lines.append('**Table R2.** Chain B (#335301, no corrector) with its (3,1) slice of κ4 replaced by the exact one at layers 1–14 '
                 '(public 10⁸-sample pair moments), against full K=4 on the same network. Share of the gain: '
                 'ln(B ÷ B+exact(3,1)) ÷ ln(B ÷ full K=4).')
    lines.append('')
    lines.append('| network | chain B | B + exact (3,1) | ratio | full K=4 | full K=4 ÷ B | share of the K=4 gain (log) | share of the error removed |')
    lines.append('|:--|--:|--:|--:|--:|--:|--:|--:|')
    for x in k31['rows']:
        i = x['net']
        k4 = NUM.get(f'k4_full_1024_net{i}', (None,))[0]
        if k4:
            sh = math.log(x['B'] / x['B_k31']) / math.log(x['B'] / k4)
            er = (x['B'] - x['B_k31']) / (x['B'] - k4)
            lines.append(f"| {i} | {x['B']:.4e} | {x['B_k31']:.4e} | {x['ratio']:.3f} | {k4:.4e} | {k4 / x['B']:.4f} | {100 * sh:.1f}% | {100 * er:.1f}% |")
        else:
            lines.append(f"| {i} | {x['B']:.4e} | {x['B_k31']:.4e} | {x['ratio']:.3f} | | | | |")
    lines.append(f"| nets 0, 1, 3–8 (sum) | | | **{k31['ratio_sum']:.3f}**, {k31['wins']}/8 lower | | | | |")
    lines.append('')
    lines.append('Sources: runs/kaggle/reduce_s1/output-v1/collected.json (nets 0–1: kernel loxigicck/arc-reduce-s1-k31 v1; nets 3–8: '
                 'runs/kaggle/k31fit/output-v3), every raw recomputed from the saved *_prelt_*.npy against data/mini_truth.npz; '
                 'full K=4: report/data/numbers.csv k4_full_1024_net0/1 (runs/cloud/k4w/k4w.log). Forum 18246: −18.1% inside V29 on 23 networks.')
    lines.append('')
    # ---------- Table C: curve ----------
    pts = []

    def add(group, label, net, units, raw, src, note='', base=None):
        pts.append(dict(group=group, label=label, net=net, units=units, raw=raw, src=src, note=note,
                        adj=adj(raw, units), ratio_B=raw / (base or (B0['raw'] if net == 0 else b_alone(1)['raw']))))

    for net in (0, 1):
        Bn = b_alone(net)
        add('chain B', 'chain B (#335301 chain, this box)', net, Bn['units'], Bn['raw'], Bn['src'])
    v, s = num('lt2_local_mini0_raw')
    add('chain B + learned corrector', 's24 #336537', 0, 0.1682 * N, v, s + '; C/B 0.1682 report/data/s24_mini0_local.log')
    # in-budget K=4 pieces fed into chain B (K=4 feedback comparison, its own machine and chain-B control)
    s9 = json.loads((ROOT / 'runs/cloud/k4s9/r2/summary.json').read_text())['mini0']
    s9B = [r for r in s9 if r['case'] == 'base:1:all'][0]['raw']
    for r in s9:
        if r['case'] in ('star:1:all',):
            add('B + K=4 pieces (prior run)', f"{r['case']} (prior run; raw × this-box B ÷ its B)", 0, r['paper_U'],
                r['raw'] / s9B * B0['raw'], 'runs/cloud/k4s9/r2/summary.json', f"its own ratio {r['raw'] / s9B:.4f}")
    for net in (0, 1):
        Bn = B0 if net == 0 else b_alone(1)
        base = batch(net, 'fed_base_1_all')
        for f in sorted((BK / f'outputs/net{net}').glob('fed_*.json')):
            if f.stem == 'fed_base_1_all':
                continue
            x = batch(net, f.stem)
            ctl = f" (same-box chain B {base['raw']:.4e})" if base else ''
            add('B + K=4 pieces fed back (this box)', x['result']['case'], net, x['units'], x['raw'], x['src'],
                f"same-box chain B {base['raw']:.4e}" if base else '', base=base['raw'] if base else None)
        for f in sorted((BK / f'outputs/net{net}').glob('*.json')):
            if f.stem.startswith('fed_') or f.stem.startswith('smoke'):
                continue
            x = batch(net, f.stem)
            stem = f.stem
            grp = ('K=4 suffix, reads by v2 weight (mean, variance, κ3)' if '_w2_weight' in stem else
                   'K=4 suffix, v2 reverse order (control)' if '_w2_reverse' in stem else
                   'K=4 suffix, reads by v1 weight (mean, variance)' if '_weight' in stem else
                   'K=4 suffix, v1 reverse order (control)' if '_reverse' in stem else
                   'K=4 suffix, whole states by v2 weight per unit' if '_stateunit' in stem else
                   'K=4 suffix, whole states by v2 weight' if '_state' in stem else
                   'K=4 suffix, whole states in birth order, no C·κ3' if '_early' in stem else
                   ('K=4 suffix, every family of births 4..e, with C·κ3' if stem.endswith('_ck3') else 'K=4 suffix, every family of births 4..e, no C·κ3') if stem.startswith('window') else
                   'K=4 suffix, every state from layer s₀')
            note = ''
            if 'plan' in x and x['plan'].get('life'):
                life = x['plan']['life']
                note = 'states kept: ' + ', '.join(f"{k} {len(v)}" for k, v in life.items() if v)
            add(grp, f.stem, net, x['units'], x['raw'], x['src'], note)
    # Table 7a columns and full K=4 (existing)
    for s0, key in ((4, 'late_start4_net0'), (6, 'late_start6_net0'), (8, 'late_start8_net0')):
        v, s = num(key)
        add('K=4 suffix, full from layer s₀ (Table 7a)', f'full from layer {s0}, old pieces exact', 0,
            ledger.units(ledger.full_life(s0), s0)[0], v, s)
    v, s = num('exec_s4_raw')
    add('K=4 suffix, full from layer s₀ (Table 7a)', 'full from layer 4, compressed as counted', 0, 60549, v, s)
    v, s = num('portable_net0')
    add('K=4 from the input (Table 7a)', 'portable set from the input, exact', 0, 77930, v, s)
    v, s = num('young_m0f0_k128_drop_triangle')
    add('K=4 from the input (Table 7a)', 'portable set from the input, compressed as counted', 0, 77930, v, s)
    for net in (0, 1):
        v, s = num(f'k4_full_1024_net{net}')
        add('full K=4 (Table 1)', 'full K=4 from the input (adds path births and slice crossings, units ≥ 77,930)', net, 77930, v, s)
    # existing late-start states with capped lives (k4s8), counted by the same ledger
    k8 = {r['name']: r['mse'] for r in json.loads((ROOT / 'runs/cloud/k4s8/verified.json').read_text())['rows']}
    for star in (0, 1):
        for age in (1, 2, 3, 0):
            name = f'tree_star{star}_age{age}'
            if name not in k8:
                continue
            life = {'tree': {b: (age if age else 14 - b) for b in range(4, 14)}}
            if star:
                life['star'] = {b: 14 - b for b in range(4, 14)}
            add('K=4 suffix, tree slices only (k4s8)', name, 0, ledger.units(life, 4)[0], k8[name],
                'runs/cloud/k4s8/verified.json', 'non-tree and C·κ3 off; slice life ' + (str(age) if age else 'full'))
    k6 = {r['name']: r['mse'] for r in json.loads((ROOT / 'runs/cloud/k4s6/verified.json').read_text())['rows']}
    sl = {'tree': {b: 14 - b for b in range(4, 14)}, 'nontree': {b: 14 - b for b in range(4, 14)}}
    add('K=4 suffix, families off (k4s6)', 'tree + non-tree slices, no star, no C·κ3', 0, ledger.units(sl, 4)[0], k6['slices'],
        'runs/cloud/k4s6/verified.json')
    sl['star'] = {b: 14 - b for b in range(4, 14)}
    add('K=4 suffix, families off (k4s6)', 'star + tree + non-tree slices, no C·κ3', 0, ledger.units(sl, 4)[0], k6['star'],
        'runs/cloud/k4s6/verified.json')
    pts.sort(key=lambda p: (p['net'], p['units']))
    for net in (0, 1):
        best = float('inf')
        for p in [q for q in pts if q['net'] == net]:
            p['frontier'] = 'corrector' not in p['group'] and p['raw'] < best
            if p['frontier']:
                best = p['raw']
    with open(OUT / 'curve.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(pts[0]))
        w.writeheader()
        w.writerows(pts)
    KEY = {'s24 #336537', 'n0_weight_602', 'n0_weight_1204', 'n0_weight_3010', 'n0_weight_6000', 'n0_w2_weight_6000',
           'n1_w2_weight_6000', 'n0_weight_20453', 'n0_w2_weight_20453', 'n1_w2_weight_20453', 'tree_star0_age1',
           'tree_star0_age0', 'n0_early_3010', 'window_s4_b4', 'window_s4_b4_treeall', 'window_s4_b9', 'window_s4_b6',
           'full_s13', 'full_s12', 'full_s10', 'n0_reverse_20453', 'n0_early_20453', 'window_s4_b6_ck3',
           'n0_stateunit_12000', 'full from layer 8, old pieces exact', 'star + tree + non-tree slices, no C·κ3'}

    def table(net, rows, title):
        Bn = b_alone(net)
        lines.append(title + f" Chain B's adj {adj(Bn['raw'], Bn['units']):.4e}. Adj = raw × max(0.1, U/1,024), blank past the "
                     "budget (scored as all zeros); 'tie units' is where the point's raw would tie chain B's score. ★ marks the "
                     "lower envelope: lower raw than every cheaper point (the corrector, which stacks on any chain, left out).")
        lines.append('')
        lines.append('| family | point | units | final raw | ÷ chain B | adj | tie units | note | source |')
        lines.append('|:--|:--|--:|--:|--:|--:|--:|:--|:--|')
        for p in rows:
            tie = adj(Bn['raw'], Bn['units']) / p['raw'] * N
            lines.append(f"| {p['group']} | {'★ ' if p['frontier'] else ''}{p['label']} | {p['units']:,.0f} | {p['raw']:.4e} | {p['ratio_B']:.3f} | {fmt(p['adj'], 4) if p['units'] <= N else ''} | {tie:,.0f} | {p['note']} | {p['src']} |")
        lines.append('')

    for net in (0, 1):
        sub = [p for p in pts if p['net'] == net]
        if sub:
            table(net, [p for p in sub if p['frontier'] or p['label'] in KEY or p['group'].startswith('chain B')],
                  f"**Table R3{'ab'[net]}.** Accuracy against counted units, official network {net}: the lower envelope and the "
                  "anchor points of each family. Every point is a measured final raw and its counted units (chain B metered; "
                  "K=4 suffix rows by the Table 7a ledger; fed pieces chain B metered plus the counted sidecar).")
    for net in (0, 1):
        sub = [p for p in pts if p['net'] == net]
        if sub:
            table(net, sub, f"**Table R4{'ab'[net]}.** Every measured point, official network {net} (curve.csv holds the same rows).")
    (OUT / 'tables.md').write_text('\n'.join(lines) + '\n')
    readout = OUT / 'readout.md'
    if readout.exists():
        head = readout.read_text().split('\n---\n')[0]
        readout.write_text(head + '\n---\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
