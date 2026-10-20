# C45 experiment note and 1024-wide checks

This English experiment note describes the hypotheses, measurements and outcomes. `research/v21_evidence.py` verifies this file and rebuilds the numerical results from the released prediction arrays.

# C45: why removing path births helps at small width

Started 12:34 JST on 17 October 2026. The section "Candidates and predictions" below was written before any new measurement; the readings are written back as they came, right and wrong alike.

## Candidates and predictions (written before measuring)

Notation: d = pred_full − pred_no_path (the net contribution of path births to the final-layer mean, including all its feedback into later layers), r_np = truth − pred_no_path. If the path contribution is added back with weight α, pred(α) = pred_no_path + α·d, the error is a parabola in α, with optimal weight β = ⟨r_np, d⟩ / |d|². α = 1 (full) beats α = 0 (no_path) if and only if β > 1/2.
So "removing it helps" has only three possible causes: β ∈ (0, 1/2) (right direction, too much), β < 0 (wrong direction), or cos(r_np, d) ≈ 0 (unrelated to the gap, pure added error).

1. Truth noise. The 128-wide truths use 10⁷ samples. Prediction (if noise is not the cause): the final-layer ΔMSE = MSE(no_path) − MSE(full) exceeds its truth-noise standard error more than 5 times;
   with an independent Monte Carlo truth the ratio reproduces to within ±0.02.
2. The expansion parameter L/n (series truncation). Path births are the first-order term of the L/n expansion, and the next order, which K=4 does not carry, has the opposite sign. Predictions:
   - Layer by layer on the same 16-layer network (the prediction at layer l is the prediction for the network's first l layers), the ratio MSE(no_path)/MSE(full) turns from >1 to <1 at a layer l*,
     and l*/n falls in one narrow band for n = 64, 96, 128, 192, 256 (a rough look at the existing every-third-layer logs puts 128 at l ≈ 8–13 and 192 at l ≈ 13–16, so l/n ≈ 0.06–0.09).
     The corresponding predictions: the 64 network turns at l ≈ 4–6, the 96 network at l ≈ 6–9, and the 256 × 16 final-layer ratio is 0.95–1.15 with the turn at l ≥ 15.
   - β falls monotonically with l/n: β > 1/2 below the turn, β ∈ (0, 1/2) and cos(r_np, d) > 0 above it. The direction stays right, and the amount overshoots more and more.
   - The mean squared correlation ρ² of the first l layers (mean square of the off-diagonal pre-activation correlations) grows with l/n, and the turn falls near one value of ρ².
3. Overlap with other birth types: |cos(d_path, d_C·κ3)| or |cos(d_path, d_star)| > 0.8 with opposite sign means mutual cancellation; removing one exposes the other.
4. Birth time: if the harm comes from local state (correlations already large at the birth layer), it concentrates in path births of the second half (b ≥ 8); if it comes from long-range transport, in path births of the first half.
   Linearity check: the α = 0.5 arm's error falls on the parabola computed from the full and no_path arms (±5%), and the α = β arm beats both full and no_path.

## Added prediction (17:2x JST on 17 October, written before the no_star and no_k3c arms of 96 × 12 and 192 × 16 gave readings)

Readings from three 128 × 16 networks and three 64 × 8 networks: on the 4 networks where removing path helps, the final-layer cos(d_path, d_star) is negative on all (−0.23 to −0.47); on the 2 where it hurts, it is positive (+0.05, +0.18).
The least-squares weight of star births is also > 1 on all 6 networks (1.09–2.04). Prediction: on three 96 × 12 networks (301, 302, 303) and three 192 × 16 networks (601, 602, 603),
networks with final-layer cos(d_path, d_star) < 0 improve when path is removed (R_last < 1), and those with > 0 get worse (R_last > 1); the sign matches on at least 10 of the 12 networks;
the star weight is > 1 on all 12.

## 1024-wide predictions (21:3x JST on 17 October, written before the no_star arm of official mini 5, 6, 7 gave readings)

Three arms exist already: full, no_path and no_k3c (`data/frozen_1024/birth{5,6,7}_*.npy`). Add one no_star arm and compute with the same formulas as the summary script. Predictions:
1. The least-squares star weight is > 1 on all three networks.
2. Scaling each network's star contribution by the mean star weight of the other two networks lowers its final-layer MSE below full K=4 on all three.
3. At 1024, removing path hurts on all three networks (measured 1.43×, 1.27×, 1.51×); by the small-width mechanism, cos(path, star) > 0 on all three.

## Commands

- Arms: `report/data/c45/arms.py` (full, no_path, no_star, no_k3c; 64 × 8, 96 × 12 and 128 × 16 in float64, 192 × 16 in float32; one local process ≤ 1.3 GB). Truths `report/data/frozen_s16/truth/kscan_<n>_<L>_<seed>_*.npz`; 128 also has an independent Monte Carlo truth (not released).
- Summary: the local summary script (not released; `research/v21_evidence.py` recomputes every value the report uses). All arms finished at 17:38 JST on 17 October and were reread at 21:2x.

## Readings

| n | L | seed | l*/n | R_last | β | cos(r,d) | w_path | w_star | w_C·κ3 | cos(path,star) | leave-one-out star weight → final MSE ÷ full |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 64 | 8 | 301 | 0.109 | 0.878 | −0.464 | −0.125 | 0.85 | 1.09 | 0.54 | −0.33 | ×1.59 → 0.722 |
| 64 | 8 | 302 | – | 1.342 | 1.974 | 0.581 | 2.02 | 1.40 | 0.81 | 0.05 | ×1.43 → 0.968 |
| 64 | 8 | 303 | – | 1.030 | 0.620 | 0.216 | 0.51 | 1.77 | 0.67 | 0.18 | ×1.25 → 0.932 |
| 96 | 12 | 301 | 0.104 | 0.953 | 0.017 | 0.004 | 1.18 | 1.45 | 0.62 | −0.39 | ×1.40 → 0.779 |
| 96 | 12 | 302 | 0.062 | 0.991 | 0.451 | 0.139 | 0.71 | 1.65 | 0.84 | −0.22 | ×1.30 → 0.911 |
| 96 | 12 | 303 | – | 1.223 | 2.006 | 0.493 | 1.49 | 1.15 | 0.48 | 0.19 | ×1.55 → 0.806 |
| 128 | 16 | 3 | 0.070 | 0.959 | 0.354 | 0.136 | 1.11 | 1.58 | 0.82 | −0.40 | ×1.93 → 0.610 |
| 128 | 16 | 4 | 0.102 | 0.825 | −1.692 | −0.372 | 0.14 | 2.04 | 0.74 | −0.47 | ×1.70 → 0.611 |
| 128 | 16 | 5 | 0.102 | 0.970 | 0.278 | 0.074 | 1.07 | 1.83 | 0.88 | −0.23 | ×1.81 → 0.436 |
| 192 | 16 | 601 | 0.083 | 0.917 | 0.262 | 0.114 | 0.27 | 1.38 | 0.94 | 0.01 | ×1.21 → 0.785 |
| 192 | 16 | 602 | – | 1.126 | 1.205 | 0.340 | 1.39 | 1.31 | 0.81 | −0.22 | ×1.25 → 0.889 |
| 192 | 16 | 603 | 0.078 | 0.896 | 0.068 | 0.025 | 0.73 | 1.11 | 0.67 | −0.39 | ×1.35 → 0.772 |

R_last = final-layer MSE(no_path)/MSE(full); w_* are the least-squares weights of the three birth types' net contributions on the final layer (1 = as is); the leave-one-out column scales this network's star contribution by the mean star weight of the other two networks of the same shape, so the network takes no part in its own fit.

## Conclusions

- **K=4 undercounts star births.** The star weight is > 1 on all 12 networks (1.09–2.04, median 1.42); the added prediction's second part holds on 12 of 12. Scaling each network by one star coefficient fitted on the other two networks of its shape (×1.21–1.93) lowers the final-layer error on all 12, to 0.436–0.968 of full K=4, geometric mean about 0.75; on the three 192 × 16 networks 0.77–0.89.
- **Removing path helps because path points against the undercounted star.** The sign of cos(path, star) matches R_last < 1 on 10 of 12 networks, and on all nine at 64, 96 and 128. Both misses are at 192: on 601 cos is only +0.01 (zero), and on 602 cos is −0.22 while R is 1.126. The added prediction's "at least 10" holds exactly; on the six new networks at 96 and 192 the sign matches on 4.
- **The turning depth falls in one narrow band.** On the 8 networks where removing path helps from some layer onward, l*/n is 0.062–0.109, slightly wider than the predicted 0.06–0.09 and of the same order; it does not drift with width from 64 to 192.
- **Verdict on candidates 1–4:** the wrong direction (β < 0) appears only on 64-301 and 128-4; the other networks that improve have β ∈ (0, 1/2) and cos > 0, the right direction with too much weight, so candidate 2's "overshoots more and more" holds. Candidate 3's mutual cancellation with |cos| > 0.8 does not hold (|cos(path, star)| ≤ 0.47): the mechanism is undercounted star births plus path births moderately opposed to them, not two terms cancelling.
- **How the report uses it:** the Section 3 sentence on path births helping at small width becomes this mechanism, with one sentence that the leave-one-out star coefficient lowers the error on 12 of 12, geometric mean 0.75. Whether 1024 undercounts the same way is not yet measured and is written as the next step (one network each of the research chain's full and no_star arms at width 1024 would test it).

---

## Outcome of the 1024 predictions (from the released files, not part of the note)

`research/v21_evidence.py` rebuilds these from `report/data/c45/arms_last.npz` and checks them against `data/numbers.csv`:

| prediction | holds on | reading on mini 5, 6, 7 |
|---|---|---|
| 1. star weight > 1 | 3 of 3 | 1.030–1.071 |
| 2. leave-one-out star scaling below full K=4 | 3 of 3 | 0.994–0.998 of full K=4 |
| 3. cos(path, star) > 0 | 0 of 3 | −0.087 to −0.027 |

At 1024 the undercount nearly closes: fitting all three weights in sample brings the final MSE only to 0.914–0.949 of full K=4, and removing star births costs 1.79–2.05×.
