# Frozen local checks for report v16 (written before any run below)

Written 2026-10-15 22:36 JST (13:36 UTC), before inference on any network named here except the parity run.
Code: `lab/k4full.py` sha256 86bdadae3630115f3a037830de9e9a3240ace64491945d5d55a1567e09b4bbd1,
`lab/kscan.py` sha256 8bdc02f081b96593f93e01753f80993caec8152a888d385a3b20c02c49631af1, repo HEAD 2a5b349.
Parity before freezing: `k4full.py kscan 128 16 5 1e7 cpu f64` (spec full) gives 1.6041e-05, the reference `k4s`
value of `runs/kscan_128_16.log` (1.604e-05). Driver: `run.sh` in this folder. Every process runs under
`timeout 295` on this Mac's CPU, float64. Every value is reported; no network or arm is dropped.

## E1. The n/L = 8 cells of 504aldo's F29, with the K=4 this report uses

Question: at the cells where F29 found augmented K=4 17–20x worse than augmented K=3 ((64,8), (96,12)),
what does ARC's unaugmented `factor_k4` (this report's "K=4 full") give against augmented K=3?
Networks: seeded MLPs as in `lab/kscan.py` (weights N(0, 2/n)), seeds 301, 302, 303 at each shape, never used.
Truth: antithetic Monte Carlo, N = 2^24 requested per network. Variants: `k3a`, `k4s` (= K=4 full), `k4a`
(augmented K=4, F29's variant); a variant that fails or exceeds the limits is reported as such.
No pass/fail criterion: the six ratios k3a/k4s and k3a/k4a are reported as measured.

## E2. Which births carry the gain, frozen on new networks

Claim under test (Section 3 of the report, measured at 1024 on official networks 0–1 and at 512 on seed 41):
C·κ3 births carry the largest share of the K=4 gain, path births the second, the crossed pairings of slice births
the least, and the C·κ3 share grows with width.
Networks: 128 x 16 seeds 3, 4, 5 (truth `data/kscan_128_16_{3,4,5}_10000000.npz`, used before only for the
order scan and lifetime runs, never for source-type ablations) and 192 x 16 seeds 601, 602, 603 (new;
truth N = 2x10^7 requested). Arms (`AW_SPEC`, as `SPEC` in `report/make_figures.py`): full, no_k3c, no_path,
no_slicex.
Criteria, fixed now:
- C1: on each of the six networks, raw final MSE(no_k3c) > MSE(no_path).
- C2: on each of the six networks, MSE(no_path) > MSE(no_slicex).
- C3: the mean over its three networks of MSE(no_k3c)/MSE(full) is larger at 192 than at 128.
