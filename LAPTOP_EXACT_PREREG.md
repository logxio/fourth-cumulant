# Frozen 128 × 16 laptop comparison

## Networks and truth

Generate three width-128, depth-16 random ReLU MLPs. For seeds 3, 4 and 5,
draw each input-by-output weight matrix with `np.random.default_rng(seed)` as
`(standard_normal((128, 128)) * sqrt(2 / 128)).astype(np.float32)`.
The input has zero mean and identity covariance. Each saved `gt` array has all
16 means from 10,100,736 antithetic Monte Carlo samples. The command checks
the SHA-256 of the complete NPZ and the stacked weight bytes before scoring.

| Seed | Truth NPZ SHA-256 | Weight-array SHA-256 | Archived full K=4 raw MSE | Archived augmented K=3 raw MSE |
|---:|---|---|---:|---:|
| 3 | `033bda54ff4c8e7cbaa0e6cc95131d2456825b425735f99fe696307041378081` | `eeaa2243c7c57d0155692f4e136fa7caa51d9807f37a5dc840f46bde1286b5e3` | `1.224e-5` | `5.100e-5` |
| 4 | `3e66c6a6adb1ea5709a206aeb3822d2b7e08f915da4870e7f2b496a1f36bc23d` | `6c01182958971fb6c63da14c64da20048e5f237b6c008e237c73bafaffad0fa3` | `3.130e-5` | `6.265e-5` |
| 5 | `135b93738cc4a1fba92c82286e208ed57300639eae02b14766f8db7840719112` | `a920e669d7d3679d8cb3d533dabbd2a30ca45195a90fcbc43b7d56a24e625855` | `1.604e-5` | `5.100e-5` |

## Arms and decision rule

- K=4 is the full fourth-cumulant chain, with every path, star, three2,
  slice4, slice31, slice22 and slice211 source. Each source retains two birth
  legs, a dense birth kernel and cumulative transport `Q`. Sources receive no
  rank truncation or joining.
- K=3 is the factored augmented third-cumulant reference, including its
  fourth-order radial feedback. Both arms run live on the same weights.
- One flopscope unit equals `2**31` charged FLOPs. Each process uses one BLAS
  and OpenMP thread. Score the last-layer raw MSE against the saved `gt`.
  Require K=3/K=4 > 1 on every network and on the mean of the three MSEs.
- The single-network command targets <15 s single-process CPU time. The
  three-network command runs three single-thread processes per arm in two
  successive stages and targets <30 s wall time. The conservative sum of
  simultaneous process high-water RSS values, including the parent, must be
  <2 GiB. Runtime measurements refer to an Apple M5 with Python 3.13.

The archived reference readings are in `data/laptop_truth_128/order_scan.log`.
`python3 verify.py --laptop` reruns and checks all three live pairs. The main
`python3 verify.py` checks the 1,935 released numerical readings directly.
