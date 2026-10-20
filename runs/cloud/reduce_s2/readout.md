# Matrix-pair compression of two expensive state families

Eight network-0 development points were measured, including one same-window control. The raw MSE values in the released JSON receipts were independently rescored from saved prediction arrays during the original run. The portable command below recomputes the charged units from each point's plan, compares the previous frozen frontier, and checks every value against the released curve. It does not perform a fresh prediction rescore because the large arrays are not included.

```bash
python lab/reduce_s2/make_readout.py
```

| point | network | non-tree / C-kappa3 rank | units | kernel-available units | final raw MSE | raw / B | old envelope raw | new / old | seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w9_n1 | 0 | 1/0 | 2492.81 | 2492.81 | 2.12893546e-08 | 0.9417 | 2.15702359e-08 | 0.9870 | 162.8 |
| w9_n4 | 0 | 4/0 | 3258.09 | 3258.09 | 1.78240473e-08 | 0.7885 | 2.15702359e-08 | 0.8263 | 163.5 |
| w9_n16 | 0 | 16/0 | 6321.35 | 6321.35 | 1.53855651e-08 | 0.6806 | 2.14513369e-08 | 0.7172 | 163.8 |
| w6_n4_c4 | 0 | 4/4 | 11926.44 | 3736.44 | 2.01241975e-08 | 0.8902 | 1.88230182e-08 | 1.0691 | 146.7 |
| w6_n8_c8 | 0 | 8/8 | 13427.07 | 5237.07 | 1.85528205e-08 | 0.8207 | 1.88230182e-08 | 0.9856 | 147.8 |
| baseline_w9 | 0 | 0/0 | 20585.72 | 20585.72 | 1.38941660e-08 | 0.6146 | 1.38967692e-08 | 0.9998 | 182.4 |
| full_n4_c4 | 0 | 4/4 | 29893.10 | 6023.10 | 1.43530524e-08 | 0.6349 | 1.38967692e-08 | 1.0328 | 183.3 |
| full_n8_c8 | 0 | 8/8 | 33215.41 | 9345.41 | 1.25382247e-08 | 0.5546 | 1.05719110e-08 | 1.1860 | 183.5 |

Validated 8 released JSON receipts; new combined-frontier points: w9_n1, w9_n16, w9_n4.

The main `units` column retains the actual C-kappa3 leg-generation charge. `kernel_available_units` represents a kernel-already-available hypothesis and omits that charge. Both columns are representation ledgers, not complete end-to-end flopscope counts. These points were selected on public development network 0; network 1 was not run. The combined frontier has three new non-tree-only points: w9_n1, w9_n4, and w9_n16. The two-family points do not enter that frontier. The control differs from the earlier same-window point by -0.01873% raw MSE.

The experimental recurrence is in `lab/reduce_s2/run.py`, matrix-pair operators in `pairs.py`, and the small-chain checks in `smoke.py`. The JSON receipts retain the plan, rank, compression, replacement events, and unit rows. `curve.csv` retains the complete comparison columns. Full prediction arrays and machine-control records are excluded from this repository.

No matched local-to-official calibration RMSE was recomputed for these points; their selection signal remains statistical noise under the competition rule. Official measured gain: 0 (no submission of these configurations).
