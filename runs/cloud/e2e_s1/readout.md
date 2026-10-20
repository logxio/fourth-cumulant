# Metered end-to-end K=4 chain

The chain starts with a measured chain B prefix at layer 4 and carries the full fourth cumulant to the final layer. `m1` retains dense slice kernels and the rank-128 old-feedback projection; `p8` compresses the (2,1,1) kernel into eight matrix pairs. The suffix executes through flopscope.numpy with all traced operations counted. Units are flopscope FLOPs divided by 2^31. Raw MSE uses the public mini-network truth.

The two network-0 runs completed all 16 layers. The network-1 dense run completed through layer 14 and the network-1 pair run through layer 6; their later readings are absent. The wall and RSS figures describe a shared CPU research machine, not the competition submission runtime. This implementation exceeds the 1,024-unit competition budget.

The original run rescored the complete network-0 prediction arrays against truth. The portable command below checks saved full-run receipts against their per-layer records and rebuilds the tables without those large prediction arrays:

```bash
python lab/e2e_s1/write_tables.py
```

| run | network | through layer | raw MSE | metered units | suffix wall | peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| m1_net0 | 0 | 15 | 6.3694e-09 | 756,620 | 6.20 h | 114.8 GiB |
| p8_net0 | 0 | 15 | 8.3429e-09 | 512,138 | 4.99 h | 72.9 GiB |
| m1_net1 | 1 | 14 | 6.4760e-09 | 656,494 | 6.14 h | 114.8 GiB |
| p8_net1 | 1 | 6 | 3.7524e-09 | 62,892 | 1.07 h | 65.8 GiB |

| metered component | m1_net0 units | p8_net0 units |
|---|---:|---:|
| prefix | 25 | 25 |
| expand | 12,389 | 12,389 |
| seed | 15 | 64 |
| linear | 34,314 | 36,789 |
| birth_dense_slice | 60 | 0 |
| birth_pairs_compress | 0 | 690 |
| birth_two_leg_store | 11 | 11 |
| feedback_projection | 33,724 | 0 |
| project_old | 4,677 | 4,677 |
| read_core | 3,340 | 3,340 |
| read_dense_age0 | 77 | 77 |
| read_dense_slice | 345,045 | 0 |
| read_dense_two_leg | 184,905 | 184,905 |
| read_slice_legs | 135,961 | 265,661 |
| read_slice_pairs | 0 | 1,443 |
| other_nonlinear | 2,076 | 2,067 |

| layer | m1_net0 | p8_net0 | m1_net1 | p8_net1 |
|---:|---:|---:|---:|---:|
| 4 | 3.0856e-09 | 3.0856e-09 | 2.7028e-09 | 2.7028e-09 |
| 5 | 4.2422e-09 | 4.1248e-09 | 3.2603e-09 | 3.2042e-09 |
| 6 | 4.4639e-09 | 4.4947e-09 | 3.8077e-09 | 3.7524e-09 |
| 7 | 4.5660e-09 | 4.5537e-09 | 4.4343e-09 |  |
| 8 | 4.5250e-09 | 4.7842e-09 | 5.1368e-09 |  |
| 9 | 4.5693e-09 | 5.1410e-09 | 4.5486e-09 |  |
| 10 | 5.1091e-09 | 5.9293e-09 | 4.8545e-09 |  |
| 11 | 5.3421e-09 | 6.3085e-09 | 4.8172e-09 |  |
| 12 | 5.5198e-09 | 6.5343e-09 | 5.8693e-09 |  |
| 13 | 5.4930e-09 | 6.7807e-09 | 5.8808e-09 |  |
| 14 | 6.2191e-09 | 8.3638e-09 | 6.4760e-09 |  |
| 15 | 6.3694e-09 | 8.3429e-09 |  |  |

| layer | m1_net0 units | p8_net0 units | m1_net1 units | p8_net1 units |
|---:|---:|---:|---:|---:|
| 4 | 6,348 | 5,410 | 6,348 | 5,410 |
| 5 | 21,782 | 17,849 | 21,782 | 17,849 |
| 6 | 37,490 | 27,156 | 37,490 | 27,156 |
| 7 | 50,230 | 36,868 | 50,230 |  |
| 8 | 56,947 | 40,557 | 56,947 |  |
| 9 | 63,118 | 43,700 | 63,118 |  |
| 10 | 69,289 | 46,843 | 69,289 |  |
| 11 | 75,459 | 49,985 | 75,459 |  |
| 12 | 81,630 | 53,128 | 81,630 |  |
| 13 | 87,800 | 56,270 | 87,800 |  |
| 14 | 93,971 | 59,413 | 93,971 |  |
| 15 | 100,126 | 62,483 |  |  |

The dense network-0 chain ends at raw MSE 6.3694e-9 and 756,620 measured units (6.20 hours, 114.8 GiB peak RSS). The eight-pair chain ends at raw MSE 8.3429e-9 and 512,138 units (4.99 hours, 72.9 GiB peak RSS). The reference dense research run gives 6.3785e-9 on the same network. This is a representation and metering study, not a scored submission. No official gain was measured.

To rerun a full network, use a CPU host with at least 128 GiB RAM and approximately six hours for the dense network-0 arm. Install the external ARC reference at the revision in `REPRODUCE.md`; use `lab/e2e_s1/pack.sh`, `remote_setup.sh`, and `run_net.sh` as described in the repository README. `verify_small.py 20 10 2` checks the compact formulas at small width.
