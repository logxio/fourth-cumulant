# Frozen report validation, inference phase

Official mini indices 5,6,7; width1024 depth16; four arms full,no_k3c,no_path,no_slicex,exact specs in config.json. Each network is a complete same-machine four-arm comparison. All results, failures and missing entries reported. mini remains evaluation-only. No coefficient fitting or selection.

Primary replication: for each network no_k3c MSE > no_path MSE > no_slicex MSE; all ratios versus full reported. Three-network claim only after all three complete. Width-growth criterion C3 of old PREREG is not independently tested by this batch.

First finite batch mini5 only (four arms) because current full-account credit supports roughly one hour. Later mini6,7 follow unchanged frozen protocol when balance permits. Full GPU job includes 10 generated width16/depth4 smoke networks, upload/receive/readback/numerical CPU comparison gate before first 1024 arm.

Second experiment fixed mini1,2,3; B preactivation snapshot at layer4 from unchanged base_b; exact versus --tree-only, same spec star,three2,slice4+slice31,!slice22+slice211. Measure final MSE relative to saved B output on same network. These are inference probes, not scoring artifacts. This phase does not train LT2; training needs a separate real optimizer-resume asset plan.
