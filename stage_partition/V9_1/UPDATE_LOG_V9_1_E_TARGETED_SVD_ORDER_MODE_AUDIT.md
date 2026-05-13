# V9.1_e targeted SVD / MCA order-mode audit

## Purpose

Adds a read-only V9/V9.1 audit branch that uses V9.1_c bootstrap year-influence scores as target variables and extracts target-guided SVD modes from V9.1_d-style whole-window multi-object anomaly features.

This branch addresses the limitation of ordinary EOF/MEOF: ordinary EOF finds the maximum-variance mode, while this branch finds the multi-object anomaly direction most associated with a specific peak/order heterogeneity target.

## Files added

- `V9_1/scripts/run_targeted_svd_order_mode_audit_v9_1_e.py`
- `V9_1/src/stage_partition_v9_1/targeted_svd_order_mode_audit_v9_1_e.py`
- `V9_1/UPDATE_LOG_V9_1_E_TARGETED_SVD_ORDER_MODE_AUDIT.md`

## Boundaries

- Does not modify V9.
- Does not overwrite V9.1_b/c/d outputs.
- Does not use single-year peak.
- Does not assign physical regime/type names.
- Does not add state, growth, or process_a.

## Method summary

1. Read V9.1_c year-influence scores.
2. Select priority/high-effect pairwise/object targets.
3. Reuse V9.1_d MEOF input construction for whole-window multi-object anomaly feature matrix `X`.
4. For each target `y`, compute target-guided SVD direction `u = X' y / ||X' y||`.
5. Validate with permutation, leave-one-year-out mode stability, and leave-one-out cross-validation.
6. Group years by target-mode score high/mid/low terciles.
7. Re-run V9 peak/order logic inside each target-mode group.
8. Output target-mode evidence tables.

## Run

```bat
python D:\easm_project01\stage_partition\V9_1\scripts\run_targeted_svd_order_mode_audit_v9_1_e.py
```

Debug:

```bat
set V9_1E_DEBUG=1
python D:\easm_project01\stage_partition\V9_1\scripts\run_targeted_svd_order_mode_audit_v9_1_e.py
```

Formal:

```bat
set V9_1E_GROUP_BOOTSTRAP_N=500
set V9_1E_PERM_N=500
python D:\easm_project01\stage_partition\V9_1\scripts\run_targeted_svd_order_mode_audit_v9_1_e.py
```
