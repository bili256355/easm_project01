# V9.1_e hotfix02 performance-only patch

## Scope

This hotfix only changes runtime behaviour for `targeted_svd_order_mode_audit_v9_1_e`.
It does not modify V9, V9.1_b/c/d, the targeted-SVD definition, target construction,
permutation semantics, LOO/CV semantics, or output schema.

## Changes

1. Vectorized permutation audit in batches.
   - Previous implementation refit one targeted mode per permutation in a Python loop.
   - New implementation performs batched matrix operations for the same permute-y / refit / |corr(score, y_perm)| calculation.
   - Controlled by `V9_1E_PERM_BATCH_SIZE` (default: 128).

2. Added a performance gate for costly target-mode phase-group peak checks.
   - By default, group-level V9 peak checks are run only when the targeted-SVD mode passes at least `usable_targeted_mode`.
   - Unsupported / overfit / unstable targets are written to `targeted_mode_group_skipped_*.csv` with skip reason:
     `targeted_mode_not_eligible_for_costly_group_peak_check`.
   - This avoids spending most runtime validating targets that are already unsupported at the targeted-SVD screen.
   - Set `V9_1E_FORCE_ALL_GROUP_PEAK_CHECKS=1` to reproduce the previous exhaustive group-peak behaviour.
   - Or set `V9_1E_SKIP_GROUP_PEAK_FOR_UNSUPPORTED=0`.

## New environment variables

```bat
set V9_1E_PERM_BATCH_SIZE=128
set V9_1E_FORCE_ALL_GROUP_PEAK_CHECKS=1
set V9_1E_SKIP_GROUP_PEAK_FOR_UNSUPPORTED=0
```

## Interpretation note

The default performance gate does not promote or demote targeted-SVD modes. It only skips the expensive
phase-group V9 peak replay for modes that have already failed the targeted-SVD eligibility screen. Full
exhaustive behaviour remains available through the environment variables above.
