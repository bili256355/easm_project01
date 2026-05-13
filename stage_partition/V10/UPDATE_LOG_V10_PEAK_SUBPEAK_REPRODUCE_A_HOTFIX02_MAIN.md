# UPDATE_LOG: V10 peak_subpeak_reproduce_v10_a hotfix02 main-tree implementation

## Purpose

This hotfix moves the confirmed V10-a1 audit-bundle corrections back into the normal V10 main project tree.

The temporary audit-bundle folder `V10/peak_subpeak_reproduce_v10_a1_audit_bundle/` is no longer required for the corrected V10 reproduction run.

## Main changes

1. `V10/src/stage_partition_v10/peak_subpeak_reproduce_v10_a.py`
   - Keeps the independent semantic rewrite boundary: no import/call of V9 or V7 modules.
   - Corrects the main-candidate selector to use V9-equivalent `overlap_fraction`, not raw `overlap_days`.
   - Adds `window_id` to `raw_profile_detector_scores` output.
   - Updates V9 regression audit to compare raw scores against V9 per-window raw score CSVs with inferred `window_id`.
   - Writes regression audit outputs under the normal main output tree:
     - `V10/outputs/peak_subpeak_reproduce_v10_a/audit/`
     - a compatibility copy of the main audit table under `cross_window/`.
   - Cleans stale diff-detail outputs before each regression audit run and writes empty diff summary tables when there are no differences.

2. `V10/scripts/run_peak_subpeak_reproduce_v10_a.py`
   - Main V10 entry remains unchanged.

## Output location

The corrected run writes to the normal V10 main output layout:

```text
V10/outputs/peak_subpeak_reproduce_v10_a/
V10/logs/peak_subpeak_reproduce_v10_a/
```

It does not write to the temporary audit-bundle folder.

## Regression success criterion

After running, check:

```text
V10/outputs/peak_subpeak_reproduce_v10_a/audit/v10_vs_v9_subpeak_regression_audit.csv
```

The four regression layers should pass:

```text
main_window_selection
object_profile_window_registry
raw_profile_detector_scores
bootstrap_selected_peak_days
```

A compatibility copy is also written to:

```text
V10/outputs/peak_subpeak_reproduce_v10_a/cross_window/v10_vs_v9_subpeak_regression_audit.csv
```

## Boundary

This hotfix is a reproduction/engineering correction only. It does not introduce sensitivity interpretation, physical subpeak classification, or scientific conclusions.
