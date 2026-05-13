# V8 state_coordinate_meaning_audit_v8_a

## Purpose

This patch adds a coordinate-meaning audit layer above `state_relation_v8_a`.
It does **not** modify peak-only results or state-relation segment/block results.

The audit answers a more basic question before interpreting any raw `ΔS_AB(t)` result:

> Is `S_A(t)` only an object-internal post-likeness coordinate, or can it be treated as a shared/cross-object seasonal-progress coordinate?

## Added files

- `V8/src/stage_partition_v8/state_coordinate_meaning_audit_v8_a.py`
- `V8/scripts/run_state_coordinate_meaning_audit_v8_a.py`
- `V8/UPDATE_LOG_V8_STATE_COORDINATE_MEANING_A.md`

## Outputs

`V8/outputs/state_coordinate_meaning_audit_v8_a/per_window/W045/`

- `object_state_coordinate_validity_W045.csv`
- `common_seasonal_process_audit_W045.csv`
- `pairwise_state_scale_comparability_W045.csv`
- `state_relation_coordinate_gate_W045.csv`
- `state_coordinate_meaning_summary_W045.md`

Cross-window summary:

- `cross_window/state_coordinate_meaning_audit_summary_all_windows.csv`
- `run_meta.json`

## Explicit exclusions

- No growth diagnostics.
- No process_a structure classification.
- No rollback / multi-pulse / non-monotonic interpretation.
- No reclassification of `state_relation_v8_a` segments or blocks.
- Raw `ΔS_AB(t)` is **not** automatically licensed as a cross-object state-progress difference.

## Interpretation rule

Unless a pair reaches a calibrated interpretation level, existing state relation blocks must be described only as object-specific relative post-likeness diagnostics.
They must not be written as lead/lag, synchrony, catch-up, or causality.
