# PATCH MANIFEST: transition_window_sensitivity_audit_a cache hotfix

## Purpose
Speed up `transition_window_sensitivity_audit_a` without changing the audit logic or outputs.

## Change
- Reuses pair diagnostics for duplicate `(day_start, day_end)` windows.
- Mainly accelerates random equal-length subwindow controls, where many draws repeat the same finite set of possible stage subwindows.
- Adds `n_unique_windows_computed` to summary/run_meta for transparency.

## Files replaced
- `lead_lag_screen/V1/transition_window_sensitivity_audit_a/run_v1_transition_window_sensitivity_audit_a.py`

## Notes
This does not change statistical definitions, input paths, window definitions, or output table schemas.
