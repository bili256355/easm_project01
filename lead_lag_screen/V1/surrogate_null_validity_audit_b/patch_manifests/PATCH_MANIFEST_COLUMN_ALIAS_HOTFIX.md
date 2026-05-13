# PATCH MANIFEST: surrogate_null_validity_audit_b column alias hotfix

## Purpose
Fix a crash when the V1 stability table uses `source_variable` / `target_variable` instead of `source_index` / `target_index`.

## Changed file
- `lead_lag_screen/V1/surrogate_null_validity_audit_b/run_v1_surrogate_null_validity_audit_b.py`

## Changes
- Added source aliases: `source_variable`, `source_var`, `x_variable`, `driver_variable`, `cause_variable`.
- Added target aliases: `target_variable`, `target_var`, `y_variable`, `response_variable`, `effect_variable`.
- Improved missing-column error message to print available columns.

## Scientific scope
No audit logic, surrogate logic, or null-threshold logic changed. This is a column-name compatibility hotfix only.
