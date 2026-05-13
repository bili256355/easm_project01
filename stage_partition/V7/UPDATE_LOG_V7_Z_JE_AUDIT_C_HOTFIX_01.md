# UPDATE_LOG_V7_Z_JE_AUDIT_C_HOTFIX_01

## Purpose
Fix a crash in `W45_Je_physical_variance_audit_v7_z_c.py` when smoothed/boundary data contain all-NaN Je profiles or all-NaN 2D regional fields for some days.

## Error fixed
`ValueError: All-NaN slice encountered` from `np.nanargmax(x)` inside `_profile_metrics()`.

## Changes
- `_profile_metrics()` now emits invalid rows for all-NaN or insufficient-finite profile days instead of crashing.
- `_region_2d_metrics()` now emits invalid rows for all-NaN 2D days instead of crashing.
- Daily speed now becomes NaN when adjacent days lack enough finite values.
- Quantile/rank flags ignore invalid NaN days safely.
- Bootstrap minimum-day selection ignores invalid NaN days and records NaN when no valid minima exist.

## Scientific effect
No intended change to the Je physical-variance audit logic. This hotfix only makes missing/smoothed boundary days explicit and non-fatal. Invalid days remain NaN and should not be interpreted as low variance.
