# T3 V→P physical hypothesis audit field-alias hotfix

## Purpose
Fixes field-name compatibility with local smooth5 preprocess files that store fields as:

- `precip_smoothed`
- `v850_smoothed`
- `z500_smoothed`
- `u200_smoothed`

instead of short keys like `precip` or `v850`.

## Files replaced

- `lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_physical_hypothesis_pipeline.py`

## Scientific logic

No scientific logic, windows, indices, regions, metrics, thresholds, or output semantics are changed. The patch only broadens accepted field-key aliases.
