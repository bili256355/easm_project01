# T3 V→P lag0 reduction audit merge hotfix

## Purpose
Fixes a pandas merge failure in `t3_v_to_p_lag0_reduction_pipeline.py` caused by duplicate merge-key columns after renaming AR(1) parameter columns.

## Error fixed
`ValueError: The column label 'source_variable' is not unique.`

## Files replaced
- `lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_lag0_reduction_pipeline.py`

## Scientific impact
None. This only changes dataframe column selection before merge. It does not change windows, variables, metrics, thresholds, figures, or interpretation logic.
