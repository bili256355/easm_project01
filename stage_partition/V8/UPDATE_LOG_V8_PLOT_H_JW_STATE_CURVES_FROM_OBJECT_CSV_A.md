# UPDATE LOG — V8 plot_h_jw_state_curves_from_object_csv_v8_a

## Purpose
Add a small plotting utility to draw H/Jw state curves from an existing `object_state_curves_*.csv` file,
including old/legacy result files.

## Files added
- `V8/src/stage_partition_v8/plot_h_jw_state_curves_from_object_csv_v8_a.py`
- `V8/scripts/run_plot_h_jw_state_curves_from_object_csv_v8_a.py`
- `V8/UPDATE_LOG_V8_PLOT_H_JW_STATE_CURVES_FROM_OBJECT_CSV_A.md`

## Input
Set:
- `V8_HJW_STATE_OBJECT_CSV` = path to the old result `object_state_curves_*.csv`

Optional:
- `V8_HJW_STATE_PLOT_LABEL` = custom label in figure titles/output names
- `V8_HJW_STATE_PLOT_OUTPUT_DIR` = custom output directory

## Output
Default output:
- `D:\easm_project01\stage_partition\V8\outputs\state_relation_v8_a\figures_h_jw_state_curves_<label>\`

## Notes
- This is plotting-only.
- It computes H-Jw delta from the object-state csv directly, so no pairwise delta csv is required.
