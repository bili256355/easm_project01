# UPDATE LOG — V8 plot_h_jw_state_curves_v8_a

## Purpose
Add a small, isolated plotting utility for the current H vs Jw state-relation review.
This utility does **not** change the state-relation method. It only reads the existing
`state_relation_v8_a` outputs and plots:

- H and Jw state curves (`S_dist`, `S_pattern`)
- `ΔS(H-Jw)` curves

for W045 and the three baseline configurations.

## Files added
- `V8/src/stage_partition_v8/plot_h_jw_state_curves_v8_a.py`
- `V8/scripts/run_plot_h_jw_state_curves_v8_a.py`
- `V8/UPDATE_LOG_V8_PLOT_H_JW_STATE_CURVES_A.md`

## Default input
- `D:\easm_project01\stage_partition\V8\outputs\state_relation_v8_a\`

Optional environment variable override:
- `V8_STATE_PLOT_INPUT_DIR`

## Default output
- `D:\easm_project01\stage_partition\V8\outputs\state_relation_v8_a\figures_h_jw_state_curves_v8_a\W045\`

Optional environment variable override:
- `V8_STATE_PLOT_OUTPUT_DIR`

## Notes
- This tool is intentionally plotting-only.
- It does not compute new segments, blocks, bootstrap summaries, or state diagnostics.
- It is meant only for visual auditing of the current H/Jw mismatch issue.
