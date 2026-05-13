# UPDATE_LOG_V7_U

## V7-u: W45 H/Jw state–growth separated transition framework

### Purpose

This patch adds a narrow H/Jw-only method trial for W45. It separates two information layers that must not be collapsed:

1. **State-progress information**: whether the raw field is closer to post-state / more post-like than pre-state.
2. **Rapid-growth information**: when the raw field changes fastest or advances most rapidly toward post-like state.

### Key changes

- `P_proj` is downgraded to `P_proj_reference` and is not used as the main adjudication metric.
- Main state metrics are `D_pre`, `D_post`, `P_dist`, `R_pre`, `R_post`, and `R_diff`.
- Main growth metrics are `field_change_norm`, `delta_P_dist`, and `delta_R_diff` with 3-day smoothed variants.
- H/Jw decisions are output separately for state events and growth events.
- Synchrony/equivalence requires a positive equivalence test; unresolved order does not imply synchrony.
- P/V/Je are intentionally not added in this patch.
- No lat-band pairing, no H-low vs Jw-low pairing, and no complex relation labels are used.

### Main entry

```bash
cd D:\easm_project01
python stage_partition\V7\scripts\run_w45_H_Jw_state_growth_transition_framework_v7_u.py
```

### Debug run

```bash
cd D:\easm_project01
set V7U_DEBUG_N_BOOTSTRAP=20
set V7U_SKIP_FIGURES=1
python stage_partition\V7\scripts\run_w45_H_Jw_state_growth_transition_framework_v7_u.py
```

### Main outputs

- `w45_H_Jw_state_metric_curves_v7_u.csv`
- `w45_H_Jw_growth_metric_curves_v7_u.csv`
- `w45_H_Jw_state_event_registry_v7_u.csv`
- `w45_H_Jw_growth_event_registry_v7_u.csv`
- `w45_H_Jw_state_order_sync_decision_v7_u.csv`
- `w45_H_Jw_growth_order_sync_decision_v7_u.csv`
- `w45_H_Jw_state_growth_integrated_summary_v7_u.csv`
- `w45_H_Jw_projection_vs_state_growth_comparison_v7_u.csv`
- `w45_H_Jw_state_growth_method_status_v7_u.csv`
- `w45_H_Jw_state_growth_transition_framework_summary_v7_u.md`
- `run_meta.json`

### Interpretation boundary

This patch does not provide a physical mechanism and does not extend the method to P/V/Je. It is a method trial: first check whether state-progress and rapid-growth layers can produce clear H/Jw decisions. Only after review should the framework be considered for all fields.
