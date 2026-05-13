# T3 V→P transition-chain report v1_b

## Purpose

This patch adds an independent reporting/evidence-chain layer:

```text
lead_lag_screen_v1_smooth5_a_t3_v_to_p_transition_chain_report_v1_b
```

It does **not** rerun the V1 lead-lag screen, does **not** rerun stability judgement,
does **not** rerun the 1000-bootstrap field-explanation audit, and does **not**
write large full-grid NPZ map outputs.

It reads the previous `t3_v_to_p_field_explanation_audit_v1_a` region-response
results and combines them with precipitation and v850 object-state/object-change
maps/tables to build a S3 → T3_early → T3_full → T3_late → S4 transition chain.

## Default command

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_transition_chain_report_v1_b.py
```

If cartopy has runtime issues:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_transition_chain_report_v1_b.py --no-cartopy
```

To skip map generation and only write tables:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_transition_chain_report_v1_b.py --no-figures
```

## Inputs

Default foundation inputs:

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_values_smoothed.csv
```

Default previous hard-evidence output:

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a\tables\region_response_summary.csv
```

## Default windows

```text
S3       = days 87–106
T3_early = days 107–112
T3_full  = days 107–117
T3_late  = days 113–117
S4       = days 118–154
```

## Key outputs

Tables:

```text
tables/object_state_region_summary.csv
tables/object_change_region_delta.csv
tables/support_region_transition_matrix.csv
tables/support_region_delta.csv
tables/north_main_south_transition_chain.csv
tables/object_support_correspondence_summary.csv
tables/transition_chain_diagnosis_table.csv
```

Figures:

```text
figures/P_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png
figures/V850_mean_state_S3_T3early_T3full_T3late_S4_cartopy.png
figures/P_change_T3full_minus_S3_cartopy.png
figures/V850_change_T3full_minus_S3_cartopy.png
figures/P_V_support_transition_panel_T3full_minus_S3.png
figures/P_V_support_transition_panel_T3late_minus_T3early.png
figures/P_V_support_transition_panel_S4_minus_T3full.png
```

## Interpretation guardrail

Every increase/decrease/shift statement must name:

```text
comparison + reference window + target window + region + V component
```

This report intentionally prevents unreferenced claims like “south/SCS is more
prominent” or “north takes over.”
