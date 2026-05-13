# Patch manifest: T3 V→P transition-chain report v1_b

## Added files

```text
scripts/run_t3_v_to_p_transition_chain_report_v1_b.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_settings.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_io.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_object_state.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_support.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_correspondence.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_figures.py
src/lead_lag_screen_v1/t3_v_to_p_transition_chain_pipeline.py
README_T3_V_TO_P_TRANSITION_CHAIN_REPORT_V1_B.md
PATCH_MANIFEST_T3_V_TO_P_TRANSITION_CHAIN_REPORT_V1_B.md
```

## Modified files

None. This is an additive patch.

## Output directory

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_transition_chain_report_v1_b
```

## Dependencies / assumptions

This patch assumes the previous field-explanation audit v1_a patch has been
applied and run, especially:

```text
outputs/lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a/tables/region_response_summary.csv
```

The patch recomputes observed support maps in memory for figures only. It does
not persist full-grid support map NPZ files.
