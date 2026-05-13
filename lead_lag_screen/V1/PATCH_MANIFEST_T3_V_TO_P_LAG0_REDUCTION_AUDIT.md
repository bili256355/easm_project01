# Patch manifest: T3 V→P lag0/lagged reduction audit

Added files:

```text
lead_lag_screen/V1/scripts/run_t3_v_to_p_lag0_reduction_audit.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_lag0_reduction_settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_lag0_reduction_io.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_lag0_reduction_logic.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_lag0_reduction_pipeline.py
lead_lag_screen/V1/README_T3_V_TO_P_LAG0_REDUCTION_AUDIT.md
lead_lag_screen/V1/PATCH_MANIFEST_T3_V_TO_P_LAG0_REDUCTION_AUDIT.md
```

No existing V1 mainline files are overwritten. The audit reads existing outputs and writes a new output directory.
