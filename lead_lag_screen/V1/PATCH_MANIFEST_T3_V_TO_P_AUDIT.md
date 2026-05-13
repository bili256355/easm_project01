# Patch manifest: T3 V→P disappearance audit

## Added files

```text
lead_lag_screen/V1/scripts/run_t3_v_to_p_disappearance_audit.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_audit_settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_audit_io.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_audit_logic.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_audit_pipeline.py
lead_lag_screen/V1/README_T3_V_TO_P_AUDIT.md
lead_lag_screen/V1/PATCH_MANIFEST_T3_V_TO_P_AUDIT.md
```

## Semantics

- Does not modify existing V1 outputs.
- Does not rerun V1 lead-lag calculations.
- Requires the V1 stability judgement output to exist.
- Focuses on V→P attrition in T3 and compares all standard windows.
