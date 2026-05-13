# Patch manifest: T3 V→P physical hypothesis audit

## Added files

```text
lead_lag_screen/V1/scripts/run_t3_v_to_p_physical_hypothesis_audit.py
lead_lag_screen/V1/src/lead_lag_screen_v1/t3_v_to_p_physical_hypothesis_pipeline.py
lead_lag_screen/V1/T3_V_TO_P_PHYSICAL_HYPOTHESIS_AUDIT_README.md
lead_lag_screen/V1/PATCH_MANIFEST_t3_v_to_p_physical_hypothesis_audit.md
```

## Runtime command

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_t3_v_to_p_physical_hypothesis_audit.py
```

## Notes

- This patch does not modify existing V1 screening outputs.
- This patch does not rerun lead-lag tests.
- This patch uses `smoothed_fields.npz` and `index_values_smoothed.csv`.
- Cartopy is optional; `--no-cartopy` falls back to ordinary lon/lat plots.
- Figures are optional; `--no-figures` runs tables only.
