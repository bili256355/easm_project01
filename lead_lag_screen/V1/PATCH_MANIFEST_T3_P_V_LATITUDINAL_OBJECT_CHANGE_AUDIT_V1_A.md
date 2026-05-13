# Patch Manifest: t3_p_v_latitudinal_object_change_audit_v1_a

## Purpose

Add an independent object-layer audit for precipitation and v850 latitudinal changes across S3 -> T3 -> S4.

This patch responds to the need to avoid over-interpreting fixed `north_northeast` boxes and to handle precipitation as a possible multi-band object rather than a single migrating band.

## Added files

```text
scripts/run_t3_p_v_latitudinal_object_change_audit_v1_a.py
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_settings.py
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_io.py
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_core.py
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_figures.py
src/lead_lag_screen_v1/t3_p_v_latitudinal_object_change_pipeline.py
README_T3_P_V_LATITUDINAL_OBJECT_CHANGE_AUDIT_V1_A.md
PATCH_MANIFEST_T3_P_V_LATITUDINAL_OBJECT_CHANGE_AUDIT_V1_A.md
```

## Does not modify

- V1 lead-lag screen.
- Stability judgement.
- Field explanation audit.
- Transition chain report.
- Any prior output directory.

## Output directory

```text
outputs/lead_lag_screen_v1_smooth5_a_t3_p_v_latitudinal_object_change_audit_v1_a
```

## Important implementation details

- Uses cartopy by default for maps, with `--no-cartopy` fallback.
- Reads only `smoothed_fields.npz` object fields.
- Does not read support/R2/pathway outputs.
- Does not output full-grid NPZ intermediate fields.
- Automatically records the actual data latitude min/max in `summary.json`.
- Generates latband summaries up to the actual latitude extent of the data.
