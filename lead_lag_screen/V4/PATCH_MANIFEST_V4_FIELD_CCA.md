# Patch manifest — lead_lag_screen/V4 field CCA audit

## Added files

```text
lead_lag_screen/V4/scripts/run_lead_lag_screen_v4.py
lead_lag_screen/V4/src/lead_lag_screen_v4/__init__.py
lead_lag_screen/V4/src/lead_lag_screen_v4/settings.py
lead_lag_screen/V4/src/lead_lag_screen_v4/logging_utils.py
lead_lag_screen/V4/src/lead_lag_screen_v4/data_io.py
lead_lag_screen/V4/src/lead_lag_screen_v4/eof_reduce.py
lead_lag_screen/V4/src/lead_lag_screen_v4/cca_core.py
lead_lag_screen/V4/src/lead_lag_screen_v4/comparison.py
lead_lag_screen/V4/src/lead_lag_screen_v4/pipeline.py
lead_lag_screen/V4/README_V4_FIELD_CCA_AUDIT.md
lead_lag_screen/V4/PATCH_MANIFEST_V4_FIELD_CCA.md
```

## Run command

```bat
cd /d D:\easm_project01
python lead_lag_screen\V4\scripts\run_lead_lag_screen_v4.py
```

## Output directory

```text
D:\easm_project01\lead_lag_screen\V4\outputs\field_cca_smooth5_v4_a
```

## Scope

This patch adds a new independent V4 audit layer. It does not modify V1, V2, or V3.
