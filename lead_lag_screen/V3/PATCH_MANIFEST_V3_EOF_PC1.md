# Patch manifest: lead_lag_screen/V3 EOF-PC1 audit

## Added files

```text
lead_lag_screen/V3/scripts/run_lead_lag_screen_v3.py
lead_lag_screen/V3/src/lead_lag_screen_v3/__init__.py
lead_lag_screen/V3/src/lead_lag_screen_v3/settings.py
lead_lag_screen/V3/src/lead_lag_screen_v3/logging_utils.py
lead_lag_screen/V3/src/lead_lag_screen_v3/data_io.py
lead_lag_screen/V3/src/lead_lag_screen_v3/eof_pc1.py
lead_lag_screen/V3/src/lead_lag_screen_v3/stats_utils.py
lead_lag_screen/V3/src/lead_lag_screen_v3/lead_lag_core.py
lead_lag_screen/V3/src/lead_lag_screen_v3/comparison.py
lead_lag_screen/V3/src/lead_lag_screen_v3/pipeline.py
lead_lag_screen/V3/README_V3_EOF_PC1_AUDIT.md
lead_lag_screen/V3/PATCH_MANIFEST_V3_EOF_PC1.md
```

## Intended run command

```bat
cd /d D:\easm_project01
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3.py
```

## Default output directory

```text
D:\easm_project01\lead_lag_screen\V3\outputs\eof_pc1_smooth5_v3_a
```

## Core design decisions implemented

- Uses smooth5 field anomalies from `foundation/V1/outputs/baseline_smooth5_a/preprocess/anomaly_fields.npz`.
- If `anomaly_fields.npz` is unavailable, computes anomaly as `smoothed_fields.npz - daily_climatology.npz`.
- Reuses foundation/V1 object domains for P/V/H/Je/Jw.
- Fits EOF PC1 separately for each window × object.
- Uses only PC1.
- Fixes PC1 sign by correlation with representative V1 index anomalies.
- Reuses V1-style target-side windows and lead-lag max-stat AR(1) surrogate testing.
- Generates V1-index vs V3-PC1 comparison tables and T3-focused audit output.

## Non-goals

- No PCMCI.
- No pathway reconstruction.
- No PC2/PC3.
- No new stage/window definition.
```
