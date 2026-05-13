# Patch manifest: V1_1 structural V→P screen

## Added files

```text
lead_lag_screen/V1_1/scripts/run_lead_lag_screen_v1_1_structural_vp_a.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/__init__.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/settings.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/data_io.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/core.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/stats_utils.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/evidence_tier.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/logging_utils.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/stability_judgement_classifier.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/structural_indices.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/reporting.py
lead_lag_screen/V1_1/src/lead_lag_screen_v1_1/pipeline.py
lead_lag_screen/V1_1/README_V1_1_STRUCTURAL_VP_A.md
lead_lag_screen/V1_1/PATCH_MANIFEST_V1_1_STRUCTURAL_VP_A.md
```

## No V1 modifications

This patch adds a new `lead_lag_screen/V1_1` tree. It does not replace or edit `lead_lag_screen/V1`.

## Main behavior

1. Read V1 old anomaly indices as read-only.
2. Read foundation smooth5 P/V fields.
3. Compute new structural P/V indices.
4. Convert new structural indices to day-of-season anomalies.
5. Combine old V1 anomalies and new V1_1 anomalies.
6. Run inherited V1-style V→P lead-lag screen across all 9 windows.
7. Apply lag-vs-tau0 and forward-vs-reverse stability judgement.
8. Summarize old-only vs new-index recovery by window.
