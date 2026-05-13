# Patch manifest: lead_lag_screen/V1 stability judgement

## Added files

```text
lead_lag_screen/V1/scripts/run_lead_lag_screen_v1_stability_judgement.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stability_judgement_settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stability_judgement_io.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stability_judgement_classifier.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stability_judgement_pipeline.py
lead_lag_screen/V1/README_STABILITY_JUDGEMENT_V1_B.md
```

## Scientific scope

This is a V1 post-processing enhancement. It does not change the original V1 lead-lag calculation, null construction, surrogate results, audit null, or evidence tiers. It reads existing V1 diagnostics and adds a stricter lag-vs-tau0 stability judgement layer.

## Default output tag

```text
lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b
```
