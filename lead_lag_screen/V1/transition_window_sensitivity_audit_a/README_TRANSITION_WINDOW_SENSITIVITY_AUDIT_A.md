# V1 transition_window_sensitivity_audit_a

## Purpose

This audit checks whether low relationship density in V1 transition windows is a window-design artifact or a robust transition-window feature.

It focuses on T1/T2/T3/T4 and compares them against:

1. equal-length subwindows from adjacent stages;
2. transition-window expansion / forward / backward / shifted variants;
3. random equal-length subwindows from adjacent stages;
4. lightweight AR(1)-effective-n null threshold diagnostics.

It does not modify V1 main code or V1 main outputs.

## Entry

```bat
python D:\easm_project01\lead_lag_screen\V1\transition_window_sensitivity_audit_a\run_v1_transition_window_sensitivity_audit_a.py
```

Fast check:

```bat
python D:\easm_project01\lead_lag_screen\V1\transition_window_sensitivity_audit_a\run_v1_transition_window_sensitivity_audit_a.py --debug-fast
```

Formal run with more random subwindows:

```bat
python D:\easm_project01\lead_lag_screen\V1\transition_window_sensitivity_audit_a\run_v1_transition_window_sensitivity_audit_a.py --n-random 1000
```

## Inputs

Strict default smooth5 index path:

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv
```

Default V1 stability table:

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv
```

## Outputs

Default output directory:

```text
D:\easm_project01\lead_lag_screen\V1\transition_window_sensitivity_audit_a\outputs
```

Key tables:

```text
tables/equal_length_window_contrast.csv
tables/transition_window_expansion_sensitivity.csv
tables/equal_length_random_subwindow_null.csv
tables/window_variant_null_threshold_summary.csv
tables/window_variant_pair_diagnostics.csv
tables/transition_window_sensitivity_diagnosis.csv
summary/window_registry_used.csv
summary/summary.json
summary/run_meta.json
logs/RUN_LOG.md
```

## Important interpretation note

This audit recomputes lightweight AR(1)-effective-n lead-lag diagnostics for window variants. It is intended for window-sensitivity comparison and does not replace V1's original surrogate/FDR judgement.
