# V1 surrogate null validity audit B

This audit is a standalone diagnostic layer under `lead_lag_screen/V1/`. It does not modify V1 main-screen code or existing V1 outputs.

## Goal

Answer two narrow questions:

1. Can a recomputed AR(1)-style surrogate p-value approximately reproduce V1's existing `p_pos_surrogate`?
2. If the null threshold is high, which layer raises it: single-lag null, max-over-lags/absolute max, or FDR pressure?

## Default required inputs

```bat
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv
```

The script intentionally does not perform broad automatic index-file discovery, to avoid mixing smooth5 and smooth9 products.

## Run

```bat
python D:\easm_project01\lead_lag_screen\V1\surrogate_null_validity_audit_b\run_v1_surrogate_null_validity_audit_b.py
```

Fast check:

```bat
python D:\easm_project01\lead_lag_screen\V1\surrogate_null_validity_audit_b\run_v1_surrogate_null_validity_audit_b.py --debug-fast --n-surrogates 100
```

Formal diagnostic:

```bat
python D:\easm_project01\lead_lag_screen\V1\surrogate_null_validity_audit_b\run_v1_surrogate_null_validity_audit_b.py --n-surrogates 1000
```

## Main outputs

```text
tables/null_input_sample_consistency.csv
tables/surrogate_p_reproduction_check.csv
tables/surrogate_p_reproduction_summary_by_window.csv
tables/null_threshold_decomposition_by_pair.csv
tables/null_threshold_decomposition_by_window.csv
tables/surrogate_null_validity_diagnosis.csv
summary/run_meta.json
summary/summary.json
logs/RUN_LOG.md
```

## Interpretation boundary

This is a diagnostic AR(1)-style surrogate reproduction, not a replacement for V1's original surrogate p-values. If reproduction is poor, the correct conclusion is not that V1 is wrong automatically; it means V1's exact surrogate implementation, sample construction, or index input must be inspected before using null-threshold explanations.
