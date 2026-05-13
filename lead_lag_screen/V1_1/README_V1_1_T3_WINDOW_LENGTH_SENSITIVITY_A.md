# V1_1 T3 Window-Length Sensitivity Audit A

This patch adds a V1_1-internal audit for the question:

> Is the low T3 V→P stable-pair count mainly caused by the short 11-day T3 window, or does the V1_1 result still indicate structural index projection mismatch?

It does **not** modify `lead_lag_screen/V1` and does **not** modify the V1_1 main structural VP run outputs. It creates a separate output tree.

## New entry point

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_t3_window_length_sensitivity_a.py
```

Quick connection test:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_t3_window_length_sensitivity_a.py --debug-fast
```

Formal run, matching the V1_1 main resampling scale:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_t3_window_length_sensitivity_a.py --n-surrogates 1000 --n-audit-surrogates 1000 --n-direction-bootstrap 1000
```

## What it runs

Each variant is a one-window V→P rerun under the existing V1_1 lead-lag, surrogate, direction-bootstrap, and lag-vs-tau0 stability framework:

- `T3_current_107_117_len11`
- `S3_center_equal11_092_102`
- `S4_center_equal11_131_141`
- `T3_expand_symmetric17_104_120`
- `T3_expand_symmetric23_101_123`
- `T3_expand_backward17_101_117`
- `T3_expand_forward17_107_123`

The equal-length controls test whether an 11-day window alone collapses pair counts. The expansions test whether T3 relationships recover when the transition window is widened.

## Main outputs

Default output root:

```text
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_t3_window_length_sensitivity_a
```

Key tables:

```text
tables/window_length_sensitivity_pair_counts.csv
tables/window_length_sensitivity_effect_size.csv
tables/t3_expansion_recovery_summary.csv
tables/window_length_sensitivity_diagnosis_table.csv
summary/window_length_sensitivity_registry.csv
summary/summary.json
summary/run_meta.json
```

Each variant also has its own full V1_1-style output subdirectory under the sensitivity output root.

## Interpretation boundary

This audit can support or weaken the claim that T3 pair loss is mostly a short-window artifact. It cannot prove physical causality. It should be interpreted together with the main V1_1 structural VP result.
