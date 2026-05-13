# T3 V→P lag0/lagged reduction audit

This patch adds a V1-internal diagnostic audit for the question:

> Why does T3 lose many fixed-index V→P relations not only in stable positive-lag candidates, but also in lag0/same-day diagnostics?

It does **not** rerun the V1 lead-lag screen. It reads existing V1 smooth5 outputs, the V1 stability judgement layer, foundation smooth5 field/index products, and optional index_validity context tables.

## Run

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_t3_v_to_p_lag0_reduction_audit.py
```

If figure generation is slow or cartopy is unavailable:

```bat
python lead_lag_screen\V1\scripts\run_t3_v_to_p_lag0_reduction_audit.py --no-figures
python lead_lag_screen\V1\scripts\run_t3_v_to_p_lag0_reduction_audit.py --no-cartopy
```

## Output

Default output directory:

```text
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a_t3_v_to_p_lag0_reduction_audit
```

Core outputs:

```text
tables/v_to_p_window_null_difficulty_lag0_pos_detail.csv
tables/v_to_p_window_null_difficulty_lag0_pos_summary.csv
tables/t3_v_to_p_subwindow_lag_profile_long.csv
tables/t3_v_to_p_subwindow_lag_profile_summary.csv
tables/t3_v_to_p_subwindow_shift_summary.csv
tables/t3_v_p_composite_metrics.csv
tables/figure_manifest.csv
tables/t3_v_to_p_index_validity_context_expanded.csv
tables/t3_v_to_p_lag0_and_lagged_reduction_reason_summary.csv
summary/summary.json
summary/run_meta.json
summary/T3_V_TO_P_LAG0_REDUCTION_AUDIT_README.md
```

## Interpretation guardrail

This is a diagnostic audit of why fixed-index V→P relations shrink in T3. It should not be interpreted as a new pathway result or as a physical mechanism proof.
