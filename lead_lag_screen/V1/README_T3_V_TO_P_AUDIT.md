# lead_lag_screen/V1 T3 V→P disappearance audit

## Purpose

This patch adds a lightweight audit inside `lead_lag_screen/V1` to explain why V→P candidates shrink in T3 after the V1 stability judgement layer.

It does **not** rerun V1 correlations, surrogates, or stability bootstrap.  
It reads existing outputs from:

```text
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b
```

Optional context is read from:

```text
index_validity/V1_b_window_family_guardrail/outputs/window_family_guardrail_v1_b_smoothed_a/tables
```

## Run

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_t3_v_to_p_disappearance_audit.py
```

Default output:

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_audit
```

## Main outputs

```text
tables/t3_v_to_p_attrition_waterfall.csv
tables/v_to_p_window_attrition_comparison.csv
tables/t3_v_to_p_by_source_v_index.csv
tables/t3_v_to_p_by_target_p_index.csv
tables/t3_v_to_p_lag_profile_long.csv
tables/t3_v_to_p_lag_profile_summary.csv
tables/v_to_p_window_null_difficulty_detail.csv
tables/v_to_p_window_null_difficulty_summary.csv
tables/t3_v_to_p_index_validity_context.csv
tables/t3_v_to_p_disappearance_reason_summary.csv
summary/summary.json
summary/T3_V_TO_P_AUDIT_README.md
```

## Interpretation guardrail

This is an attrition audit. It can say where T3 V→P candidates are lost:

- weak positive-lag signal,
- main AR(1) surrogate failure,
- audit-null instability,
- reverse/forward direction competition,
- tau0 competition,
- or survival as stable lag / tau0-coupled candidate.

It cannot prove pathway, causality, or physical mechanism.
