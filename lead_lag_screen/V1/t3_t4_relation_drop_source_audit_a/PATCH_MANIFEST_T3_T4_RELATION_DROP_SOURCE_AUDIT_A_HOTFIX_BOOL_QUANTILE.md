# Hotfix: bool quantile failure in effect distribution

## Scope

Replacement file only:

- `lead_lag_screen/V1/t3_t4_relation_drop_source_audit_a/run_v1_t3_t4_relation_drop_source_audit_a.py`

## Fix

`window_effect_size_distribution_summary.csv` previously selected numeric-looking columns using pandas dtype checks. In the user's NumPy/pandas environment, boolean flag columns can be treated as numeric-like, then fail during `Series.quantile()` with:

```text
TypeError: numpy boolean subtract, the `-` operator, is not supported
```

This hotfix:

1. Excludes boolean columns from `existing_effect_cols()`.
2. Adds `numeric_effect_values()` to coerce effect metrics to float and skip flags.
3. Leaves all scientific/audit logic unchanged.

## Validation

Syntax compile check passed with:

```text
python -S -m py_compile run_v1_t3_t4_relation_drop_source_audit_a.py
```
