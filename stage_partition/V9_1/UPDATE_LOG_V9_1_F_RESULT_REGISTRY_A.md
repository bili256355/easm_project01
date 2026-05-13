# V9.1_f Result Registry A Update Log

## Purpose

Adds a read-only statistical evidence registry layer for V9 + V9.1_f hotfix02.

This patch does not rerun V9, V9.1_f, peak detection, bootstrap, MCA/SVD, or physical interpretation. It only reads existing outputs and creates frozen result-registry tables.

## Added files

- `V9_1/scripts/run_summarize_v9_1_f_result_registry_a.py`
- `V9_1/src/stage_partition_v9_1/summarize_v9_1_f_result_registry_a.py`

## Main outputs

Output directory:

```text
V9_1/outputs/v9_1_f_result_registry_a/
```

Key files:

- `v9_1_f_statistical_result_registry.csv`
- `v9_1_f_result_tier_summary.csv`
- `v9_1_f_window_summary.csv`
- `v9_1_f_pair_summary.csv`
- `v9_1_f_method_audit_freeze_status.csv`
- `v9_1_f_ready_for_physical_audit.csv`
- `v9_1_auxiliary_method_results.csv`
- `V9_1_F_RESULT_REGISTRY_A_SUMMARY.md`
- `v9_1_f_result_registry_input_audit.csv`
- `run_meta.json`

## Interpretation boundary

This registry is a statistical evidence registry. It does not assign physical type names, causal mechanisms, year types, or object-driving claims.

Physical interpretation should only be attempted after the registry has identified results that are ready for physical-audit.
