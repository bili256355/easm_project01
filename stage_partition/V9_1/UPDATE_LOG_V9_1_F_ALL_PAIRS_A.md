# UPDATE_LOG_V9_1_F_ALL_PAIRS_A

## Version

`v9_1_f_all_pairs_a`

## Purpose

Original V9.1_f audited 15 preselected priority peak-order targets rather than all possible object pairs. This creates a target-selection coverage limitation: the original output cannot be interpreted as a complete window-wide order-heterogeneity diagnosis.

This patch preserves the V9.1_f bootstrap-composite MCA method but expands the target registry to all 10 P/V/H/Je/Jw object pairs for each accepted window:

- W045: 10 pairs
- W081: 10 pairs
- W113: 10 pairs
- W160: 10 pairs

Total: 40 targets.

## Semantic change

Only target coverage is changed.

The method remains:

```text
bootstrap composite X_b
Y_b = peak_B(b) - peak_A(b)
u ∝ XᵀY
score_b = X_b · u
```

## Original-priority tracking

Original V9.1_f targets are preserved with their original A/B orientation and marked:

```text
target_set = original_priority
```

Newly added targets are marked:

```text
target_set = added_all_pair
```

## New outputs

Under:

```text
outputs/bootstrap_composite_mca_audit_v9_1_f_all_pairs_a/cross_window/
```

New registry and audit files include:

```text
v9_1_f_all_pairs_target_registry.csv
v9_1_f_all_pairs_statistical_result_registry.csv
v9_1_f_all_pairs_multiple_testing_audit.csv
v9_1_f_all_pairs_original_priority_vs_added_summary.csv
v9_1_f_all_pairs_per_window_result_density.csv
v9_1_f_all_pairs_window_order_heterogeneity_summary.csv
v9_1_f_all_pairs_pair_summary.csv
v9_1_f_all_pairs_method_audit_summary.csv
V9_1_F_ALL_PAIRS_A_SUMMARY.md
```

## Interpretation boundary

This remains a bootstrap-space diagnostic. It does not identify direct-year physical year types and does not prove physical mechanisms. New all-pair results must be read with multiple-testing and target-coverage audits.

## Usage

```bat
python D:\easm_project01\stage_partition\V9_1\scripts\run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a.py
```

For a fast diagnostic run only:

```bat
set V9_1_F_ALL_PAIRS_BOOTSTRAP_N=300
set V9_1_F_ALL_PAIRS_PERM_N=100
set V9_1_F_ALL_PAIRS_MODE_STABILITY_N=50
set V9_1_F_ALL_PAIRS_SIGNFLIP_N=50
python D:\easm_project01\stage_partition\V9_1\scripts\run_bootstrap_composite_mca_audit_v9_1_f_all_pairs_a.py
```

Fast diagnostic outputs must not be mixed with formal full-run outputs.
