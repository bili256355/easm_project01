# V10.4 Object-to-Object Order Sensitivity Summary

## Purpose
V10.4 audits object-to-object peak timing order near each joint lineage / accepted window. It reads V10.1, V10.2, and V10.3 outputs and does not rerun peak discovery.

## Run status
- status: success
- n_lineages: 8
- n_configs: 40
- n_assignment_rows: 1600
- n_pairwise_rows: 3200
- n_reversal_rows: None

## Key counts
- n_lineages: 8
- n_configs: 40
- n_assignment_rows: 1600
- n_pairwise_rows: 3200
- n_pairwise_summary_rows: 1120
- n_sequence_rows: 320
- n_reversal_rows_tau2: 0
- pairwise_order_stability_class_counts_tau2: {'ORDER_STABLE': 726, 'ORDER_NEAR_TIE_DOMINATED': 380, 'ORDER_CONFIG_SENSITIVE': 14}
- n_missing_assignments: 5
- n_candidate_reassignments: 0
- n_sequence_changed_from_reference_tau2: 39

## Interpretation boundary
This is a timing/order stability audit only. It does not establish causality, physical mechanism, or whether a non-strict candidate should enter the main result.
