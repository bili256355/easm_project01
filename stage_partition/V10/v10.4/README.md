# V10.4 object_order_sensitivity_v10_4

This submodule audits **object-to-object peak timing order sensitivity** near each joint lineage / accepted window.

It reads existing outputs from:

- `V10/v10.1` joint main-window lineage
- `V10/v10.2` object-native peak catalog and mapping
- `V10/v10.3` peak discovery sensitivity results

It does **not** rerun peak discovery, bootstrap, detector scoring, band construction, or window merging. It only performs candidate assignment and object-to-object order comparison.

## Run

```bat
python D:\easm_project01\stage_partition\V10\v10.4\scripts\run_object_order_sensitivity_v10_4.py
```

## Main outputs

```text
V10\v10.4\outputs\object_order_sensitivity_v10_4\assignment\object_candidate_assignment_by_lineage_config_v10_4.csv
V10\v10.4\outputs\object_order_sensitivity_v10_4\order\object_pairwise_order_by_lineage_config_v10_4.csv
V10\v10.4\outputs\object_order_sensitivity_v10_4\order\object_pairwise_order_stability_summary_v10_4.csv
V10\v10.4\outputs\object_order_sensitivity_v10_4\order\object_order_sequence_by_lineage_config_v10_4.csv
V10\v10.4\outputs\object_order_sensitivity_v10_4\order\object_order_reversal_inventory_v10_4.csv
```

## Boundary

This is a method-layer timing/order stability audit. It does not establish causality, physical mechanism, or whether a non-strict candidate should enter the main result.
