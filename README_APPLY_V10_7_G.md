# V10.7_g patch: W45 multisource method-control audit

## Purpose

This patch adds a new V10.7_g entry that tests whether the W45 cross-object incremental-audit method can detect signals from multiple source objects, not only H.

It fixes the Je/Jw input issue by deriving both from the shared `u200` field:

- Je: `u200`, lon 120–150E, lat 25–45N, jet q90-strength reducer
- Jw: `u200`, lon 80–110E, lat 25–45N, jet q90-strength reducer

The version is intended to answer:

1. Is the H negative result meaningful, or is the current method unable to detect cross-object signals in general?
2. Which source object packages show yearwise incremental association with W45 main-cluster targets?

## Apply

Copy the `stage_partition` folder in this patch into the project root:

```text
D:\easm_project01
```

This adds:

```text
stage_partition\V10\v10.7\scripts\run_w45_multisource_method_control_v10_7_g.py
stage_partition\V10\v10.7\src\stage_partition_v10_7\w45_multisource_method_control_pipeline.py
```

## Run

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_multisource_method_control_v10_7_g.py
```

## Outputs

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_multisource_method_control_v10_7_g
```

Core tables:

```text
tables\w45_multisource_input_audit_v10_7_g.csv
tables\w45_multisource_yearwise_strength_v10_7_g.csv
tables\w45_multisource_incremental_explanatory_power_v10_7_g.csv
tables\w45_method_control_decision_v10_7_g.csv
tables\w45_object_route_decision_v10_7_g.csv
```

Core figures:

```text
figures\w45_multisource_delta_r2_top_cross_targets_v10_7_g.png
figures\w45_multisource_delta_r2_heatmap_v10_7_g.png
```

## Interpretation boundary

- Self-target results are diagnostic only.
- Cross-target results are the main evidence.
- Positive results are yearwise incremental associations, not causality.
- If non-H sources show cross-object support while H does not, the H negative result becomes more meaningful.
- If no source shows cross-object support, the method/metric/window design may be low-power or unsuitable.
