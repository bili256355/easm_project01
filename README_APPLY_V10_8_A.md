# Apply patch: V10.8_a object-internal transition content audit

## Purpose

V10.8_a answers a narrow question:

> For each P/V/H/Je/Jw object-native breakpoint detected in V10.2, what object-internal content changed abruptly around that break day?

It does **not** rerun breakpoint detection and does **not** use joint windows, accepted windows, cross-object alignment, precursor inference, pathway interpretation, or causality.

## Files added

```text
stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py
stage_partition/V10/v10.8/src/stage_partition_v10_8/__init__.py
stage_partition/V10/v10.8/src/stage_partition_v10_8/config.py
stage_partition/V10/v10.8/src/stage_partition_v10_8/pipeline.py
```

The run also writes root logs when executed:

```text
ROOT_LOG_V10_8_A_OBJECT_INTERNAL_TRANSITION_CONTENT.md
root_log_append/ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN__V10_8_A_APPEND.md
root_log_append/ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS__V10_8_A_APPEND.md
```

## Input assumptions

The patch expects these existing project files:

```text
stage_partition/V10/v10.2/outputs/object_native_peak_discovery_v10_2/cross_object/object_native_derived_windows_all_objects_v10_2.csv
stage_partition/V10/v10.2/outputs/object_native_peak_discovery_v10_2/config_used.json
foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz
```

## Run

From the project root:

```bash
python stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py --progress
```

Optional overrides:

```bash
python stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py --progress --flank-half-width 10
python stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py --progress --no-figures
```

## Output directory

```text
stage_partition/V10/v10.8/outputs/object_internal_transition_content_v10_8_a/
```

Main tables:

```text
tables/input_field_audit_v10_8_a.csv
tables/object_window_inventory_v10_8_a.csv
tables/object_detector_native_change_decomposition_v10_8_a.csv
tables/object_breakpoint_feature_group_contribution_v10_8_a.csv
tables/object_semantic_metric_translation_v10_8_a.csv
tables/object_breakpoint_content_interpretation_v10_8_a.csv
tables/object_internal_breakpoint_content_sequence_v10_8_a.csv
tables/object_breakpoint_lat_profile_delta_v10_8_a.csv
tables/object_field_delta_summary_v10_8_a.csv
run_meta/run_meta_v10_8_a.json
summary_v10_8_a.md
```

## Interpretation boundary

Use `object_detector_native_change_decomposition_v10_8_a.csv` and `object_breakpoint_feature_group_contribution_v10_8_a.csv` as the direct diagnostic layer.

Use `object_breakpoint_content_interpretation_v10_8_a.csv` and `object_internal_breakpoint_content_sequence_v10_8_a.csv` as derived summaries only.

Do not write:

- any breakpoint is a precursor;
- any object causes another object;
- H35 affects W45;
- W33 connects to W45;
- the V10.8 labels are direct detector outputs.

V10.8_a is an object-internal breakpoint content audit only.
