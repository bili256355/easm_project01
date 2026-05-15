# ROOT LOG: V10.8_a object-internal transition content audit

Created: 2026-05-15T07:27:14.719524+00:00

## Task type
Engineering + research diagnostic patch.

## Version
`stage_partition/V10/v10.8`

## Entry
`stage_partition/V10/v10.8/scripts/run_object_internal_transition_content_v10_8_a.py`

## Output
`D:\easm_project01\stage_partition\V10\v10.8\outputs\object_internal_transition_content_v10_8_a`

## Input baseline
- V10.2 object-native breakpoint windows: `D:\easm_project01\stage_partition\V10\v10.2\outputs\object_native_peak_discovery_v10_2\cross_object\object_native_derived_windows_all_objects_v10_2.csv`
- V10.2 config: `D:\easm_project01\stage_partition\V10\v10.2\outputs\object_native_peak_discovery_v10_2\config_used.json`
- Smoothed fields: `D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz`

## Scope
V10.8_a answers: for each P/V/H/Je/Jw object-native break day from V10.2, what object-internal detector-content changed most strongly?

Number of V10.2 object-native windows processed: 39

## Method boundary
- Does not rerun breakpoint detection.
- Does not use joint windows.
- Does not align to W045/W081/W113/W160.
- Does not infer synchrony, precursor, pathway, or causality.
- Reconstructs detector left/right flanks using V10.2 detector width and decomposes the reconstructed state-vector change.

## Main tables
- `tables/object_detector_native_change_decomposition_v10_8_a.csv`
- `tables/object_breakpoint_feature_group_contribution_v10_8_a.csv`
- `tables/object_semantic_metric_translation_v10_8_a.csv`
- `tables/object_breakpoint_content_interpretation_v10_8_a.csv`
- `tables/object_internal_breakpoint_content_sequence_v10_8_a.csv`
- `tables/object_breakpoint_lat_profile_delta_v10_8_a.csv`
- `tables/object_field_delta_summary_v10_8_a.csv`

## Forbidden interpretation
- Do not call any object breakpoint a precursor based on V10.8_a alone.
- Do not infer cross-object influence.
- Do not infer H35 -> W45, W33 -> W45, or any causal pathway.
- Do not treat derived content labels as direct detector outputs.
