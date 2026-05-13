# state_coordinate_meaning_audit_v8_a summary

## Purpose
This audit checks whether V8 pre/post state curves can be interpreted as object-internal post-likeness, as common seasonal-process diagnostics, or as calibrated cross-object state-progress coordinates.

## Input status
- W045:object_state_curves: found
- W045:state_valid_day_audit: found
- W045:state_profile_reference_validity_audit: found
- W045:state_reproducibility_null_segment_summary: found
- W045:pairwise_state_relation_blocks: found
- W045:pairwise_state_block_bootstrap: found
- W045:main_window_selection: found
- W045:pairwise_peak_order_test: found
- W045:pairwise_synchrony_equivalence_test: found

## Object-internal coordinate validity
- caution_object_internal_coordinate: 25
- valid_object_internal_coordinate: 5

## Common seasonal process audit
- common_process_consistent: 30

## Pairwise scale comparability
- calibrated_time_to_q_progress_comparison_allowed: 36
- common_process_diagnostic_raw_deltaS_not_interpretable: 24
Interpretation levels:
- level 2: 24
- level 3: 36

## Existing state-relation block interpretation gate
- level 2: 87 blocks
- level 3: 108 blocks

## Interpretation rule
Unless a pair reaches a calibrated interpretation level, raw ΔS blocks must be described only as object-specific relative post-likeness diagnostics. They must not be written as lead/lag, synchrony, catch-up, or causality.