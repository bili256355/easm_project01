# W45 H feature-level progress audit v7_f

## Scope
- Window: W002 / anchor day 45
- Field: H / z500 only
- Purpose: diagnose whether W45 H whole-field early-broad progress is caused by feature/latitude timing heterogeneity
- This run does not infer causality and does not upgrade W45 whole-field pairwise order evidence.

## Whole-field vs feature-level summary
- whole_observed_midpoint: 39.0
- whole_bootstrap_midpoint_median: 39.0
- feature_midpoint_min: 37.0
- feature_midpoint_max: 49.0
- feature_midpoint_median: 39.0
- feature_midpoint_iqr: 0.5
- n_early_features: 0
- n_middle_features: 0
- n_late_features: 1
- n_unstable_features: 10
- interpretation: feature_level_unstable: many H features have unstable or invalid progress timing; whole-field H early signal needs caution.

## Next-step decision
- is_worth_region_level_progress: False
- region_level_priority: low_until_quality_checked
- recommended_next_action: do_not_upgrade_W45_H_order; inspect_H_feature_progress_quality
- decision_reason: Many H features are unstable/invalid, so region-level progress should first diagnose quality rather than seek order.

## Field-feature timing classes
- unstable_feature: 10
- late_feature: 1

## Input audit
- smoothed_fields_exists: True
- v7e_output_dir_exists: True
- window_source: field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv