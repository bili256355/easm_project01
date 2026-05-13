# Je raw-supported early signal audit v7_z_b

## Purpose
Test whether Je day26–33 is supported by raw-profile evidence, or only by the normalization-sensitive shape-pattern detector.

## Final decision
- `Je_day26_33_status`: `raw_supported_weak_early_peak`

## Key checks
- day26_or_day33_raw_before_after_support: `early_raw_change_supported` — day26 raw_l2=supported_positive_change; day33 raw_l2=supported_positive_change
- day46_raw_main_support: `late_raw_main_supported` — day46 raw_l2=supported_positive_change
- raw_feature_early_support: `feature_early_support_present` — number of direct raw features with peak in day26-33
- early_window_detector_support: `early_detector_reproduced` — early-only / late-excluded raw detector bootstrap
- peak_vs_ramp_structure: `weak_peak_like` — cumulative raw change + detector/feature peak audit
- previous_normalization_risk: `previous_audit_flagged_normalization_sensitivity` — V7-z-je-audit-a decision table
- final_Je_day26_33_status: `raw_supported_weak_early_peak` — combined raw before-after + feature + detector + ramp + audit-a sensitivity

## Before-after bootstrap summary
- day26_raw_early_candidate raw_l2_change: median=3.5629965138841366, q025=2.610625662766635, q975=4.76759744245676, decision=supported_positive_change
- day33_shape_sensitive_candidate raw_l2_change: median=3.675131881561197, q025=2.174234571883968, q975=5.152967551216048, decision=supported_positive_change
- day46_raw_main_peak raw_l2_change: median=2.34099817668029, q025=1.6282309370420436, q975=3.457329530901489, decision=supported_positive_change

## Feature peak audit
- Raw features peaking in day26–33: 6
  - strength_mean: early peak day 28, early/late ratio=1.8840448167881678
  - strength_max: early peak day 28, early/late ratio=2.120225827494726
  - amplitude: early peak day 31, early/late ratio=1.165846109956979
  - weighted_l2_norm: early peak day 28, early/late ratio=2.056604315401072
  - axis_lat: early peak day 32, early/late ratio=1.0
  - centroid_lat: early peak day 32, early/late ratio=1.1832085902401055

## Detector bootstrap
- early_only_day0_40: P_has_any_early_peak_day26_33=0.8866666666666667, decision=early_raw_peak_reproduced
- full_day0_70: P_has_any_early_peak_day26_33=0.85, decision=early_raw_peak_reproduced
- late_excluded_day0_39: P_has_any_early_peak_day26_33=0.8866666666666667, decision=early_raw_peak_reproduced

## Allowed statement
Je has a raw-profile-supported weak early structural peak around day26–33, although it remains weaker than the day46 raw-profile main adjustment.

## Forbidden statement
- Do not write that Je day33 is a confirmed early physical transition unless raw-supported weak peak criteria are met.
- Do not use this audit to infer Je causal influence on other objects.
