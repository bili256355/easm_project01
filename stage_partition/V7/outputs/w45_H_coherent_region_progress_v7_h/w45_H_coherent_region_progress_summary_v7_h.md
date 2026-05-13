# W45 H transition-coherent region progress audit v7_h

## Purpose
- Diagnose whether W45/H whole-field early-broad behavior can be explained by transition-coherent latitude regions.
- Regions are built from contiguous H latitude features with the same bootstrap 90% dH sign class.
- This is a W45/H-only implementation diagnostic. It does not infer causality or modify V7-e/V7-e1/V7-e2 outputs.

## Transition-vector classes
- stable_positive_change: 7
- stable_negative_change: 2
- ambiguous_change: 2

## Coherent regions
- H_region_0_stable_negative_change_15_17: lat 15.0–17.0, class=stable_negative_change, n_features=2
- H_region_1_ambiguous_change_19_21: lat 19.0–21.0, class=ambiguous_change, n_features=2
- H_region_2_stable_positive_change_23_35: lat 23.0–35.0, class=stable_positive_change, n_features=7

## Unit comparison
- whole_field_H: n=1, median_q90_width=14.0, unstable=0
- single_lat_feature: n=11, median_q90_width=14.0, unstable=10
- local_3_lat_band: n=9, median_q90_width=14.0, unstable=7
- coherent_region: n=3, median_q90_width=17.0, unstable=3

## Upstream implication
- coherent_region_improves_stability: partial | transition-coherent region partially improves stability, but not enough for automatic promotion.
- transition_vector_has_stable_structure: yes | Region construction has data support only where dH sign is stable under bootstrap.
- upstream_analysis_unit_should_be_revised: yes | If coherent units improve stability, future P/V/H/Je/Jw region progress should not default to single features or mechanical sliding bands.

## Interpretation rule
- Stable dH regions and improved region progress can support revising future region-level progress units.
- If coherent regions remain unstable, W45-H should remain an early-broad candidate, not a confirmed regional timing result.