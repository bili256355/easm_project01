# W45 H local latitude-band progress audit v7_g

## Purpose
- Diagnose whether W45/H single-lat feature instability is reduced by using sliding local 3-latitude bands.
- This is a W45/H-only implementation diagnostic and does not upgrade W45 order evidence by itself.
- It does not infer causality and does not alter V7-e/V7-e1/V7-e2 outputs.

## Lat-band construction
- band_size: 3
- band_step: 1
- n_latbands: 9

## Lat-band timing classes
- unstable_band: 7
- early_band_candidate: 1
- late_band_candidate: 1

## Upstream implication
- latband_improves_stability: partial | lat-band aggregation partially improves stability, but not enough to treat H regional timing as confirmed.
- single_feature_unit_too_noisy: yes | If confirmed, single-lat feature should not be the base unit for future feature/region-level progress.
- region_level_progress_recommended: conditional | lat-band aggregation partially improves stability, but not enough to treat H regional timing as confirmed.

## Recommended next step
- latband_improves_stability: partial
- recommended_next_action: review_latband_quality_before_generalizing_to_other_objects
- region_level_priority: diagnostic
- decision_reason: Lat-band aggregation partially improves single-feature instability, but evidence is still not ready for confirmed regional timing.

## Input audit
- smoothed_fields_exists: True
- v7e_output_dir_exists: True
- v7f_output_dir_exists: True
- window_source: field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv