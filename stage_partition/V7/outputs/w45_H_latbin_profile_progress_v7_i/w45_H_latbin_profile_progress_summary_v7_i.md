# W45 H lat-bin profile progress audit v7_i

## Purpose
This audit fixes the upstream implementation question: current 2-degree H features are produced by lon-mean plus latitude interpolation, not by true 2-degree latitude-bin averaging. V7-i tests a lat-bin-mean alternative before any further W45-H region aggregation.

## Scope
- Window: W002 / anchor day 45
- Field: H / z500 only
- Main implementation change: current np.interp 2-degree latitude feature -> true raw-lat bin mean after lon mean
- No causal/pathway interpretation; no statistical-threshold change.

## Unit-level comparison
- whole_field_H_current_interp_profile: n_units=1, median_q90_width=nan, n_unstable_units=nan, dominant_quality=monotonic_clear_progress
- single_2deg_interp_feature_v7_f: n_units=11, median_q90_width=14.0, n_unstable_units=10.0, dominant_quality=nonmonotonic_progress
- single_2deg_latbin_mean_feature_v7_i: n_units=11, median_q90_width=14.0, n_unstable_units=10.0, dominant_quality=nonmonotonic_progress
- latbin_vs_interp_verdict: latbin_mean_does_not_improve_stability

## Upstream implications
- current_2deg_profile_is_interpolation_not_lat_mean: confirmed_by_V6_state_builder_code — Previous V7-f/g/h feature-space regional diagnostics were based on interpolated 2-degree latitude points, not true 2-degree latitude-bin averages.
- latbin_mean_improves_stability: no_or_unclear — If yes, instability was partly caused by missing latitude-bin denoising at the 2-degree construction step.
- latbin_mean_is_not_enough: yes — If yes, W45-H instability is not mainly fixed by changing interpolation to latitude-bin mean; inspect progress definition/window/field behavior instead.
- interp_vs_latbin_numeric_difference: reported — Large differences imply that profile-construction choice can materially affect progress timing.

## Lat-bin timing classes
- unstable_latbin_feature: 10
- late_latbin_candidate: 1

## Input audit
- smoothed_fields_exists: True
- v7e_output_dir_exists: True
- v7f_output_dir_exists: True
- window_source: field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv
- implementation_change: latbin_mean_after_lon_mean_replaces_np_interp_lat_sampling_for_H_profile