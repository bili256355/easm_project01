# W45 H/Jw raw025 process audit V7-s

Created: 2026-05-03T16:27:38

## 1. Input audit

- Window: {'window_id': 'W002', 'anchor_day': 45, 'accepted_window_start': 40, 'accepted_window_end': 48, 'analysis_window_start': 30, 'analysis_window_end': 60, 'pre_period_start': 30, 'pre_period_end': 37, 'post_period_start': 53, 'post_period_end': 60, 'source': 'field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv'}
- H: key=z500_smoothed, shape=[45, 183, 81, 121], lat=15.000..35.000, lon=110.000..140.000
- Jw: key=u200_smoothed, shape=[45, 183, 81, 121], lat=25.000..45.000, lon=80.000..110.000

## 2. What is being tested

- H early-frontloaded
- H anchor-near retreat / non-monotonic segment
- Jw mid/late catch-up
- H/Jw progress crossing

## 3. Direct raw025 outputs

- H retreat candidate: {'retreat_detected': True, 'retreat_start_day': 43.0, 'retreat_end_day': 47.0, 'retreat_start_progress': 0.5374646581077257, 'retreat_end_progress': 0.4988286802363925, 'retreat_drop': -0.038635977871333216, 'retreat_span_days': 4, 'retreat_label': 'multi_day_retreat_candidate'}
- Crossing events:
  - day 35: Jw_to_H_progress_crossing
  - day 47: H_retreat_with_Jw_overtake
  - day 56: Jw_to_H_progress_crossing

## 4. Comparison with V7-p/q/r

- H early-frontloaded: raw025=yes; decision=robust_across_representation_candidate; evidence=H day35-43 progress change=0.4177
- H anchor retreat: raw025=yes; decision=raw025_supports_phenomenon; evidence=multi_day_retreat_candidate drop=-0.038635977871333216 start=43.0 end=47.0
- Jw mid_late_catchup: raw025=yes; decision=raw025_supports_phenomenon; evidence=Jw day43-53 progress change=0.5702
- H_Jw progress crossing: raw025=yes; decision=raw025_supports_phenomenon; evidence=Jw_to_H_progress_crossing;H_retreat_with_Jw_overtake;Jw_to_H_progress_crossing
- same_departure_candidate: raw025=yes; decision=robust_across_representation_candidate; evidence=departure90 Jw-H=0 days
- global_clean_order: raw025=no; decision=raw025_does_not_support; evidence=retreat/crossing evidence prevents clean global H-Jw order

## 5. Interpretation allowed

- Raw-field support / non-support for H/Jw process phenomena.
- Region-level candidate patterns when contribution maps and bootstrap stability support them.
- Representation sensitivity relative to 2-degree profile / feature diagnostics.

## 6. Interpretation forbidden

- Do not infer causal H→Jw or Jw→H from this audit.
- Do not call near-same departure synchrony without equivalence testing.
- Do not claim a global clean order if retreat/crossing/phase-specific evidence is present.
- Do not treat contribution-defined masks as physical regions without independent provenance and spatial-coherence checks.

## 7. Recommended next step

If raw025 supports spatially coherent H retreat and Jw catch-up, continue to a spatial interpretation audit. If it does not, close the V7-r H/Jw relation as representation-sensitive / feature-diagnostic only.
