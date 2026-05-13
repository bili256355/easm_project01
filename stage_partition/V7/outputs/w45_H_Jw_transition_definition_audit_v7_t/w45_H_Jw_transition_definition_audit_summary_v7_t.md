# V7-t W45 H/Jw raw-field transition-definition audit

## 1. Purpose
This branch audits whether the previous pre→post projection progress creates artificial H retreat / H-Jw crossing or order ambiguity.
It does not introduce lat-band pairing, component labels, or complex relation labels as main results.

## 2. Compared transition definitions
- `P_proj`: weighted pre→post projection progress.
- `P_dist`: dual-distance progress using distance-to-pre and distance-to-post.
- `D_pre` / `D_post`: raw weighted distances to endpoint fields.
- `R_diff`: pattern correlation to post minus pattern correlation to pre.
- `orthogonal_ratio`: off-axis residual relative to the pre→post transition vector.
- `daily_cos_to_post_direction`: daily change direction relative to the pre→post vector.

## 3. H retreat adjudication
- Interval: day 43.0 to 47.0
- Interpretation: `mixed_retreat_and_reorganization`
- P_proj change: -0.038635977871333216
- P_dist change: -0.03140401544518567
- D_post change: -0.030513437376026076
- R_diff change: 0.002569068646786743
- Orthogonal ratio change: -0.10248366176573903

## 4. Order / synchrony by metric
Decision counts: `{'unresolved': 13}`

Selected decisions:
- Dpost_closest_to_post_day: unresolved (observed Δ=1.0)
- P_dist_t25: unresolved (observed Δ=3.0)
- P_dist_t50: unresolved (observed Δ=5.0)
- P_proj_t25: unresolved (observed Δ=4.0)
- P_proj_t50: unresolved (observed Δ=5.0)
- Rdiff_post_dominance_day: unresolved (observed Δ=-4.0)

## 5. Definition consistency
- H early lead: `unresolved` → unresolved
- H/Jw middle transition: `unresolved` → unresolved
- completion relation: `unresolved` → unresolved
- departure/initial post-likeness: `unresolved` → unresolved
- H anchor retreat: `mixed_retreat_and_reorganization` → unresolved_or_mixed

## 6. Continue / stop decision
- definition_conflict_count: 0 → continue_if_other_outputs_are_informative
- H_retreat_projection_risk: mixed_retreat_and_reorganization → projection_retreat_not_rejected_by_this_audit
- robust_across_definition_count: 0 → if_zero_and_no_other_increment_stop_this_line

## 7. Interpretation boundary
This branch adjudicates metric sensitivity. It does not prove causality, physical mechanism, or region-to-region correspondence between H and Jw.
If definitions conflict, the next step is transition-definition method redesign, not more spatial post-processing.

## 8. run_meta excerpt
```json
{
  "version": "v7_t",
  "output_tag": "w45_H_Jw_transition_definition_audit_v7_t",
  "status": "success",
  "created_at": "2026-05-03T18:04:52",
  "primary_goal": "audit whether pre_to_post_projection creates artificial progress retreat or order ambiguity",
  "main_input_representation": "raw025_smoothed_field",
  "smoothed_fields_path": "D:\\easm_project01\\foundation\\V1\\outputs\\baseline_a\\preprocess\\smoothed_fields.npz",
  "fields": [
    "H",
    "Jw"
  ],
  "window_id": "W002",
  "anchor_day": 45,
  "window": {
    "window_id": "W002",
    "anchor_day": 45,
    "accepted_window_start": 40,
    "accepted_window_end": 48,
    "analysis_window_start": 30,
    "analysis_window_end": 60,
    "pre_period_start": 30,
    "pre_period_end": 37,
    "post_period_start": 53,
    "post_period_end": 60,
    "source": "field_transition_progress_timing_v7_e/accepted_windows_used_v7_e.csv"
  },
  "n_bootstrap": 1000,
  "compared_transition_definitions": [
    "pre_to_post_projection_progress",
    "distance_to_pre_post_progress",
    "dual_distance_Dpre_Dpost",
    "pattern_correlation_to_pre_post",
    "orthogonal_residual_ratio",
    "daily_change_direction_cosine"
  ],
  "main_decision": "transition-definition sensitivity before further order/synchrony adjudication",
  "if_definition_sensitive": "return_to_transition_definition_method_design",
  "no_spatial_pairing": true,
  "no_latband_pairing": true,
  "no_complex_relation_labels": true,
  "synchrony_requires_positive_equivalence_test": true,
  "field_meta": {
    "H": {
      "transition_norm": 16.658942129382318,
      "projection_denom": 2452328.0484152203,
      "pre_mean": 5826.299369206776,
      "post_mean": 5838.658170290836,
      "field_key": "z500_smoothed",
      "n_years": 45,
      "n_days": 183,
      "lat_min": 15.0,
      "lat_max": 35.0,
      "lon_min": 110.0,
      "lon_max": 140.0,
      "n_lat": 81,
      "n_lon": 121
    },
    "Jw": {
      "transition_norm": 6.763920214605359,
      "projection_denom": 365400.7938608099,
      "pre_mean": 27.357801120717067,
      "post_mean": 26.597275247384992,
      "field_key": "u200_smoothed",
      "n_years": 45,
      "n_days": 183,
      "lat_min": 25.0,
      "lat_max": 45.0,
      "lon_min": 80.0,
      "lon_max": 110.0,
      "n_lat": 81,
      "n_lon": 121
    }
  },
  "key_outputs": [
    "w45_H_Jw_transition_metric_curves_v7_t.csv",
    "w45_H_retreat_definition_adjudication_v7_t.csv",
    "w45_H_Jw_order_sync_by_metric_v7_t.csv",
    "w45_H_Jw_definition_consistency_summary_v7_t.csv",
    "w45_H_Jw_method_failure_or_continue_v7_t.csv"
  ]
}
```
