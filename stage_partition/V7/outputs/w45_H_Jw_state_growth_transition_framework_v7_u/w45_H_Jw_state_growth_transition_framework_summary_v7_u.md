# V7-u W45 H/Jw state–growth separated transition framework

## 1. Purpose
This branch separates state-progress information from rapid-growth information. It uses only H/Jw and does not add P/V/Je.

## 2. What changed from V7-t
- `P_proj` is downgraded to `P_proj_reference`.
- Main state decisions are based on endpoint-distance and pattern-likeness.
- Main growth decisions are based on raw-field daily change and postward growth metrics.
- State lead and growth lead are not collapsed into one transition order.

## 3. State order/synchrony decisions
Hotfix note: departure events are searched only after the pre-period; `durable_departure_from_pre_3d` is the main departure event, while non-durable `departure_from_pre` is candidate-only.

- departure_from_pre: `synchronous_equivalent` / `method_unclosed`; Δmedian=0.0; P_H=0.014; P_Jw=0.008
- durable_departure_from_pre_3d: `synchronous_equivalent` / `hard_decision`; Δmedian=0.0; P_H=0.014; P_Jw=0.01
- post_dominance_day: `unresolved` / `method_unclosed`; Δmedian=-2.0; P_H=0.202; P_Jw=0.555
- durable_post_dominance_2d: `unresolved` / `method_unclosed`; Δmedian=-2.0; P_H=0.186; P_Jw=0.57
- durable_post_dominance_3d: `unresolved` / `method_unclosed`; Δmedian=-2.0; P_H=0.178; P_Jw=0.578
- durable_post_dominance_4d: `unresolved` / `method_unclosed`; Δmedian=-2.0; P_H=0.171; P_Jw=0.581

## 4. Growth order/synchrony decisions
- max_growth_day: `unresolved` / `directional_tendency_only`; Δmedian=9.0; P_H=0.696; P_Jw=0.199
- postward_growth_peak_day: `unresolved` / `method_unclosed`; Δmedian=3.0; P_H=0.611; P_Jw=0.179
- rapid_growth_start: `unresolved` / `directional_tendency_only`; Δmedian=10.0; P_H=0.736; P_Jw=0.186
- rapid_growth_center: `unresolved` / `directional_tendency_only`; Δmedian=10.604410076128811; P_H=0.739; P_Jw=0.213
- rapid_growth_end: `unresolved` / `directional_tendency_only`; Δmedian=11.0; P_H=0.729; P_Jw=0.192

## 5. Integrated state–growth reading
- who_leaves_pre_first: state=`synchronous_equivalent (hard_decision; Δ=0.0)`, growth=`not_a_growth_question`, status=`hard_decision_available`
- who_becomes_post_like_first: state=`unresolved (method_unclosed; Δ=-2.0)`, growth=`not_a_growth_question`, status=`method_unclosed`
- who_becomes_durably_post_like_first: state=`unresolved (method_unclosed; Δ=-2.0)`, growth=`not_a_growth_question`, status=`method_unclosed`
- who_has_earlier_max_growth: state=`not_a_state_question`, growth=`unresolved (directional_tendency_only; Δ=9.0)`, status=`tendency_only`
- who_has_earlier_postward_growth: state=`not_a_state_question`, growth=`unresolved (method_unclosed; Δ=3.0)`, status=`method_unclosed`
- whether_state_and_growth_order_agree: state=`hard_state_decisions=1`, growth=`hard_growth_decisions=0`, status=`usable_for_comparison`

## 6. Method status
- state_framework_status: `usable_for_H_Jw`; next=extend_state_framework_after_review
- growth_framework_status: `partially_usable_tendency_only`; next=do_not_extend_growth_framework_yet
- projection_method_status: `reference_only`; next=do_not_use_projection_as_main_decision
- overall_method_status: `partially_usable`; next=only_extend_the_layer_with_hard_decisions

## 7. Interpretation boundary
State-progress lead and growth-rate lead are separate. A growth decision must not be written as a state-transition order. Projection-only tendencies remain downgraded unless state/growth decisions support them.

## 8. run_meta excerpt
```json
{
  "version": "v7_u",
  "hotfix_id": "v7_u_hotfix_01_departure_after_pre",
  "output_tag": "w45_H_Jw_state_growth_transition_framework_v7_u",
  "status": "success",
  "created_at": "2026-05-03T19:09:19",
  "primary_goal": "separate state-progress information from rapid-growth information for H/Jw only",
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
  "state_metrics": [
    "D_pre",
    "D_post",
    "P_dist",
    "R_pre",
    "R_post",
    "R_diff"
  ],
  "growth_metrics": [
    "field_change_norm",
    "delta_P_dist",
    "delta_R_diff"
  ],
  "P_proj_role": "reference_only_not_main_decision",
  "state_events": [
    "departure_from_pre",
    "durable_departure_from_pre_3d",
    "post_dominance_day",
    "durable_post_dominance_2d",
    "durable_post_dominance_3d",
    "durable_post_dominance_4d"
  ],
  "growth_events": [
    "max_growth_day",
    "postward_growth_peak_day",
    "rapid_growth_start",
    "rapid_growth_center",
    "rapid_growth_end"
  ],
  "synchrony_requires_positive_equivalence_test": true,
  "order_margin_days": 1.0,
  "hard_decision_probability_threshold": 0.9,
  "no_spatial_pairing": true,
  "no_latband_pairing": true,
  "no_complex_relation_labels": true,
  "extension_rule": "Do not add P/V/Je until H/Jw state-growth framework is reviewed.",
  "departure_hotfix_rule": "departure search starts after pre_period_end; durable_departure_from_pre_3d is the main departure event; non-durable departure_from_pre is candidate-only",
  "field_meta": {
    "H": {
      "transition_norm": 16.658942129382318,
      "projection_denom": 2452328.0484152203,
      "pre_mean": 5826.299369206776,
      "post_mean": 5838.658170290836,
      "pre_days": [
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37
      ],
      "post_days": [
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60
      ],
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
      "pre_days": [
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37
      ],
      "post_days": [
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60
      ],
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
    "w45_H_Jw_state_metric_curves_v7_u.csv",
    "w45_H_Jw_growth_metric_curves_v7_u.csv",
    "w45_H_Jw_state_order_sync_decision_v7_u.csv",
    "w45_H_Jw_growth_order_sync_decision_v7_u.csv",
    "w45_H_Jw_projection_vs_state_growth_comparison_v7_u.csv",
    "w45_H_Jw_state_growth_method_status_v7_u.csv"
  ]
}
```
