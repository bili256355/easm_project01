# V7-v W45 H/Jw baseline-sensitive state-growth framework

## 1. Purpose
This branch only tests H/Jw. It separates state progress from growth speed, tests C0/C1/C2 baseline sensitivity, and treats S_dist and S_pattern as parallel state-progress branches.

## 2. Baseline configurations
- `C0_full_stage`: pre=(0, 39), post=(49, 74), search=(35, 53), diagnostic=(0, 74). Use the full non-accepted-transition intervals before and after W45 as pre/post; use W45 ±5d expanded event-search window to avoid treating the accepted core as the full event search window.
- `C1_buffered_stage`: pre=(0, 34), post=(54, 69), search=(35, 53), diagnostic=(0, 74). Exclude five-day buffers around W45 and before W003/day81.
- `C2_immediate_pre`: pre=(25, 34), post=(54, 69), search=(35, 53), diagnostic=(0, 74). Use the near-W45 pre-core to test dependence on immediate background state.

## 3. State and growth definitions
- State distance branch: `S_dist = D_pre / (D_pre + D_post)`.
- State pattern branch: `S_pattern = (R_diff - R0) / (R1 - R0)`.
- Growth speed: first difference of each state branch, `V = dS/dt`, with centered 3-day mean for event detection.
- `P_proj` is reference only and is not used for main decisions.

## 4. Baseline sensitivity summary
- state / distance / S25_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- state / distance / S50_day: C0=`invalid_event`, C1=`invalid_event`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- state / distance / S75_day: C0=`invalid_event`, C1=`unresolved`, C2=`invalid_event`, sensitivity=`mixed_or_unresolved`
- state / distance / durable_S50_day: C0=`invalid_event`, C1=`invalid_event`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- state / distance / durable_S75_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- state / pattern / S25_day: C0=`invalid_event`, C1=`invalid_event`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- state / pattern / S50_day: C0=`unresolved`, C1=`unresolved`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- state / pattern / S75_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- state / pattern / durable_S50_day: C0=`unresolved`, C1=`unresolved`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- state / pattern / durable_S75_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / distance / growth_onset_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / distance / growth_peak_day: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / distance / growth_window_center: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / distance / growth_window_end: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / distance / growth_window_start: C0=`invalid_event`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`invalid_across_baselines`
- growth / pattern / growth_onset_day: C0=`unresolved`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`mixed_or_unresolved`
- growth / pattern / growth_peak_day: C0=`unresolved`, C1=`unresolved`, C2=`unresolved`, sensitivity=`mixed_or_unresolved`
- growth / pattern / growth_window_center: C0=`unresolved`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`mixed_or_unresolved`
- growth / pattern / growth_window_end: C0=`unresolved`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`mixed_or_unresolved`
- growth / pattern / growth_window_start: C0=`unresolved`, C1=`invalid_event`, C2=`invalid_event`, sensitivity=`mixed_or_unresolved`

## 5. Final H/Jw state-growth summary
- state_progress_order_distance_branch: distance=`mixed_or_unresolved`, pattern=`not_applicable`; next=review_state_order_sync_decision
- state_progress_order_pattern_branch: distance=`not_applicable`, pattern=`mixed_or_unresolved`; next=review_state_order_sync_decision
- growth_speed_order_distance_branch: distance=`invalid_across_baselines`, pattern=`not_applicable`; next=review_growth_order_sync_decision
- growth_speed_order_pattern_branch: distance=`not_applicable`, pattern=`mixed_or_unresolved`; next=review_growth_order_sync_decision

## 6. Method status
- state_framework: `method_unclosed`; evidence=hard_state_decisions=0; next=inspect_state_decisions
- growth_framework: `method_unclosed`; evidence=hard_growth_decisions=0; next=inspect_growth_decisions
- baseline_sensitivity: `not_stable`; evidence=stable_across_baselines=0; next=review_baseline_sensitivity_summary
- overall_method_status: `method_unclosed`; evidence=hard_state=0; hard_growth=0; stable=0; branch_conflict=False; next=do_not_extend_to_P_V_Je

## 7. Interpretation boundary
Distance-state, pattern-state, distance-growth speed, and pattern-growth speed are separate result layers. A growth-speed lead cannot be written as a state-progress lead. Invalid events cannot be written as unresolved or synchronous.

## 8. run_meta excerpt
```json
{
  "version": "v7_v",
  "hotfix_id": "v7_v_hotfix_02_unified_expanded_search",
  "output_tag": "w45_H_Jw_baseline_sensitive_state_growth_v7_v",
  "status": "success",
  "finished_at": "2026-05-03T23:30:40",
  "primary_goal": "baseline-sensitive H/Jw state-progress and growth-speed adjudication",
  "fields": [
    "H",
    "Jw"
  ],
  "window_id": "W002",
  "anchor_day": 45,
  "accepted_window": [
    40,
    48
  ],
  "baseline_configs": {
    "C0_full_stage": {
      "pre": [
        0,
        39
      ],
      "post": [
        49,
        74
      ],
      "search": [
        35,
        53
      ],
      "diagnostic": [
        0,
        74
      ]
    },
    "C1_buffered_stage": {
      "pre": [
        0,
        34
      ],
      "post": [
        54,
        69
      ],
      "search": [
        35,
        53
      ],
      "diagnostic": [
        0,
        74
      ]
    },
    "C2_immediate_pre": {
      "pre": [
        25,
        34
      ],
      "post": [
        54,
        69
      ],
      "search": [
        35,
        53
      ],
      "diagnostic": [
        0,
        74
      ]
    }
  },
  "state_branches": [
    "distance",
    "pattern"
  ],
  "growth_definition": "first_difference_of_state_progress_dS_dt",
  "P_proj_role": "reference_only_not_used_for_main_decision",
  "no_spatial_pairing": true,
  "no_latband_pairing": true,
  "no_P_V_Je": true,
  "event_validity_required_before_decision": true,
  "bootstrap_valid_fraction_minimum": 0.8,
  "n_bootstrap": 1000,
  "output_files": [
    "w45_H_Jw_baseline_config_table_v7_v.csv",
    "w45_H_Jw_baseline_sensitivity_summary_v7_v.csv",
    "w45_H_Jw_event_validity_table_v7_v.csv",
    "w45_H_Jw_growth_bootstrap_events_v7_v.csv",
    "w45_H_Jw_growth_event_registry_v7_v.csv",
    "w45_H_Jw_growth_observed_events_v7_v.csv",
    "w45_H_Jw_growth_order_sync_decision_v7_v.csv",
    "w45_H_Jw_growth_speed_curves_v7_v.csv",
    "w45_H_Jw_negative_growth_observed_events_v7_v.csv",
    "w45_H_Jw_state_bootstrap_events_v7_v.csv",
    "w45_H_Jw_state_event_registry_v7_v.csv",
    "w45_H_Jw_state_growth_final_summary_v7_v.csv",
    "w45_H_Jw_state_growth_method_status_v7_v.csv",
    "w45_H_Jw_state_observed_events_v7_v.csv",
    "w45_H_Jw_state_order_sync_decision_v7_v.csv",
    "w45_H_Jw_state_progress_curves_v7_v.csv"
  ]
}
```
