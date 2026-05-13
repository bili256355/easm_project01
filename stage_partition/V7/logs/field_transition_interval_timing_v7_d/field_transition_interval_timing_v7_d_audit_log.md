# field_transition_interval_timing_v7_d audit log

created_utc: 2026-04-30T08:40:21.480481+00:00

## Scope
- Answers only the first question: for each accepted transition window, when each field is most actively transitioning.
- Changes the observation target from argmax peak_day to active intervals: onset / center / end / duration.
- Does not infer causality, does not analyze spatial earliest regions, and does not read downstream pathway or lead-lag outputs.

## Accepted windows
window_id  anchor_day anchor_month_day  accepted_window_start  accepted_window_end  analysis_window_start  analysis_window_end  analysis_radius_days nearby_excluded_candidate_days  n_nearby_excluded_candidates source_window_id                                                                                                          note
     W002          45            05-16                     40                   48                     30                   60                    15                           none                             0             W002 V7-d interval analysis window: accepted anchor +/- radius; excluded candidates only flagged, not reintroduced
     W003          81            06-21                     75                   86                     66                   96                    15                             96                             1             W003 V7-d interval analysis window: accepted anchor +/- radius; excluded candidates only flagged, not reintroduced
     W005         113            07-23                    107                  117                     98                  128                    15                             96                             1             W005 V7-d interval analysis window: accepted anchor +/- radius; excluded candidates only flagged, not reintroduced
     W007         160            09-08                    155                  164                    145                  175                    15                           none                             0             W007 V7-d interval analysis window: accepted anchor +/- radius; excluded candidates only flagged, not reintroduced

## Method notes
- analysis_radius_days = 15
- prepost_k_days = 5
- n_bootstrap = 1000
- detector_score is the main metric inherited from stage_partition; prepost_contrast is an independent diagnostic metric.
- excluded candidates are flagged if close to an active interval but are not reintroduced as main windows.

## Run meta
{
  "status": "success",
  "created_utc": "2026-04-30T08:40:21.478213+00:00",
  "run_label": "field_transition_interval_timing_v7_d",
  "accepted_peak_days": [
    45,
    81,
    113,
    160
  ],
  "excluded_candidate_days": [
    18,
    96,
    132,
    135
  ],
  "n_windows": 4,
  "n_bootstrap": 1000,
  "n_interval_observed_rows": 40,
  "n_bootstrap_rows": 40000,
  "n_loyo_rows": 1800,
  "window_class_counts": {
    "weak_partial_order_window": 3,
    "boundary_limited_window": 1
  },
  "pair_label_counts_detector_score": {
    "ambiguous_interval_order": 17,
    "moderate_shifted_order": 10,
    "boundary_limited_interval_order": 7,
    "shifted_overlap_order": 3,
    "sync_or_overlap": 2,
    "separated_order": 1
  },
  "method": "active_interval_timing_from_detector_score_and_prepost_contrast; timing order only, not causality",
  "input_smoothed_fields_path": "D:\\easm_project01\\foundation\\V1\\outputs\\baseline_a\\preprocess\\smoothed_fields.npz",
  "source_logs_checked": [
    "D:\\easm_project01\\stage_partition\\V6\\UPDATE_LOG_V6.md",
    "D:\\easm_project01\\stage_partition\\V6_1\\UPDATE_LOG_V6_1.md"
  ],
  "source_tables_checked": [
    "D:\\easm_project01\\stage_partition\\V6\\outputs\\mainline_v6_a\\candidate_points_bootstrap_summary.csv",
    "D:\\easm_project01\\stage_partition\\V6_1\\outputs\\mainline_v6_1_a\\derived_windows_registry.csv"
  ],
  "notes": [
    "V7-d is an experiment-level change, not a result-picking layer.",
    "It replaces single-day argmax peak timing with transition active intervals.",
    "Detector score is the main inherited metric; pre/post contrast is an independent diagnostic metric.",
    "No downstream lead-lag/pathway outputs are used."
  ]
}