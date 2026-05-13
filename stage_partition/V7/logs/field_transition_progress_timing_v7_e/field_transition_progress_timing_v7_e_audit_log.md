# field_transition_progress_timing_v7_e audit log

created_utc: 2026-04-30T09:23:29.248634+00:00

## Scope
- Answers only the first question: field-level timing around accepted transition windows.
- Changes the observation target from change-intensity peaks/intervals to pre/post transition progress.
- Does not infer causality, does not analyze spatial earliest regions, and does not read downstream pathway or lead-lag outputs.

## Accepted windows
window_id  anchor_day anchor_month_day  accepted_window_start  accepted_window_end  analysis_window_start  analysis_window_end  pre_period_start  pre_period_end  post_period_start  post_period_end  analysis_radius_days nearby_excluded_candidate_days  n_nearby_excluded_candidates source_window_id                                                                                                 note
     W002          45            05-16                     40                   48                     30                   60                30              37                 53               60                    15                           none                             0             W002 V7-e progress window: accepted anchor +/- radius; excluded candidates flagged only, not reintroduced
     W003          81            06-21                     75                   86                     66                   96                66              73                 89               96                    15                             96                             1             W003 V7-e progress window: accepted anchor +/- radius; excluded candidates flagged only, not reintroduced
     W005         113            07-23                    107                  117                     98                  128                98             105                121              128                    15                             96                             1             W005 V7-e progress window: accepted anchor +/- radius; excluded candidates flagged only, not reintroduced
     W007         160            09-08                    155                  164                    145                  175               145             152                168              175                    15                           none                             0             W007 V7-e progress window: accepted anchor +/- radius; excluded candidates flagged only, not reintroduced

## Method notes
- analysis_radius_days = 15
- pre_period offsets = -15..-8
- post_period offsets = +8..+15
- n_bootstrap = 1000
- Progress is projection from pre prototype to post prototype, clipped to [0, 1].
- Excluded candidates are flagged if close but are not reintroduced as main windows.

## Run meta
{
  "status": "success",
  "created_utc": "2026-04-30T09:23:29.246680+00:00",
  "run_label": "field_transition_progress_timing_v7_e",
  "method": "pre_post_prototype_projection_progress_timing",
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
  "n_years": 45,
  "n_observed_rows": 20,
  "n_bootstrap_rows": 20000,
  "n_loyo_rows": 900,
  "downstream_lead_lag_included": false,
  "pathway_included": false,
  "spatial_earliest_region_included": false,
  "causal_interpretation_included": false,
  "notes": [
    "V7-e answers only field-level transition progress timing around accepted windows.",
    "Progress is projection from pre prototype to post prototype, not a causal path.",
    "18/96/132/135 are excluded from main windows and only flagged if nearby."
  ]
}