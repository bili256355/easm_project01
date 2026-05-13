# V7-c pairwise field-order audit log

## Scope

V7-c reads V7-b bootstrap/LOYO peak-day samples and audits pairwise relative timing order between fields.
It does not rerun detector, does not infer causality, and does not analyze spatial earliest regions.

## Input evidence checked

V7-b root: D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b

Required files:
- bootstrap_samples: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\field_transition_peak_days_bootstrap_samples_v7_b.csv
- loyo_samples: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\field_transition_peak_days_loyo_samples_v7_b.csv
- main_table: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\field_transition_peak_days_bootstrap_v7_b.csv
- accepted_windows: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\accepted_windows_used_v7_b.csv
- run_meta: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\run_meta.json
- summary: exists=True path=D:\easm_project01\stage_partition\V7\outputs\field_transition_timing_v7_b\summary.json

## V7-b status evidence

run_meta.status: success
run_meta.accepted_peak_days: [45, 81, 113, 160]
run_meta.excluded_candidate_days: [18, 96, 132, 135]
summary.timing_confidence_counts: {'timing_uncertain': 11, 'boundary_truncated': 6, 'timing_moderate': 3}

## V7-c interpretation rules

- robust_order / moderate_order mean relative timing order, not causality.
- robust_censored_order means direction is supported but exact peak day is boundary-censored.
- sync_or_overlap means two fields are too close in timing to force order.
- ambiguous_order means insufficient pairwise support.

Logs and V7-b run metadata are evidence layers, not automatic truth; they are cross-checked with required output tables.