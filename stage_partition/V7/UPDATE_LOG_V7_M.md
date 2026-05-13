# UPDATE_LOG_V7_M

## w45_allfield_transition_marker_definition_audit_v7_m

Purpose:
- Adds a W45-only all-field transition marker definition audit.
- Keeps the current V7-e progress/profile representation fixed.
- Tests whether the former t(0.25) early-progress crossing is robust to alternative marker definitions.

Scope:
- Window: W002 / anchor_day = 45.
- Fields: P, V, H, Je, Jw.
- Inputs: V7-e progress timing outputs; if bootstrap curves are not present, W45 bootstrap progress curves are recomputed under the same V7-e representation.

New marker families:
- threshold sweep: t(0.10), t(0.15), ..., t(0.50), t(0.75)
- departure from pre-state background: departure90 / departure95
- progress-derivative peak day: raw and 3-day smoothed
- duration_25_75, tail_50_75, early_span_25_50

Outputs:
- input_audit_v7_m.json
- w45_allfield_marker_observed_v7_m.csv
- w45_allfield_marker_bootstrap_samples_v7_m.csv
- w45_allfield_marker_stability_summary_v7_m.csv
- w45_allfield_threshold_sweep_pairwise_v7_m.csv
- w45_allfield_departure_pairwise_v7_m.csv
- w45_allfield_peak_change_pairwise_v7_m.csv
- w45_allfield_marker_consistency_ledger_v7_m.csv
- w45_allfield_transition_type_classification_v7_m.csv
- w45_transition_marker_definition_window_summary_v7_m.csv
- w45_transition_marker_definition_audit_summary_v7_m.md
- run_meta.json

Interpretation restrictions:
- t25 is not treated as a physical onset by default.
- Threshold-only results are not robust transition order.
- Peak-change day is strongest-change timing, not onset or completion.
- Marker-inconsistent pairs must not be forced into a clean order.
- Not-distinguishable pairs must not be called synchronous without an equivalence test.
