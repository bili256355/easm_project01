# W45 profile order-test summary (hotfix06)

This summary is profile-only. It does not use 2D mirror outputs and does not change the detector.

## Key output tables

- timing_resolution_audit: 5 rows
- tau_sync_estimate: 1 rows
- pairwise_peak_order_test: 10 rows
- pairwise_synchrony_equivalence_test: 10 rows
- pairwise_window_overlap_test: 10 rows
- pairwise_state_progress_difference: 420 rows
- pairwise_state_catchup_reversal: 60 rows
- object_growth_sign_structure: 30 rows
- object_growth_pulse_structure: 30 rows
- pairwise_growth_process_difference: 360 rows
- pairwise_prepost_curve_interpretation: 10 rows
- pairwise_order_interpretation_summary: 10 rows

## Pairwise combined interpretations

- P-V: peak_indeterminate_A_state_growth_ahead | peak=A_peak_earlier_tendency | sync=synchrony_tendency | prepost=A_distance_state_or_growth_ahead
- P-H: peak_indeterminate_A_state_growth_ahead | peak=B_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- P-Je: peak_indeterminate_A_state_growth_ahead | peak=A_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- P-Jw: peak_indeterminate_A_state_growth_ahead | peak=B_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- V-H: peak_indeterminate_A_state_growth_ahead | peak=B_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- V-Je: peak_indeterminate_A_state_growth_ahead | peak=A_peak_earlier_tendency | sync=synchrony_tendency | prepost=A_distance_state_or_growth_ahead
- V-Jw: peak_indeterminate_A_state_growth_ahead | peak=B_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- H-Je: peak_indeterminate_A_state_growth_ahead | peak=A_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- H-Jw: peak_indeterminate_A_state_growth_ahead | peak=A_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
- Je-Jw: peak_indeterminate_A_state_growth_ahead | peak=B_peak_earlier_tendency | sync=synchrony_indeterminate | prepost=A_distance_state_or_growth_ahead
