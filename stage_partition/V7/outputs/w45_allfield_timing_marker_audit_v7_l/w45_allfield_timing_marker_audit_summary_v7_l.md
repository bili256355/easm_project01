# W45 all-field timing-marker audit v7_l

## Purpose
Audit whether W45 P/V/H/Je/Jw should be compared by midpoint, onset, finish, or treated as marker-mixed / broad-transition objects.

## Window-level result
- window_timing_type: `onset_layer_with_H_early_broad_candidate`
- interpretation: W45 has robust onset-layer information for H, but H broad-transition is a candidate shape unless duration/finish-tail differences are statistically confirmed. Midpoint-order should not be the default interpretation.
- unique onset-best fields: 2
- marker-tie fields: 3
- H statistically confirmed broad transition: False

## Field recommended markers
- P: recommended_marker=`marker_tie`, selection=`tie`, tie_markers=`onset;midpoint;finish`, shape=`marker_tie_transition`, onset_q90=5.0, midpoint_q90=5.0, finish_q90=5.0
- V: recommended_marker=`marker_tie`, selection=`tie`, tie_markers=`onset;midpoint`, shape=`marker_tie_transition`, onset_q90=5.0, midpoint_q90=5.0, finish_q90=6.0
- H: recommended_marker=`onset`, selection=`unique_best`, tie_markers=`onset`, shape=`early_onset_broad_transition`, onset_q90=3.0, midpoint_q90=13.0, finish_q90=15.0
- Je: recommended_marker=`marker_tie`, selection=`tie`, tie_markers=`onset;midpoint`, shape=`marker_tie_transition`, onset_q90=5.0, midpoint_q90=5.0, finish_q90=7.0
- Jw: recommended_marker=`onset`, selection=`unique_best`, tie_markers=`onset`, shape=`early_onset_broad_transition`, onset_q90=5.0, midpoint_q90=8.0, finish_q90=8.0

## Onset-order results
- P < Je: confirmed_onset_order_90, median_delta=4.0, q05=0.9500000000000028, q95=7.0
- H < Je: confirmed_onset_order_95, median_delta=7.0, q05=3.0, q95=10.0

## Duration / finish-tail notes
- P-H: H_duration_longer_median;H_post_midpoint_tail_longer_median;H_finish_later_median; pass90_duration=False; pass90_tail=False
- V-H: H_duration_longer_median;H_post_midpoint_tail_longer_median;H_finish_later_median; pass90_duration=False; pass90_tail=False
- H-Je: H_duration_longer_median;H_post_midpoint_tail_longer_median; pass90_duration=False; pass90_tail=False
- H-Jw: H_duration_longer_median;H_post_midpoint_tail_longer_median;H_finish_later_median; pass90_duration=False; pass90_tail=False

## Field role reinterpretation
- P: marker_tie_unresolved; marker=marker_tie; usable_midpoint=False; usable_onset=False
- V: marker_tie_unresolved; marker=marker_tie; usable_midpoint=False; usable_onset=False
- H: early_onset_broad_transition_candidate; marker=onset; usable_midpoint=False; usable_onset=True
- Je: marker_tie_unresolved; marker=marker_tie; usable_midpoint=False; usable_onset=False
- Jw: onset_marker_candidate; marker=onset; usable_midpoint=False; usable_onset=True

## Prohibited interpretations
- Onset order is not full transition order.
- Early-onset does not imply causal upstream status.
- Broad-transition is not a statistical failure to be hidden.
- Midpoint instability must not be rewritten as synchrony without an equivalence test.
- Do not omit P/V/H/Je/Jw when reporting W45.
- Do not promote marker ties to onset or midpoint without explicitly reporting the tie.
- Observed broad-transition shape is not a statistically confirmed broadness difference unless the duration/tail test passes.
