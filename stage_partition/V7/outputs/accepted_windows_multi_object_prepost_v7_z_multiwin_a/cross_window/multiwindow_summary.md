# V7-z-multiwin-a summary

Accepted windows processed: 1
Accepted windows available: 4
run_mode: w45; targets: W045,45; run_2d: False

## Window scopes processed
- W045: system day40-48, detector day0-69, analysis day0-74

## Final claims by window
### W045
- H-Je: co_transition_with_B_curve_tendency (Level2_curve_tendency_only)
- H-Jw: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- H-P: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- H-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- Je-Jw: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- Je-P: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- Je-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- Jw-P: co_transition_with_B_curve_tendency (Level2_curve_tendency_only)
- Jw-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)
- P-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only)

## Method boundary
- system_window is the accepted core band, not detector search range.
- raw/profile object-window detector uses detector_search_range.
- profile and 2D pre-post metrics use analysis_range and C0/C1/C2 baselines.
- 2D mirror does not rediscover windows and does not rewrite final claims by itself.