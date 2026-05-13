# V7-z W45 multi-object pre-post statistical validation

Run time: 2026-05-05T14:17:07
Window: W002, accepted_window=day40–48, anchor_day=45

## Scope
Fixed W45; objects P/V/H/Je/Jw; profile-object and shape-pattern detectors; C0/C1/C2 pre-post curves; paired year bootstrap; evidence gate.

## Final claims
- **P-V: co_transition_with_A_curve_tendency** — Level2_curve_tendency_only; window_cotransition_curve_tendency
- **P-H: B_leads_A_candidate** — Level4_window_and_curve_supported; window_and_curve_B
- **P-Je: B_layer_specific_lead_or_front** — Level3_single_family_supported; layer_specific_with_curve_support
- **V-H: B_leads_A_candidate** — Level4_window_and_curve_supported; window_and_curve_B
- **V-Je: B_layer_specific_lead_or_front** — Level3_single_family_supported; layer_specific_with_curve_support
- **V-Jw: B_leads_A_candidate** — Level4_window_and_curve_supported; window_and_curve_B
- **H-Je: detector_split** — branch_or_detector_specific; detector_conflict
- **H-Jw: A_layer_specific_lead_or_front** — Level3_single_family_supported; layer_specific_with_curve_support
- **Je-Jw: detector_split** — branch_or_detector_specific; detector_conflict

## Timing structure classifications
- P-V: co_transition_with_A_curve_tendency (Level2_curve_tendency_only; window_cotransition_curve_tendency)
- P-H: B_leads_A_candidate (Level4_window_and_curve_supported; window_and_curve_B)
- P-Je: B_layer_specific_lead_or_front (Level3_single_family_supported; layer_specific_with_curve_support)
- P-Jw: A_curve_tendency_only (Level2_curve_tendency_only; curve_only)
- V-H: B_leads_A_candidate (Level4_window_and_curve_supported; window_and_curve_B)
- V-Je: B_layer_specific_lead_or_front (Level3_single_family_supported; layer_specific_with_curve_support)
- V-Jw: B_leads_A_candidate (Level4_window_and_curve_supported; window_and_curve_B)
- H-Je: detector_split (branch_or_detector_specific; detector_conflict)
- H-Jw: A_layer_specific_lead_or_front (Level3_single_family_supported; layer_specific_with_curve_support)
- Je-Jw: detector_split (branch_or_detector_specific; detector_conflict)

## Downgraded signals
Number of downgraded signals: 1

## Forbidden interpretations
- Do not infer causality/pathway from V7-z timing structure.
- Do not collapse branch-split or baseline-sensitive results into a single winner.
- Do not treat candidate_window support <0.95 as accepted_window.