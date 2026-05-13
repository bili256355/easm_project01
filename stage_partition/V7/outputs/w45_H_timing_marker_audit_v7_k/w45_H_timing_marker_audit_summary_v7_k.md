# W45-H timing marker audit v7_k

## Purpose
Assess whether W45-H should continue to be represented by midpoint timing, or whether it is better described as early-onset / broad-transition.

## Inputs
- V7-e whole-field H available: True
- V7-j raw025 three regions complete: True

## Window-level decision
- window_level_timing_type: `early_onset_broad_transition_candidate`
- whole_field_recommended_marker: `onset`
- recommended_next_action: `interpret_H_as_timing-shape_candidate_before_using_midpoint_order`

Onset is frequently the preferred marker; W45-H should be assessed as early-onset/broad-transition before any midpoint-order claim. Censoring risk is present in at least one unit; an extension audit may be needed before final interpretation.

## Per-unit marker decisions
- `whole_field_H`: recommended_marker=`onset`, decision=`prefer_onset_over_midpoint`. Onset is more stable than midpoint/finish by q90 width; this unit is better described as early-onset candidate than midpoint-order object.
- `R1_low`: recommended_marker=`onset`, decision=`prefer_onset_over_midpoint`. Onset is more stable than midpoint/finish by q90 width; this unit is better described as early-onset candidate than midpoint-order object.
- `R2_mid`: recommended_marker=`onset`, decision=`prefer_onset_over_midpoint`. Onset is more stable than midpoint/finish by q90 width; this unit is better described as early-onset candidate than midpoint-order object.
- `R3_high`: recommended_marker=`onset`, decision=`prefer_onset_over_midpoint`. Onset is more stable than midpoint/finish by q90 width; this unit is better described as early-onset candidate than midpoint-order object.

## Prohibited interpretations
- Do not treat order-not-resolved as synchrony; synchrony requires a separate equivalence test.
- Do not convert early-onset / broad-transition into confirmed field order.
- Do not infer causality or pathway direction from this timing-marker audit.
- Do not hide whole-field, R1_low, R2_mid, or R3_high; each unit remains part of the result state.