# W45 all-field process relation layer V7-n

## Purpose
This is an implementation-layer audit. It does not try to force W45 into a clean field-order chain. It represents each field as a pre-to-post progress process and each pair as a curve/phase/marker relation object.

## Input representation
- Input representation: current_v7e_progress_profile_via_v7m_curves
- Fields: P, V, H, Je, Jw. All five fields participate in calculations.
- Progress is relative pre-to-post progress, not physical magnitude and not causality.

## Field process quality
- P: compact_transition / midpoint_or_peak_may_be_usable — compact transition candidate; midpoint may be representative.
- V: compact_transition / midpoint_or_peak_may_be_usable — compact transition candidate; midpoint may be representative.
- H: early_departure_broad_finish / marker_family_required — early progress is more stable than midpoint/finish and finish tail is broader.
- Je: compact_transition / midpoint_or_peak_may_be_usable — compact transition candidate; midpoint may be representative.
- Jw: broad_or_marker_mixed_transition / marker_family_required — monotonic but no single marker is clearly sufficient.

## Pairwise relation types
- P-V: weak_lead_tendency — Only tendency-level marker support; no 90% directional order.
- P-H: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- P-Je: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- P-Jw: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- V-H: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- V-Je: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- V-Jw: weak_lead_tendency — Only tendency-level marker support; no 90% directional order.
- H-Je: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.
- H-Jw: front_loaded_vs_catchup — Curve-level evidence suggests phase-dependent relation: early-progress advantage and later catch-up; do not reduce to a single lead.
- Je-Jw: phase_crossing_relation — Curve relation changes sign across phases; single order is not adequate.

## H/Jw focus
H/Jw relation type: front_loaded_vs_catchup
Interpretation: Curve-level evidence suggests phase-dependent relation: early-progress advantage and later catch-up; do not reduce to a single lead.
Use `w45_pairwise_curve_relation_daily_v7_n.csv` and `w45_pairwise_phase_relation_v7_n.csv` to inspect the day-by-day and phase-level evidence. This summary must not replace those tables.

## Organization layers
- departure_layer: has_confirmed_relations — P-Je:departure90:A_leads; H-Je:departure90:A_leads; Je-Jw:departure90:B_leads; H-Je:departure95:A_leads; Je-Jw:departure95:B_leads
- early_progress_layer: has_confirmed_relations — H-Je:t15:A_leads; P-Je:t20:A_leads; H-Je:t20:A_leads; P-Je:t25:A_leads; H-Je:t25:A_leads; H-Je:t30:A_leads; H-Je:t35:A_leads; H-Je:t40:A_leads
- peak_change_layer: tendency_only — no pass90 relations in this layer
- midpoint_layer: tendency_only — no pass90 relations in this layer
- finish_tail_layer: tendency_only — no pass90 relations in this layer
- curve_phase_layer: has_phase_crossing_or_catchup — P-H:phase_crossing_relation; P-Je:phase_crossing_relation; P-Jw:phase_crossing_relation; V-H:phase_crossing_relation; V-Je:phase_crossing_relation; H-Je:phase_crossing_relation; H-Jw:front_loaded_vs_catchup; Je-Jw:phase_crossing_relation

## Relation complexity
Complexity counts: {'phase_crossing': 8, 'weak_order': 2}
A complex or mixed result is not an implementation failure. It means W45 should not be forced into a clean sequence unless later evidence supports it.

## Prohibited interpretations
- Do not interpret progress difference as physical strength difference.
- Do not interpret progress relation as causality.
- Do not convert not_resolved into synchrony; near-equivalence needs an explicit margin.
- Do not hide any of P/V/H/Je/Jw from computation.
- Do not force phase-crossing relations into one lead/lag edge.
