# progress_order_failure_audit_v7_e2

Created at: 2026-05-02T14:47:44

## Purpose
Explain why most V7-e / V7-e1 pairwise progress orders did not pass the 90% bootstrap CI criterion.
This is a failure decomposition audit, not a new timing method and not a result-picking layer.

## Inputs
- V7-e output: `D:\easm_project01\stage_partition\V7\outputs\field_transition_progress_timing_v7_e`
- V7-e1 output: `D:\easm_project01\stage_partition\V7\outputs\progress_order_significance_audit_v7_e1`

## Failure categories
- `passed_90`: pair passes 90% CI and is not treated as a failure.
- `central_overlap`: zero lies inside the 25–75% interval of Δ; current whole-field progress does not separate the pair.
- `tail_uncertainty`: zero is outside the IQR but inside the 90% CI; direction exists in the center but fails in tails.
- `quality_bottleneck`: at least one field has problematic progress quality or pre/post separation.
- `loyo_conflict`: bootstrap median direction conflicts with LOYO median direction.

## Counts

- central_overlap: 18
- tail_uncertainty: 16
- quality_bottleneck: 4
- passed_90: 2

## Window summary

### W002
- pass95/pass90: 0 / 0
- dominant_failure_type: tail_uncertainty
- interpretation: Most failures are tail-uncertainty failures; direction exists in the distribution center but does not pass 90%.
- recommended_next_action: inspect_tail_samples_and_consider_region_level_progress

### W003
- pass95/pass90: 0 / 0
- dominant_failure_type: central_overlap
- interpretation: Most failures are central-overlap failures; current whole-field progress does not separate many fields in this window.
- recommended_next_action: accept_indistinguishable_pairs_or_move_to_region_level_only_if_scientifically_needed

### W005
- pass95/pass90: 1 / 2
- dominant_failure_type: central_overlap
- interpretation: Window has accepted 90% edges plus tail-uncertain candidate relations.
- recommended_next_action: keep_confirmed_edges_and_audit_tail_uncertainty

### W007
- pass95/pass90: 0 / 0
- dominant_failure_type: quality_bottleneck
- interpretation: Progress-quality bottlenecks dominate or tie for the dominant limitation; field progress curves should be inspected before interpretation.
- recommended_next_action: inspect_field_progress_quality

## Interpretation boundaries

- This audit does not prove causality.
- This audit does not create new progress-order evidence.
- Pairs that fail 90% should not be upgraded by narrative wording.
- `method_resolution_limit_risk` only indicates where region/component-level progress may be worth checking.
