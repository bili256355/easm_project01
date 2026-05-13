# W45 progress implementation/failure audit v7_e3

Created at: 2026-05-02T15:28:37

## Purpose
Audit whether the V7-e progress-midpoint implementation is interpretable for W45, and explain why W45 directional tendencies do not pass 90% confirmation.

This audit does not rerun progress timing, does not change thresholds, and does not upgrade supported tendencies to confirmed results.

## Implementation validity by field

- P: valid_for_statistical_audit — pre/post separation and progress quality are usable, and midpoint is not unusually broad within W45.
- V: valid_for_statistical_audit — pre/post separation and progress quality are usable, and midpoint is not unusually broad within W45.
- H: partly_valid_with_caution — field has the broadest W45 transition and high midpoint uncertainty rank; midpoint is interpretable but not a sharp timing marker.
- Je: valid_for_statistical_audit — pre/post separation and progress quality are usable, and midpoint is not unusually broad within W45.
- Jw: partly_valid_with_caution — field has broad duration and/or wide midpoint uncertainty relative to other W45 fields.

## Tail failure summary

- H-P (H<P): dominant_tail_type=mixed_tail; tail_fraction=0.184
- H-V (H<V): dominant_tail_type=mixed_tail; tail_fraction=0.145
- H-Je (H<Je): dominant_tail_type=early_field_late_tail; tail_fraction=0.119
- H-Jw (H<Jw): dominant_tail_type=all_fields_clustered; tail_fraction=0.153
- P-Je (P<Je): dominant_tail_type=all_fields_clustered; tail_fraction=0.113
- V-Je (V<Je): dominant_tail_type=all_fields_clustered; tail_fraction=0.336
- Jw-Je (Jw<Je): dominant_tail_type=all_fields_clustered; tail_fraction=0.224
- P-Jw (none<none): dominant_tail_type=no_direction_assigned; tail_fraction=0.000

## Five-field role audit

- P: role=middle_early_unresolved; evidence=supported_tendency_only; main_failure=tail_uncertainty; caution=Do not treat supported tendencies as 90% confirmed field order.
- V: role=middle_late_unresolved; evidence=supported_tendency_only; main_failure=tail_uncertainty; caution=Do not treat supported tendencies as 90% confirmed field order.
- H: role=early_broad_candidate; evidence=supported_tendency_only; main_failure=tail_uncertainty; caution=Do not treat supported tendencies as 90% confirmed field order.
- Je: role=late_candidate; evidence=supported_tendency_only; main_failure=tail_uncertainty; caution=Do not treat supported tendencies as 90% confirmed field order.
- Jw: role=middle_early_unresolved; evidence=supported_tendency_only; main_failure=central_overlap; caution=Do not treat supported tendencies as 90% confirmed field order.

## Next-step decision

- implementation_validity_overall: partly_valid_with_caution
- h_morphology_label: broad_transition_midpoint_limited
- dominant_h_tail_type: mixed_tail
- recommended_next_action: run_H_focused_region_or_feature_level_progress_before_upgrading_W45_order
- reason: H shows broad/uncertain midpoint behavior or H-tail failures; W45 H early tendency may be spatially heterogeneous.

## Interpretation boundaries

- Do not interpret not-distinguishable pairs as synchrony without an equivalence test.
- Do not interpret supported tendencies as 90% confirmed orders.
- Do not infer causality or pathway direction from this timing audit.
- Do not omit any of P/V/H/Je/Jw from the W45 interpretation; unresolved roles are still results.
