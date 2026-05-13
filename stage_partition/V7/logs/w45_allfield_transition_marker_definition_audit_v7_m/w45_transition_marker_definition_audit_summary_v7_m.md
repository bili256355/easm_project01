# W45 all-field transition marker definition audit v7_m

## Purpose
This audit checks whether the former t(0.25) early-progress crossing is robust to alternative field-transition marker definitions.
It keeps the current V7-e progress/profile representation fixed and does not switch to raw025 inputs.

## Input representation
- V7-e output dir: `D:\easm_project01\stage_partition\V7\outputs\field_transition_progress_timing_v7_e`
- Bootstrap marker rows: 5000
- Build notes: ['recomputed_w45_progress_curves_from_v7e_representation']

## Field transition types
- P: early_departure_broad_transition | recommended_marker_family=departure
- V: early_departure_broad_transition | recommended_marker_family=departure
- H: early_departure_broad_transition | recommended_marker_family=departure
- Je: early_departure_broad_transition | recommended_marker_family=early_threshold;midpoint
- Jw: early_departure_broad_transition | recommended_marker_family=departure

## Marker consistency ledger
- P-V: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- P-H: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- P-Je: departure_supported_order | Departure-from-pre supports the order; threshold/peak support should be checked.
- P-Jw: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- V-H: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- V-Je: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- V-Jw: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- H-Je: robust_early_transition_order | Same direction appears across several threshold levels and at least one non-threshold marker family.
- H-Jw: not_distinguishable_across_markers | No marker family gives a 90% confirmed direction.
- Je-Jw: departure_supported_order | Departure-from-pre supports the order; threshold/peak support should be checked.

## Window-level interpretation
- window_transition_type: early_transition_order_window
- interpretation: At least one pair is supported across multiple marker families; interpret only as marker-consistent early-transition order.

## Prohibited interpretations
- Do not call t25 a physical onset unless marker-consistency support is present.
- Do not treat threshold-only results as robust transition order.
- Do not treat peak-change day as departure or completion timing.
- Do not force marker-inconsistent pairs into an order.
- Do not call not-distinguishable pairs synchronous without an equivalence test.
- Do not omit P/V/H/Je/Jw from W45 summaries.