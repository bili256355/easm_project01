# V10.5_a field/index validation summary

## Scope

This run validates V10.4 object-order timing skeletons for W045, W113, and W160 using profile rolling pre/post energy and simple low-dimensional profile metrics.
It does not rerun peak discovery, redefine accepted windows, perform yearwise validation, or make physical/causal claims.

## Run status

- status: `success`
- target_lineages: `['W045', 'W113', 'W160']`
- objects: `['P', 'V', 'H', 'Je', 'Jw']`
- primary_k: `9`

## Object validation status counts

```json
{
  "PARTIALLY_SUPPORTED": 6,
  "SUPPORTED": 6,
  "NOT_SUPPORTED": 3
}
```

## Object-order validation status counts

```json
{
  "ORDER_SUPPORTED": 19,
  "ORDER_NEAR_TIE_SUPPORTED": 6,
  "ORDER_AMBIGUOUS": 5
}
```

## Candidate-family top-k validation counts (V10.5_b)

```json
{
  "FAMILY_TOP1_SUPPORTED": 12,
  "FAMILY_TOPK_SUPPORTED": 3
}
```

## Candidate-family switch counts (V10.5_b)

```json
{
  "SAME_FAMILY": 36,
  "SWITCH_TO_KNOWN_NON_STRICT_LINEAGE": 7,
  "SWITCH_TO_NEARBY_STRICT_LINEAGE": 2
}
```

## Key output files

- `validation_summary_v10_5_a.csv`
- `profile_validation/profile_energy_peak_by_window_object_k_v10_5_a.csv`
- `profile_validation/profile_energy_topk_peaks_by_window_object_v10_5_b.csv`
- `profile_validation/v10_4_assigned_peak_energy_rank_v10_5_b.csv`
- `profile_validation/candidate_family_switch_inventory_v10_5_b.csv`
- `validation_summary_candidate_family_v10_5_b.csv`
- `profile_validation/candidate_family_selection_reason_audit_v10_5_c.csv`
- `profile_validation/candidate_family_role_long_v10_5_c.csv`
- `validation_summary_candidate_family_selection_reason_v10_5_c.csv`
- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`
- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`
- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`
- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`
- `profile_validation/key_competition_cases_v10_5_d.csv`
- `FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md`
- `index_validation/object_metric_timing_by_window_v10_5_a.csv`
- `order_validation/object_order_validation_by_window_v10_5_a.csv`
- `figures/*_profile_energy_all_objects_v10_5_a.png`

## Interpretation boundary

These outputs are external validation evidence for method-layer timing skeletons. They are not physical mechanism proof, causal evidence, or a re-decision of accepted windows.
V10.5_b top-k outputs distinguish same-family support from candidate-family switches; a top1 mismatch is not automatically a rejection of the V10.4 assigned peak.
V10.5_c selection-reason outputs classify whether the energy-top1 family was not selected because it belongs to a known non-strict lineage, another strict lineage, an object-native secondary family, or requires review.