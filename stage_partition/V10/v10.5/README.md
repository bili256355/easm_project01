# V10.5 field/index validation

This subpackage performs a first detector-external validation of V10.4 object-order timing skeletons.

## Current entry

```bat
python D:\easm_project01\stage_partition\V10\v10.5\scripts\run_field_index_validation_v10_5_a.py
```

Optional foundation input override:

```bat
set V10_5_SMOOTHED_FIELDS=D:\path\to\smoothed_fields.npz
```

## Scope

- Target lineages/windows: `W045`, `W113`, `W160`.
- Objects: `P`, `V`, `H`, `Je`, `Jw`.
- Uses the foundation smoothed fields and V10.4 baseline object assignments.
- Does not rerun peak discovery.
- Does not redefine accepted windows.
- Does not provide physical mechanism or causal proof.

## V10.5_a outputs

- `validation_summary_v10_5_a.csv`
- `profile_validation/profile_energy_peak_by_window_object_k_v10_5_a.csv`
- `profile_validation/profile_energy_curves_by_window_object_v10_5_a.csv`
- `profile_validation/profile_energy_primary_k_summary_v10_5_a.csv`
- `index_validation/object_metric_timing_by_window_v10_5_a.csv`
- `index_validation/object_index_support_summary_v10_5_a.csv`
- `order_validation/object_order_validation_by_window_v10_5_a.csv`

## V10.5_b candidate-family-aware outputs

V10.5_b adds top-k profile-energy validation to avoid treating a top1 mismatch as automatic non-support.

- `profile_validation/profile_energy_topk_peaks_by_window_object_v10_5_b.csv`
- `profile_validation/v10_4_assigned_peak_energy_rank_v10_5_b.csv`
- `profile_validation/candidate_family_switch_inventory_v10_5_b.csv`
- `validation_summary_candidate_family_v10_5_b.csv`

These outputs answer:

- Is the V10.4 assigned peak family the top1 profile-energy family?
- If not, does it appear in top-k as a secondary profile-energy peak?
- Did profile-energy top1 switch to another known object-native or joint non-strict lineage candidate?
- What is the assigned-day energy score ratio relative to the top1 energy peak?

## V10.5_c candidate-family selection-reason outputs

V10.5_c adds a descriptive audit for cases where profile-energy top1 is stronger than the V10.4 assigned family. It asks why the energy-dominant family was not selected by the current V10.4 lineage assignment.

- `profile_validation/candidate_family_selection_reason_audit_v10_5_c.csv`
- `profile_validation/candidate_family_role_long_v10_5_c.csv`
- `validation_summary_candidate_family_selection_reason_v10_5_c.csv`

These outputs distinguish:

- energy top1 is the same family as the V10.4 assigned peak;
- energy top1 belongs to a known non-strict lineage;
- energy top1 belongs to another strict lineage;
- energy top1 is an object-native secondary / other family;
- reason remains unclear and needs review.


## V10.5_d strength-stability audit outputs

V10.5_d separates peak strength from year-bootstrap stability.
It checks whether non-strict main-method peaks are high-score but lower-bootstrap, and whether profile-energy top1 families are stable under year resampling.

New outputs:

- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`
- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`
- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`
- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`
- `profile_validation/key_competition_cases_v10_5_d.csv`
- `FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md`

Runtime controls:

```bat
set V10_5_DEBUG_N_BOOTSTRAP=20
python D:\easm_project01\stage_partition\V1010.5\scriptsun_field_index_validation_v10_5_a.py
```

Formal run:

```bat
set V10_5_N_BOOTSTRAP=1000
python D:\easm_project01\stage_partition\V1010.5\scriptsun_field_index_validation_v10_5_a.py
```

## Interpretation boundary

V10.5 is a method-layer validation package. It is useful for identifying profile-energy support, secondary support, or candidate-family competition. It is not a physical explanation layer and should not be used as direct causal evidence.
