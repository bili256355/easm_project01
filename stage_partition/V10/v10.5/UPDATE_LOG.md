# V10.5 update log

## field_index_validation_v10_5_a

Added first external validation layer for the V10.4 object-order timing skeleton.

Scope:

- W045, W113, W160 only.
- Objects: P, V, H, Je, Jw.
- Profile rolling pre/post contrast validation.
- Simple low-dimensional profile metric timing validation.
- Object-order comparison against V10.4 baseline tau=2 order.

Boundaries:

- Does not rerun peak discovery.
- Does not redefine accepted windows.
- Does not perform yearwise validation.
- Does not perform physical or causal interpretation.
- Does not make cartopy maps in this first patch.

## 2026-05-13 - HOTFIX01 / V10.5_b candidate-family-aware profile-energy validation

Purpose:
- Preserve the original V10.5_a field/index validation outputs.
- Add a top-k profile-energy validation layer so that a top1 mismatch is not automatically interpreted as a rejection of the V10.4 assigned object peak.
- Explicitly distinguish same-candidate-family support from switches to known non-strict / secondary candidate families.

New outputs:
- `profile_validation/profile_energy_topk_peaks_by_window_object_v10_5_b.csv`
- `profile_validation/v10_4_assigned_peak_energy_rank_v10_5_b.csv`
- `profile_validation/candidate_family_switch_inventory_v10_5_b.csv`
- `validation_summary_candidate_family_v10_5_b.csv`

Interpretation boundary:
- These outputs are still detector-external validation evidence only.
- They do not prove physical mechanisms, causality, or redefine accepted windows.
- A profile-energy top1 switch to another known candidate family is recorded as candidate-family competition, not as immediate failure of the V10.4 assigned peak.

## 2026-05-13 - HOTFIX02 / V10.5_c candidate-family selection-reason audit

Purpose:
- Preserve V10.5_a and V10.5_b outputs.
- Add a selection-reason audit for cases where profile-energy top1 is stronger than the V10.4 assigned family.
- Classify whether the energy-top1 family was not selected because it belongs to a known non-strict lineage, another strict lineage, an object-native secondary family, or an unclear case requiring review.

New outputs:
- `profile_validation/candidate_family_selection_reason_audit_v10_5_c.csv`
- `profile_validation/candidate_family_role_long_v10_5_c.csv`
- `validation_summary_candidate_family_selection_reason_v10_5_c.csv`

Interpretation boundary:
- This audit does not change V10.4 assignments and does not redefine accepted windows.
- It compares energy-dominant families with lineage-assigned families.
- It is a descriptive method-layer audit, not a physical or causal interpretation.

## 2026-05-13 - HOTFIX03 / V10.5_d peak strength vs bootstrap stability audit

Purpose:
- Preserve V10.5_a/b/c outputs.
- Add a two-dimensional strength-stability audit that separates observed peak strength from year-bootstrap recurrence.
- Check whether non-strict main-method candidate peaks are high-score but lower-bootstrap.
- Check whether profile-energy top1 families are themselves stable under year resampling, or whether they can be high-strength but unstable.

New outputs:
- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`
- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`
- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`
- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`
- `profile_validation/key_competition_cases_v10_5_d.csv`
- `FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md`

Runtime controls:
- `V10_5_DEBUG_N_BOOTSTRAP=20` for a quick test run.
- `V10_5_N_BOOTSTRAP=1000` for the formal profile-energy family bootstrap audit.
- `V10_5_BOOTSTRAP_SEED=42` by default.

Interpretation boundary:
- A high profile-energy score is not treated as proof of strict accepted status.
- A high detector score is not treated as proof of bootstrap stability.
- The audit is a method-layer comparison of strength and stability, not a physical or causal interpretation.
