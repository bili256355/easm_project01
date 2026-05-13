# UPDATE LOG: V9.1_f bootstrap-composite MCA

## hotfix02 — specificity + stricter null + interpretability

Scope: replacement-file hotfix for `bootstrap_composite_mca_audit_v9_1_f.py`. Entry script unchanged.

What changed:

1. Added a target-Y table for all per-window MCA targets, allowing target-specificity and cross-target null checks.
2. Added target specificity audit:
   - tests whether a mode score explains its own target better than other pairwise Y targets.
3. Added cross-target null audit:
   - compares the true target MCA strength against MCA strengths obtained from other target Y vectors in the same window.
4. Added sign-flip direction null:
   - preserves the target magnitude distribution while randomizing order direction.
5. Added phase-composite profile outputs:
   - high / mid / low and top/bottom decile standardized composite structures.
6. Added phase-composite difference outputs:
   - high-minus-low and top-decile-minus-bottom-decile standardized differences.
7. Added pattern summary outputs:
   - object contribution, early/late contribution, and dominant day/profile-coordinate ranges.
8. Added evidence_v3:
   - integrates evidence_v2, specificity, sign-flip null, cross-target specificity, pattern summary, and year-leverage status.

Preserved from hotfix01:

- Boundary all-NaN features are dropped before standardization and MCA/SVD.
- Partial NaNs are imputed only after all-NaN removal.
- Zero-variance features are dropped.
- Object block equal weighting is applied after NaN/zero-variance masking.
- Quantile sensitivity, score-gradient, and evidence_v2 remain available.

Non-changes:

- Does not modify V9.
- Does not modify V9 peak logic.
- Does not modify `Y = peak_B - peak_A`.
- Does not add state/growth/process_a.
- Does not treat bootstrap samples as independent physical years.
