# V9.1_f Result Registry A Summary

Generated: 2026-05-09 18:40:48

## 1. Scope

This is a statistical evidence registry. It does not provide physical interpretations, physical type names, or causal pathway claims.

## 2. Method freeze status

- **V9 replay audit**: pass — 60/60 pass (freeze_ok)
- **boundary NaN feature audit**: pass — 20 object-window blocks audited; all-NaN boundary features explicitly handled (freeze_ok)
- **evidence_v3**: pass — 15 targets in evidence_v3 (freeze_ok)
- **permutation audit**: pass — 15/15 pass-prefix strict_99 (freeze_ok)
- **signflip direction null**: pass — 15/15 pass-prefix direction_specific (freeze_ok)
- **mode stability**: pass — 15/15 stable (freeze_ok)
- **year leverage audit**: pass — 0/15 extreme-year dominated (freeze_ok)
- **target specificity audit**: pass — 59 rows from bootstrap_composite_mca_target_specificity_all_windows.csv (freeze_ok)
- **cross-target null**: pass — 44 rows from bootstrap_composite_mca_cross_target_null_all_windows.csv (freeze_ok)
- **quantile sensitivity**: pass — 60 rows from bootstrap_composite_mca_quantile_sensitivity_all_windows.csv (freeze_ok)
- **score-gradient audit**: pass — 15 rows from bootstrap_composite_mca_score_gradient_summary_all_windows.csv (freeze_ok)
- **pattern summary**: pass — 75 rows from bootstrap_composite_mca_pattern_summary_all_windows.csv (freeze_ok)

## 3. Result tier counts

- Tier_1_main_statistical_result / robust_common_mode_reversal: 7
- Tier_1_main_statistical_result / robust_pair_specific_reversal: 4
- Tier_2_important_auxiliary_result / robust_continuous_gradient: 1
- Tier_2_important_auxiliary_result / robust_one_sided_locking: 3

## 4. Window summaries

- W045: 3 target(s); dominant statistical pattern = robust_common_mode_reversal; Tier1=3, Tier2=0. Main pairs: Je-Jw:robust_common_mode_reversal; P-Jw:robust_pair_specific_reversal; V-Jw:robust_common_mode_reversal
- W081: 3 target(s); dominant statistical pattern = robust_common_mode_reversal; Tier1=3, Tier2=0. Main pairs: P-V:robust_common_mode_reversal; V-Jw:robust_common_mode_reversal; H-Jw:robust_pair_specific_reversal
- W113: 5 target(s); dominant statistical pattern = robust_one_sided_locking; Tier1=2, Tier2=3. Main pairs: V-Je:robust_pair_specific_reversal; H-Je:robust_one_sided_locking; P-V:robust_common_mode_reversal; Jw-H:robust_one_sided_locking; Jw-V:robust_one_sided_locking
- W160: 4 target(s); dominant statistical pattern = robust_common_mode_reversal; Tier1=3, Tier2=1. Main pairs: V-Je:robust_pair_specific_reversal; H-Jw:robust_continuous_gradient; P-V:robust_common_mode_reversal; Jw-V:robust_common_mode_reversal

## 5. Pair summaries

- Jw-V: windows=W045;W081;W113;W160; evidence=W045:robust_common_mode_reversal; W081:robust_common_mode_reversal; W113:robust_one_sided_locking; W160:robust_common_mode_reversal
- H-Jw: windows=W081;W113;W160; evidence=W081:robust_pair_specific_reversal; W113:robust_one_sided_locking; W160:robust_continuous_gradient
- P-V: windows=W081;W113;W160; evidence=W081:robust_common_mode_reversal; W113:robust_common_mode_reversal; W160:robust_common_mode_reversal
- Je-V: windows=W113;W160; evidence=W113:robust_pair_specific_reversal; W160:robust_pair_specific_reversal
- H-Je: windows=W113; evidence=W113:robust_one_sided_locking
- Je-Jw: windows=W045; evidence=W045:robust_common_mode_reversal
- Jw-P: windows=W045; evidence=W045:robust_pair_specific_reversal

## 6. Ready for physical-audit queue

- W045 P-Jw: robust_pair_specific_reversal (priority_1_pair_specific_reversal)
- W081 H-Jw: robust_pair_specific_reversal (priority_1_pair_specific_reversal)
- W113 V-Je: robust_pair_specific_reversal (priority_1_pair_specific_reversal)
- W160 V-Je: robust_pair_specific_reversal (priority_1_pair_specific_reversal)
- W045 Je-Jw: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W045 V-Jw: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W081 P-V: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W081 V-Jw: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W113 P-V: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W160 Jw-V: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W160 P-V: robust_common_mode_reversal (priority_2_common_mode_reversal)
- W113 H-Je: robust_one_sided_locking (priority_3_locking_or_gradient)
- W113 Jw-H: robust_one_sided_locking (priority_3_locking_or_gradient)
- W113 Jw-V: robust_one_sided_locking (priority_3_locking_or_gradient)
- W160 H-Jw: robust_continuous_gradient (priority_3_locking_or_gradient)

## 7. Interpretation boundary

Allowed terms: robust pair-specific reversal, robust common-mode reversal, robust one-sided locking, robust continuous gradient, bootstrap-composite coupling, order-relevant mode.

Disallowed at this stage: physical type, causal pathway, real year-type, high/low physical naming, object A drives object B.
