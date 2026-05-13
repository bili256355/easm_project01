# W045_Jw_H_peak_sensitivity_cause_audit_v1 summary

## 0. Interpretation boundary
This output is a cause audit for W045 H/Jw peak-selection sensitivity. It does not assign physical sub-peak interpretations and does not replace V9 peak_all_windows_v9_a.

## 1. Run status
- started_at: 2026-05-12T11:12:00
- finished_at: 2026-05-12T11:13:40
- score_landscape_status: OK
- profile_component_status: OK
- figure_status: OK

## 2. Implementation consistency
- H: WARN — Baseline day reproduces, but profile build audit reports used_v7_profile_helper=False; keep helper-consistency risk flag.
- Jw: WARN — Baseline day reproduces, but profile build audit reports used_v7_profile_helper=False; keep helper-consistency risk flag.

## 3. Selected-day clusters
- H C01_017_019: day 17-19, n=12, frac=0.17, status=RULE_OR_SEARCH_LOCKED_CLUSTER
- H C02_030_048: day 30-48, n=60, frac=0.83, status=STABLE_CROSS_SMOOTH_CLUSTER
- Jw C01_041_043: day 41-43, n=24, frac=0.33, status=STABLE_CROSS_SMOOTH_CLUSTER
- Jw C02_048_049: day 48-49, n=37, frac=0.51, status=STABLE_CROSS_SMOOTH_CLUSTER
- Jw C03_069_069: day 69-69, n=5, frac=0.07, status=RULE_OR_SEARCH_LOCKED_CLUSTER
- Jw C04_074_074: day 74-74, n=6, frac=0.08, status=RULE_OR_SEARCH_LOCKED_CLUSTER

## 4. Dominant configuration factors
- H: selection_rule eta2=0.639, hint=selection_rule_dominant
- Jw: selection_rule eta2=0.588, hint=selection_rule_dominant

## 5. Landscape / profile evidence
- H score landscape modal type: MULTI_LOCAL_PEAK
- Jw score landscape modal type: EDGE_PEAK
- H cluster distinctness statuses: NOT_DISTINCT
- Jw cluster distinctness statuses: NOT_DISTINCT

## 6. Jw-H order sensitivity
- lag class counts: {'Jw_after_H': 62, 'Jw_before_H': 6, 'near_tie': 4}
- dominant source of lag variation: BOTH_OR_NEAR_BALANCED

## 7. Cause diagnosis
- H: SEARCH_WINDOW_MIXING | next: Selected peaks are boundary/outside-system dominated; audit search window before subpeak interpretation. | confidence=0.75
- Jw: STATISTICAL_MULTICLUSTER_NOT_YET_PHYSICAL | next: Multiple clusters exist but physical distinctness is not strong enough for interpretation. | confidence=0.65
- Jw_minus_H: ORDER_RELATIVELY_STABLE | next: Pairwise order is more stable than object peak days; compare with object-level cause audit. | confidence=0.6

## 8. Output files
- cross_window/implementation_consistency_audit_W045_Jw_H.csv
- cross_window/selected_day_by_config_W045_Jw_H.csv
- cross_window/selected_day_cluster_summary_W045_Jw_H.csv
- cross_window/factor_contribution_W045_Jw_H.csv
- cross_window/factor_cluster_crosstab_W045_Jw_H.csv
- cross_window/search_window_boundary_audit_W045_Jw_H.csv
- cross_window/score_landscape_by_config_W045_Jw_H.csv
- cross_window/score_landscape_summary_W045_Jw_H.csv
- cross_window/profile_component_audit_W045_Jw_H.csv
- cross_window/cluster_physical_distinctness_audit_W045_Jw_H.csv
- cross_window/jw_h_order_sensitivity_decomposition_W045.csv
- cross_window/sensitivity_cause_diagnosis_W045_Jw_H.csv
- cross_window/W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_SUMMARY.md
- run_meta.json
- summary.json
- figures/fig1_selected_day_by_config_W045_Jw_H.png
- figures/fig2_selected_day_histogram_W045_Jw_H.png
- figures/fig3_factor_contribution_W045_Jw_H.png
- figures/fig4_score_landscape_W045_Jw_H.png
- figures/fig5_cluster_profile_feature_W045_Jw_H.png
