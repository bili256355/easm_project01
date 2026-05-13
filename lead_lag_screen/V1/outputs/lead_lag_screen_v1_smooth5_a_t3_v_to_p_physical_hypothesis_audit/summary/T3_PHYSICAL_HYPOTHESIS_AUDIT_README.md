# T3 V→P physical hypothesis audit

## Scope

This audit does not rerun the V1 lead-lag screen and does not establish pathway causality. It checks whether the T3 V→P contraction is consistent with several physical hypotheses: rain-band reorganization, V-component shift, internal T3 state mixing/dilution, P-target component shift, and synchronous multi-family reorganization.

## Key outputs

- `tables/t3_subwindow_v_to_p_dilution_classification.csv`
- `tables/window_subwindow_regional_precip_contribution.csv`
- `tables/t3_v_index_to_regional_p_response.csv`
- `tables/t3_p_target_group_v_to_p_summary.csv`
- `tables/s3_t3_s4_multi_family_stability_shift.csv`
- `tables/t3_physical_hypothesis_evidence_summary.csv`

## Hypothesis evidence summary

| hypothesis_id   | hypothesis_name                                  | evidence_strength   | supporting_evidence                                                                                                                                                                                                                                | contradicting_evidence   | missing_evidence                                                                                    | interpretation_status   |
|:----------------|:-------------------------------------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------|:----------------------------------------------------------------------------------------------------|:------------------------|
| H1              | rain_band_spatial_reorganization                 | strong              | Meiyu share decreased (-0.028). South/SCS share increased (0.028).                                                                                                                                                                                 |                          | Manual review of precip maps is still recommended.                                                  | partially_supported     |
| H2              | V_component_shift_strength_to_NSdiff_or_position | strong              | V_strength abs response early/late=1.32/0.962; V_NS_diff=1.72/1.66. V_strength weakens while V_NS_diff is retained/stronger.                                                                                                                       |                          | Needs physical interpretation of V composite patterns.                                              | partially_supported     |
| H3              | T3_internal_state_mixing_dilution                | strong              | 14/21 V→P pairs changed early/late profile type; 17 dilution-like pairs.                                                                                                                                                                           |                          | Needs regional/spatial confirmation that early and late T3 correspond to different physical states. | partially_supported     |
| H4              | P_target_component_shift                         | strong              | Mainband late positive=0.144, change=-0.071; spread/centroid/south late positive=0.303, change=-0.078. Target shifts away from mainband toward spread/centroid/south. Index validity context: 10/10 listed P/V indices are strong where available. |                          | Needs manual map review for P target component composite structure.                                 | partially_supported     |
| H5              | synchronous_multi_family_reorganization          | moderate            | 4/20 family-directions lower stable fraction at T3; 0/20 higher tau0-coupled fraction.                                                                                                                                                             |                          | Needs spatial/object-field confirmation; this table only uses V1 stability counts.                  | partially_supported     |


## Interpretation guardrail

Evidence strength in this audit is a diagnostic classification, not a physical mechanism proof. Any `strong` or `moderate` hypothesis should be treated as support for further physical interpretation and map review, not as a final causal pathway.