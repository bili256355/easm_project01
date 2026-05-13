# w45_process_curve_foundation_audit_v7_o

## Purpose
This audit tests whether the process/progress curves used by V7-n can serve as an implementation-layer basis for W45 ordering. It does not claim the curves are absolutely correct.

## Input coverage
- state_rebuild_status: success
- input_representation: current V7-e progress representation

## Six assumptions audited
1. pre/post prototypes are usable statistical state prototypes.
2. pre→post direction explains a meaningful share of the W45 trajectory.
3. projection progress agrees with distance-to-pre/post progress.
4. a high-dimensional field can be summarized by a single curve, or must be downgraded as multi-component.
5. pairwise phase comparison is allowed only where both field curves are valid enough.
6. curve relations are checked against projection/distance agreement and pre/post sensitivity to reduce projection-artifact risk.

## Field foundation decisions
- P: usable_with_caution | prepost=strong_prepost_foundation | projection=usable_progress_proxy_with_caution | single_curve=multi_component_transition
- V: usable_with_caution | prepost=strong_prepost_foundation | projection=valid_progress_proxy | single_curve=single_curve_usable_with_heterogeneity
- H: usable_with_caution | prepost=strong_prepost_foundation | projection=usable_progress_proxy_with_caution | single_curve=single_curve_adequate
- Je: usable_with_caution | prepost=strong_prepost_foundation | projection=usable_progress_proxy_with_caution | single_curve=multi_component_transition
- Jw: usable_with_caution | prepost=strong_prepost_foundation | projection=valid_progress_proxy | single_curve=single_curve_usable_with_heterogeneity

## Pair foundation decisions
- P-V: not_comparable | comparable=phase_comparable_with_caution | artifact=high_projection_artifact_risk | global_lead_lag=False
- P-H: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- P-Je: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- P-Jw: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- V-H: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- V-Je: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- V-Jw: curve_relation_usable_with_caution | comparable=phase_comparable_with_caution | artifact=moderate_artifact_risk | global_lead_lag=False
- H-Je: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- H-Jw: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False
- Je-Jw: curve_relation_usable_with_caution | comparable=layer_specific_only | artifact=moderate_artifact_risk | global_lead_lag=False

## Prohibited interpretations
- Do not treat process curves as absolutely correct.
- Do not treat progress difference as physical strength difference or causality.
- Do not interpret projection-based crossing as a physical crossing without distance/residual support.
- Do not convert order-not-resolved into synchrony; near-equivalence needs an explicit equivalence margin.
- If field_curve_foundation_label is multi_component_process or not_valid_as_single_curve, do not use that field for clean global lead/lag.