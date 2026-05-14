# V10.7_d H35 residual/anomaly independence audit

## Method boundary
- Route-decision experiment for H35 independence.
- Does not test H35 -> W045/P/V/Je/Jw.
- Does not infer causality.
- H18 precursor-to-H35 is interpreted only if H35 residual independence passes the gate.

## Input status
- smoothed_fields_path: D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz
- field_key: z500_smoothed
- original_shape: (45, 183, 201, 521)
- object_domain_shape_year_day_lat_lon: (45, 183, 81, 121)
- year_dimension: detected
- object_domain_lat: [15.0, 35.0]
- object_domain_lon: [110.0, 140.0]
- available_modes: ['anomaly', 'local_background_removed', 'raw']
- anomaly_status: computed_from_year_mean_daily_climatology

## Route decision
### H35 single-point line
- status: `not_independent`
- evidence: mode=anomaly, domain=object_domain_spatial, residual_fraction_median=0.769, pseudo_q90=0.989, consistency_median=-0.094
- route implication: stop_H35_single_point_line; use H18-H35 package only if H remains relevant

### H18 precursor to H35 residual
- status: `not_tested_because_H35_not_independent`
- evidence: mode=anomaly, domain=object_domain_spatial, spearman_r=-0.0914, perm_p=0.536, loo_sign=1
- route implication: invalid_question_under_current_evidence

## H35 independence rows
- anomaly / object_domain_spatial: status=`not_independent`, residual_fraction=0.769, pseudo_q90=0.989, consistency=-0.094, pseudo_consistency_q90=0.327
- anomaly / profile: status=`not_independent`, residual_fraction=0.644, pseudo_q90=0.987, consistency=0.189, pseudo_consistency_q90=0.803
- local_background_removed / object_domain_spatial: status=`not_independent`, residual_fraction=0.618, pseudo_q90=0.988, consistency=-0.162, pseudo_consistency_q90=0.639
- local_background_removed / profile: status=`not_independent`, residual_fraction=0.463, pseudo_q90=0.962, consistency=0.0634, pseudo_consistency_q90=0.909
- raw / object_domain_spatial: status=`not_independent`, residual_fraction=0.557, pseudo_q90=0.96, consistency=9.98e-05, pseudo_consistency_q90=0.714
- raw / profile: status=`not_independent`, residual_fraction=0.247, pseudo_q90=0.885, consistency=0.278, pseudo_consistency_q90=0.914

## Forbidden interpretations
- Do not call H35 independent unless the route decision supports it.
- Do not ask H18 -> H35 if H35 residual independence is rejected.
- Do not treat this as H35 -> W045 evidence; cross-object audit is separate.