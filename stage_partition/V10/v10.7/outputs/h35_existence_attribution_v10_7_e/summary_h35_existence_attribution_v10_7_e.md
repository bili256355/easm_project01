# V10.7_e H35 existence attribution audit

## Method boundary
- This is an attribution audit for why H35 is extracted by the H-only main-method context.
- It does not test H35 -> W045/P/V/Je/Jw.
- It does not prove causality or physical mechanism.
- It distinguishes candidate sources: H18-like second stage, background/local curvature, few-year driven, method shoulder, or unresolved.

## Input status
- smoothed_fields_path: D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz
- field_key: z500_smoothed
- original_shape: (45, 183, 201, 521)
- object_domain_shape_year_day_lat_lon: (45, 183, 81, 121)
- object_domain_lat: [15.0, 35.0]
- object_domain_lon: [110.0, 140.0]
- year_dimension: detected
- available_modes: ['anomaly', 'local_background_removed', 'raw']
- anomaly_status: computed_from_year_mean_daily_climatology

## Route decision
### H35 existence attribution
- status: `method_score_shoulder_or_background`
- evidence: primary=anomaly/object_domain_spatial; H35 residual_fraction median=0.769, pseudo q90=0.989; median score drop after H18-like removal=0.15; median score drop after background removal=0.424; top 20% year score share=0.335; top feature contribution overlap=0.2
- route implication: H35 exists as method-level candidate but has no stable independent attribution

### E2 multi-object component
- status: `not_tested_in_v10_7_e`
- evidence: V10.7_e is H-only. Cross-object E2 requires P/V/Je/Jw inputs.
- route implication: test separately only if H package remains relevant

## Required reading boundary
- If the decision is `H18_like_second_stage`, stop H35 single-point line and use H18-H35 package only if H remains relevant.
- If the decision is `seasonal_background_or_local_curvature`, downgrade H35 from event interpretation.
- If the decision is `few_year_driven`, do not use climatological H35 as a stable event; inspect year subsets.
- E2 multi-object attribution is not tested here.