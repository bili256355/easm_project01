# V10.6_a W045 precluster audit summary

## Scope
This output audits W045 only. It decomposes day16-19, day30-35, and day41-46 into fixed candidate clusters E1/E2/M, plus H_post_reference around day57.
It does not perform yearwise prediction, spatial/cartopy validation, or causal inference.

## HOTFIX02 interpretation rule
`candidate_inside_cluster` is marker-supported activity and is the event-semantics core.
`curve_peak_without_marker` is curve-only/ramp/shoulder evidence. It must not be treated as equal to marker-supported activity.

## Cluster participation snapshot
- E1_early_precluster:
  - marker_supported_core_objects = ['joint_all', 'P', 'V', 'H']
  - curve_only_ramp_or_shoulder_objects = []
  - weak_curve_signal_objects = ['Je', 'Jw']
  - absent_or_missing_objects = []
- E2_second_precluster:
  - marker_supported_core_objects = ['P', 'V', 'H', 'Je']
  - curve_only_ramp_or_shoulder_objects = ['joint_all', 'Jw']
  - weak_curve_signal_objects = []
  - absent_or_missing_objects = []
- M_w045_main_cluster:
  - marker_supported_core_objects = ['joint_all', 'P', 'V', 'Je', 'Jw']
  - curve_only_ramp_or_shoulder_objects = []
  - weak_curve_signal_objects = []
  - absent_or_missing_objects = ['H']
- H_post_reference:
  - marker_supported_core_objects = ['H']
  - curve_only_ramp_or_shoulder_objects = []
  - weak_curve_signal_objects = ['joint_all', 'P', 'V', 'Je', 'Jw']
  - absent_or_missing_objects = []

## H day35 role audit
- role_class: lineage_assigned_secondary_or_background_conditioning_component
- confirmed_weak_precursor: False
- recommended_wording: H day35 should be described as a W045-context H family inside the E2 second precluster. Current evidence is insufficient to call it a confirmed weak precursor; it is safer to treat it as a lineage-assigned secondary/background-conditioning component until yearwise and spatial checks are completed. HOTFIX02 interpretation rule: E2 marker-supported objects and curve-only/ramp objects must be separated.
- forbidden_wording: Do not write: H day35 is a confirmed/stable weak precursor that triggers W045. Do not write: H leads W045 as a single clean event.

## Interpretation claims
- W045_CLAIM_001 [method_supported_fixed_window_audit]: E1/E2/M candidate-cluster decomposition is detected from fixed windows: marker-supported core objects are E1=['joint_all', 'P', 'V', 'H'], E2=['P', 'V', 'H', 'Je'], M=['joint_all', 'P', 'V', 'Je', 'Jw']; curve-only/ramp objects are E1=[], E2=['joint_all', 'Jw'], M=[].
- W045_CLAIM_002 [supported_as_interpretation_boundary]: H day35 should be described as a W045-context H family inside the E2 second precluster. Current evidence is insufficient to call it a confirmed weak precursor; it is safer to treat it as a lineage-assigned secondary/background-conditioning component until yearwise and spatial checks are completed. HOTFIX02 interpretation rule: E2 marker-supported objects and curve-only/ramp objects must be separated.
- W045_CLAIM_003 [explicit_not_implemented]: V10.6_a does not test yearwise prediction, spatial continuity, or causality.

## Boundary
V10.6_a is a method-layer / derived-structure audit. It should not be used to claim that H day35 is a confirmed weak precursor or causal trigger of W045.