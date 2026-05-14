# V10.7_k W33→W45 structure-transition mapping audit

## Method boundary
- This version tests structure/state/transition mapping, not activity-amplitude mapping.
- It does not control away P/V/Je/Jw as covariates.
- It allows H_E2 structural metrics to map to non-H M metrics.
- It does not infer causality.

## Main route decisions
- **E2_to_M_structure_mapping**: `not_evaluated_multivariate_skipped` — Multivariate ridge mapping was skipped by runtime policy; inspect pairwise/H-specific target mapping only.  
  Implication: Do not draw any overall E2→M structure-mapping conclusion from this run.
- **H_E2_structure_contribution**: `not_evaluated_object_contribution_skipped` — Remove-one-source contribution was skipped by runtime policy.  
  Implication: Use H-specific pairwise mapping only; do not claim H is key/non-key from object-contribution tables.
- **H_E2_to_M_target_specific_mapping**: `H_target_specific_candidate_lines` — H_north_edge_lat->H.H_north_edge_lat:r=0.580,p=0.000,clear_structure_mapping_support; H_west_extent_lon->P.P_centroid_lat:r=-0.600,p=0.000,clear_structure_mapping_support; H_north_edge_lat->H.H_north_edge_lat:r=-0.705,p=0.000,clear_structure_mapping_support; H_south_edge_lat->H.H_south_edge_lat:r=-0.559,p=0.000,clear_structure_mapping_support; H_west_extent_lon->H.H_north_edge_lat:r=-0.532,p=0.000,clear_structure_mapping_support  
  Implication: Only these narrow H-structure target mappings should be considered for follow-up.

## Ridge mapping skill snapshot

## Forbidden interpretations
- Do not interpret support as causality.
- Do not treat a negative result here as proof that H has no W45 role.
- Do not treat raw-mode results as primary evidence when anomaly/local-background disagree.