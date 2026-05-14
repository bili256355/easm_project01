# V10.7_l H_E2 structure → M_P rainband spatial verification

## Task
This audit verifies the narrow V10.7_k candidate that E2/W33 H morphology transitions correspond to M/W45 precipitation rainband structure changes.

## Method boundary
- This is not causal inference.
- This is not a full E2→M multivariate mapping audit.
- It does not control away W45 component objects.
- It tests high-H vs low-H year composites, lat profiles, metric differences, permutation tests, bootstrap CIs, and year influence.

## Settings
- n_perm = 5000
- n_boot = 1000
- group_frac = 0.3
- H metrics = ['H_west_extent_lon_transition', 'H_zonal_width_transition', 'H_north_edge_lat_transition']
- P metrics = ['P_centroid_lat_transition', 'P_main_band_share_transition', 'P_south_band_share_18_24_transition', 'P_main_minus_south_transition', 'P_spread_lat_transition']

## Route status
spatial_metric_support_for_H_to_P_structure_mapping_candidate

Candidate P rainband metric rows: 9

## Key files
- tables/h_e2_group_years_v10_7_l.csv
- tables/h_group_p_metric_composite_summary_v10_7_l.csv
- tables/h_group_p_spatial_composite_summary_v10_7_l.csv
- tables/h_to_p_influence_by_year_v10_7_l.csv
- tables/h_metric_direction_audit_v10_7_l.csv
- tables/h_e2_to_m_p_spatial_route_decision_v10_7_l.csv

## Forbidden interpretations
- Do not state that H causes P.
- Do not state that H controls W45.
- Do not interpret the sign of H_west_extent_lon before checking the direction audit.
- Do not generalize this H→P verification to full W33→W45 mapping.
