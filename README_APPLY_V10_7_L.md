# V10.7_l H_E2 structure → M_P rainband spatial verification patch

## Purpose

This patch adds a narrow verification audit for the strongest V10.7_k H-source structural candidate:

- E2/W33 H morphology transition, especially `H_west_extent_lon`, `H_zonal_width`, and `H_north_edge_lat`
- against M/W45 precipitation rainband structure, especially `P_centroid_lat`, `P_main_band_share`, `P_south_band_share_18_24`, and `P_main_minus_south`

It does **not** run a full E2→M multivariate mapping audit and does **not** control away W45 component objects.

## Files to copy

Copy the included `stage_partition` folder into your project root:

```bat
xcopy /E /I /Y stage_partition D:\easm_project01\stage_partition
```

Append the files under `root_log_append` to your root logs if you are maintaining consolidated root logs.

## Run

Small validation run:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_e2_to_m_p_spatial_verification_v10_7_l.py --n-perm 37 --n-boot 23 --group-frac 0.30 --progress
```

Formal run:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_e2_to_m_p_spatial_verification_v10_7_l.py --n-perm 5000 --n-boot 1000 --group-frac 0.30 --progress
```

## Output

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h_e2_to_m_p_spatial_verification_v10_7_l
```

Key tables:

```text
tables\h_to_p_input_audit_v10_7_l.csv
tables\h_p_structure_metrics_by_year_v10_7_l.csv
tables\h_e2_group_years_v10_7_l.csv
tables\h_group_p_metric_composite_summary_v10_7_l.csv
tables\h_group_p_spatial_composite_summary_v10_7_l.csv
tables\h_to_p_influence_by_year_v10_7_l.csv
tables\h_metric_direction_audit_v10_7_l.csv
tables\h_e2_to_m_p_spatial_route_decision_v10_7_l.csv
```

Key figures:

```text
figures\p_spatial_*_v10_7_l.png
figures\p_profile_*_v10_7_l.png
figures\scatter_*_v10_7_l.png
```

## Interpretation boundary

This audit can support only:

> H_E2 morphology changes and M_P rainband structural changes show spatial/metric/yearwise correspondence.

It cannot prove causality, and it cannot be generalized to the full W33→W45 mapping without further tests.
