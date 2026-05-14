# V10.7_c HOTFIX01 — H-object-domain spatial metrics + H18/H19 scale-context alias

## Purpose

This hotfix does **not** change the V10.7_c scientific task. It only fixes two interpretation-layer issues found after the first V10.7_c run:

1. **H18/H19 scale-context mismatch**
   - V10.7_b labels the early stable ridge as `H19`.
   - V10.7_c event-content windows label the same early event as `H18`.
   - The original role summary could therefore report `no_clear_scale_ridge_near_target` for H18 even when the V10.7_b H19 ridge exists.
   - HOTFIX01 treats `H18` and `H19` as aliases for the early-H scale context.

2. **Full-domain spatial metrics overreach**
   - The original spatial composite metrics used the broad domain `10–60N, 20–150E`.
   - The H profile/object definition uses the H-object domain `15–35N, 110–140E`.
   - HOTFIX01 keeps the full-domain maps/metrics as background context, but adds H-object-domain spatial metrics and object-domain similarity tables.
   - Event role classification now uses H-object-domain spatial metrics/similarity when available.

## Files to replace

Copy these files into the existing V10.7 source directory:

```text
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\event_content_config.py
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\spatial_composite.py
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\event_content_classifier.py
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\event_content_pipeline.py
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\event_content_summary.py
D:\easm_project01\stage_partition\V10\v10.7\src\stage_partition_v10_7\yearwise_content.py
```

## Re-run command

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_w045_event_content_audit_v10_7_c.py
```

## New/changed outputs

The original outputs are still generated. HOTFIX01 additionally writes:

```text
tables\h_w045_event_spatial_composite_metrics_object_domain_v10_7_c.csv
tables\h_w045_H18_H35_similarity_object_domain_v10_7_c.csv
tables\h_w045_event_spatial_diff_maps_object_domain_v10_7_c.npz
figures\h_w045_H18_H35_H45_H57_spatial_diff_panel_object_domain_cartopy_v10_7_c.png
figures\h_w045_H18_H35_similarity_heatmap_object_domain_v10_7_c.png
```

## Interpretation boundary

- Full-domain spatial composites remain useful as broad background maps.
- H-object-domain metrics are the primary spatial evidence for H event-content classification.
- This hotfix still does not test H18→H35 influence, H35→W045 influence, causality, or confirmed weak-precursor status.
