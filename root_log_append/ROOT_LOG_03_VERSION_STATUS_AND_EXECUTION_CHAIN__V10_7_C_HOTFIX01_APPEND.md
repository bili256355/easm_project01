# ROOT_LOG_03 append — V10.7_c HOTFIX01

## V10.7_c HOTFIX01 status

- Version: `V10.7_c HOTFIX01`
- Scope: replacement-file hotfix for V10.7_c event-content audit.
- Entry unchanged:
  `stage_partition/V10/v10.7/scripts/run_h_w045_event_content_audit_v10_7_c.py`
- Output directory unchanged:
  `stage_partition/V10/v10.7/outputs/h_w045_event_content_audit_v10_7_c`

## What changed

1. Added H-object-domain spatial metrics using the H object/profile domain:
   - latitude: `15–35N`
   - longitude: `110–140E`
2. Retained full-domain composites (`10–60N, 20–150E`) as background context only.
3. Event role classification now prefers H-object-domain spatial metrics/similarity when available.
4. Fixed H18/H19 scale-context aliasing so the V10.7_b `H19` ridge can be mapped to the V10.7_c `H18` event-content window.

## New outputs

- `h_w045_event_spatial_composite_metrics_object_domain_v10_7_c.csv`
- `h_w045_H18_H35_similarity_object_domain_v10_7_c.csv`
- `h_w045_event_spatial_diff_maps_object_domain_v10_7_c.npz`
- `h_w045_H18_H35_H45_H57_spatial_diff_panel_object_domain_cartopy_v10_7_c.png`
- `h_w045_H18_H35_similarity_heatmap_object_domain_v10_7_c.png`

## Interpretation boundary

This hotfix does not change the scientific task of V10.7_c. It only corrects evidence routing and spatial-domain scope. V10.7_c remains an event-content audit, not an influence test or causal test.
