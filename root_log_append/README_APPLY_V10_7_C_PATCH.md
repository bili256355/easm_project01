# V10.7_c H W045 Event-Content Audit Patch

## Purpose

This patch adds **V10.7_c = H18/H35/H45 event-content audit**.

It answers what H changes occur around H18, H35, H45, and H57 before any influence / precursor / causal interpretation.
It does **not** test H18→H35, H35→W045, lead-lag, causality, or physical mechanism.

## Apply

Copy the included `stage_partition/V10/v10.7` files into:

```text
D:\easm_project01\stage_partition\V10\v10.7
```

This patch only adds new V10.7_c files and does not modify V10.7_a or V10.7_b outputs.

## Run

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_w045_event_content_audit_v10_7_c.py
```

## Main outputs

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h_w045_event_content_audit_v10_7_c
```

Tables:

```text
tables\h_w045_event_profile_diff_v10_7_c.csv
tables\h_w045_event_feature_contribution_v10_7_c.csv
tables\h_w045_event_spatial_composite_metrics_v10_7_c.csv
tables\h_w045_H18_H35_similarity_v10_7_c.csv
tables\h_w045_event_yearwise_change_v10_7_c.csv
tables\h_w045_event_yearwise_consistency_summary_v10_7_c.csv
tables\h_w045_event_content_role_summary_v10_7_c.csv
```

Figures:

```text
figures\h_w045_H18_H35_H45_profile_diff_panel_v10_7_c.png
figures\h_w045_event_feature_contribution_top_v10_7_c.png
figures\h_w045_H18_H35_H45_H57_spatial_diff_panel_cartopy_v10_7_c.png
figures\h_w045_H18_H35_similarity_heatmap_v10_7_c.png
```

Metadata and summary:

```text
run_meta\run_meta_v10_7_c.json
summary_h_w045_event_content_audit_v10_7_c.md
```

## Inputs

The patch reconstructs H profile/state through the existing V10.7_a H builder from:

```text
foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz
```

It also tries to detect H/z500 spatial field and lat/lon/year keys from the same file.
If spatial or year information is missing, the corresponding section is skipped and recorded in run_meta. It does not fabricate outputs.

## Interpretation boundary

Use V10.7_c to decide whether H18 and H35 have similar content, different content, weak content, or yearwise/spatial consistency.
Do not cite V10.7_c as evidence for H18→H35, H35→W045, or confirmed weak precursor status.
