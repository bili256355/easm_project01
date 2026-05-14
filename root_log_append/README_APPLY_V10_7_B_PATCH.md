# V10.7_b patch: H W045 Gaussian derivative scale-space diagnostic

## Purpose

This patch adds a dedicated H-only W045 scale diagnostic. It is designed to look for heuristic clues around H19, H35, H45, and H57 using Gaussian derivative scale-space on the H object state matrix.

It does **not** rerun `ruptures.Window` and does **not** treat `detector_width` as a physical scale axis.

## Apply

Copy the patch contents into your project root `D:\easm_project01`, preserving paths.

New entry:

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_w045_scale_diagnostic_v10_7_b.py
```

## Output

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h_w045_scale_diagnostic_v10_7_b
```

Main tables:

```text
tables\h_w045_scale_energy_map_v10_7_b.csv
tables\h_w045_scale_local_maxima_v10_7_b.csv
tables\h_w045_scale_ridges_v10_7_b.csv
tables\h_w045_ridge_family_summary_v10_7_b.csv
tables\h_w045_target_day_scale_response_v10_7_b.csv
tables\h_w045_scale_interpretation_summary_v10_7_b.csv
```

Main figures:

```text
figures\h_w045_scale_energy_map_v10_7_b.png
figures\h_w045_scale_ridge_overlay_v10_7_b.png
figures\h_w045_target_day_scale_response_v10_7_b.png
figures\h_w045_H19_H35_H45_H57_derivative_panel_v10_7_b.png
```

Root-log append materials are in `root_log_append/`.

## Interpretation boundary

This is a heuristic scale diagnostic. It only helps decide whether later yearwise/spatial tests should target H35, the H19-H35 package, H45 absence, or H57 as a post-W045 reference. It does not prove precursor, condition, or mechanism.
