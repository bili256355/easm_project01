# V10.7_a H-only main-method event atlas patch

## Purpose

This patch adds a new independent version:

```text
stage_partition/V10/v10.7
```

V10.7_a is **not** a physical interpretation layer. It is a H-only main-method baseline export designed to provide later tests with a richer object-event base.

It reruns the H object-native detector score curve across several `detector_width` values and exports:

- full-season H detector score curves by width;
- H local candidate peaks by width;
- width=20 baseline reproduction audit against inherited H candidate days;
- strong-window H event packages;
- width-stability summary;
- figures for full-season curves, candidate raster, and strong-window panels.

## Entry point

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_object_event_atlas_v10_7_a.py
```

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h_object_event_atlas_v10_7_a
```

## Main tables

```text
tables/h_detector_score_curves_by_width_v10_7_a.csv
tables/h_candidate_catalog_by_width_v10_7_a.csv
tables/h_width20_baseline_reproduction_audit_v10_7_a.csv
tables/h_event_atlas_by_window_width_v10_7_a.csv
tables/h_width_stability_summary_v10_7_a.csv
tables/h_profile_validity_v10_7_a.csv
tables/h_state_feature_table_v10_7_a.csv
```

## Main figures

```text
figures/h_detector_score_fullseason_by_width_v10_7_a.png
figures/h_candidate_days_by_width_raster_v10_7_a.png
figures/h_window_event_panels_v10_7_a.png
```

## Interpretation boundary

This patch deliberately does **not** do:

- bootstrap recurrence support;
- yearwise validation;
- cartopy spatial validation;
- causal inference;
- multi-object atlas expansion beyond H.

It should not be used alone to claim that H is a weak precursor or background condition. It only builds the H event baseline that later tests can target.

## Detector-width grid

Default grid:

```text
12, 16, 20, 24, 28
```

Width=20 is treated as the inherited baseline. The expected width=20 H candidate list is:

```text
19, 35, 57, 77, 95, 115, 129, 155
```

The run writes a baseline reproduction audit and warns in the summary if reproduction fails.

## Root-log append

This patch does not add subdirectory UPDATE_LOG files. Optional root-log append material is included under:

```text
root_log_append/
```
