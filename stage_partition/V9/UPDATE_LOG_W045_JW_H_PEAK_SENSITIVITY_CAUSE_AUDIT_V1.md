# UPDATE_LOG_W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1

## Purpose

Add a minimal cause-audit patch for the high V9 peak-selection sensitivity, restricted to:

- window: W045
- objects: H and Jw
- task: diagnose why the original peak selection is sensitive

This patch is not a physical sub-peak classification and does not replace V9 peak_all_windows_v9_a or peak_selection_sensitivity_v9_a.

## New entry

```bat
python D:\easm_project01\stage_partition\V9\scripts\run_peak_sensitivity_cause_audit_w045_jw_h_v1.py
```

The script uses a relative V9 root based on its own location. No hardcoded Python executable path is required.

## New module

```text
V9/src/stage_partition_v9/peak_sensitivity_cause_audit_w045_jw_h_v1.py
```

## New output directory

```text
V9/outputs/peak_sensitivity_cause_audit_w045_jw_h_v1
```

## Main outputs

```text
cross_window/implementation_consistency_audit_W045_Jw_H.csv
cross_window/selected_day_by_config_W045_Jw_H.csv
cross_window/selected_day_cluster_summary_W045_Jw_H.csv
cross_window/factor_contribution_W045_Jw_H.csv
cross_window/factor_cluster_crosstab_W045_Jw_H.csv
cross_window/search_window_boundary_audit_W045_Jw_H.csv
cross_window/score_landscape_by_config_W045_Jw_H.csv
cross_window/score_landscape_summary_W045_Jw_H.csv
cross_window/profile_component_audit_W045_Jw_H.csv
cross_window/cluster_physical_distinctness_audit_W045_Jw_H.csv
cross_window/jw_h_order_sensitivity_decomposition_W045.csv
cross_window/sensitivity_cause_diagnosis_W045_Jw_H.csv
cross_window/W045_JW_H_PEAK_SENSITIVITY_CAUSE_AUDIT_V1_SUMMARY.md
run_meta.json
summary.json
```

## Design notes

1. Existing V9 sensitivity outputs are used as the primary evidence layer.
2. If local foundation smoothed fields are available, the patch recomputes W045 H/Jw score landscapes and profile-feature audits using the same helper functions from `peak_selection_sensitivity_v9_a`.
3. If foundation fields are missing, score/profile outputs are still written with `UNAVAILABLE` status, while the CSV-based cause audit still runs.
4. No changepoint re-detection is performed.
5. No physical interpretation is assigned by default.

## Interpretation boundary

This patch diagnoses whether sensitivity is more consistent with:

- implementation/helper risk;
- flat peak plateau;
- search-window mixing;
- selection-rule semantic differences;
- detector/smoothing scale dependence;
- statistical multi-cluster without physical distinctness;
- physically distinct multi-candidate peaks;
- low-contrast uninterpretable behavior.

Only the last physically distinct category should lead to a later sub-peak interpretation step.
