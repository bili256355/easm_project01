# UPDATE_LOG_V7_S

## V7-s: W45 H/Jw raw025 process audit

### Purpose

V7-s is a raw-field input-representation sensitivity branch for W45 H/Jw. It tests whether the H/Jw structure seen in 2-degree/profile diagnostics is also visible in raw025 smoothed fields.

This branch is not a replacement for V7-r and is not a global order detector.

### Main questions

1. Does H early-frontloaded progress exist in raw025 fields?
2. Does H show an anchor-near retreat / non-monotonic segment in raw025 fields?
3. Does Jw show mid/late catch-up in raw025 fields?
4. Does H/Jw crossing reflect a raw-field process or a 2-degree/profile compression artifact?

### Entry point

```bash
cd D:\easm_project01
python stage_partition\V7\scripts\run_w45_H_Jw_raw025_process_audit_v7_s.py
```

### Files added

```text
stage_partition/V7/scripts/run_w45_H_Jw_raw025_process_audit_v7_s.py
stage_partition/V7/src/stage_partition_v7/w45_H_Jw_raw025_process_audit.py
stage_partition/V7/UPDATE_LOG_V7_S.md
```

### Output directory

```text
D:\easm_project01\stage_partition\V7\outputs\w45_H_Jw_raw025_process_audit_v7_s
```

### Log directory

```text
D:\easm_project01\stage_partition\V7\logs\w45_H_Jw_raw025_process_audit_v7_s
```

### Important boundaries

- Main computation uses raw025 smoothed fields from `foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz`.
- V7-p/q/r may be read for comparison status only.
- V7-p/q/r outputs are not used to construct raw025 progress.
- This branch only covers H and Jw.
- It does not establish causal direction.
- It does not establish global clean order.
- Contribution-defined regions are diagnostic masks, not confirmed physical regions.

### Expected key outputs

```text
input_audit_v7_s.json
run_meta.json
w45_H_Jw_raw025_wholefield_progress_curves_v7_s.csv
w45_H_Jw_raw025_crossing_events_v7_s.csv
w45_H_raw025_early_contribution_map_v7_s.csv
w45_H_raw025_retreat_contribution_map_v7_s.csv
w45_H_raw025_late_recovery_contribution_map_v7_s.csv
w45_Jw_raw025_catchup_contribution_map_v7_s.csv
w45_H_Jw_raw025_contribution_maps_all_v7_s.csv
w45_H_Jw_raw025_region_definitions_v7_s.csv
w45_H_Jw_raw025_region_progress_curves_v7_s.csv
w45_H_Jw_raw025_region_timing_markers_v7_s.csv
w45_H_Jw_raw025_region_bootstrap_samples_v7_s.csv
w45_H_Jw_raw025_region_bootstrap_summary_v7_s.csv
w45_H_Jw_raw025_vs_2deg_comparison_v7_s.csv
w45_H_Jw_raw025_process_audit_summary_v7_s.md
```

### Interpretation rule

If raw025 supports H early-frontloaded, H retreat, Jw catch-up and H/Jw crossing, the current H/Jw structure can be upgraded to a raw-field-supported phase-specific structure. If raw025 does not support those phenomena, V7-r should remain a 2-degree/profile feature-diagnostic result and should not be regionalized.
