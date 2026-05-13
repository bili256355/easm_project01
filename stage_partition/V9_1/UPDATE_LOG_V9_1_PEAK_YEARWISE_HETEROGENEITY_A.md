# V9.1 peak_yearwise_heterogeneity_v9_1_a

## Purpose

This patch adds an audit-only V9.1 branch that reads V9 peak-only outputs and tests whether V9 peak-order instability may reflect yearwise/event-type heterogeneity.

## Boundary

- Does not modify V9 source files.
- Does not overwrite V9 outputs.
- Does not redefine the V9/V7 peak detector.
- Does not include state, growth, pre-post process, coordinate-audit, or physical mechanism interpretation.
- Year-type clusters are provisional audit structures, not physical climate regimes.

## Main outputs

For each V9 window W045/W081/W113/W160:

- `yearwise_object_peak_registry_Wxxx.csv`
- `jackknife_object_peak_influence_Wxxx.csv`
- `bootstrap_peak_mode_audit_Wxxx.csv`
- `year_type_cluster_audit_Wxxx.csv`
- `pairwise_order_by_year_type_Wxxx.csv`
- `peak_instability_source_summary_Wxxx.csv`

Cross-window:

- `cross_window/yearwise_object_peak_registry_all_windows.csv`
- `cross_window/jackknife_object_peak_influence_all_windows.csv`
- `cross_window/bootstrap_peak_mode_audit_all_windows.csv`
- `cross_window/year_type_cluster_audit_all_windows.csv`
- `cross_window/pairwise_order_by_year_type_all_windows.csv`
- `cross_window/v9_peak_instability_source_summary.csv`

## Run

```bat
python D:\easm_project01\stage_partition\V9_1\scripts\run_peak_yearwise_heterogeneity_v9_1_a.py
```

Optional:

```bat
set V9_1_MODE_MERGE_GAP_DAYS=2
set V9_1_CLUSTER_K_VALUES=2,3
python D:\easm_project01\stage_partition\V9_1\scripts\run_peak_yearwise_heterogeneity_v9_1_a.py
```

## hotfix01

- 修复 `_bootstrap_mode_audit()` 调用 `_merge_day_modes()` 时的关键字参数名错误：`mode_merge_gap_days` -> `merge_gap_days`。
- 本修复只影响 bootstrap peak mode audit 的函数调用接通，不改变 V9.1 的方法语义、不修改 V9、不改输出目录结构。
