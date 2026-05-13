# UPDATE_LOG_V7_K

## w45_H_timing_marker_audit_v7_k

新增只读审计分支，用于判断 W45-H 是否适合继续使用 midpoint 作为时序标记，或应改写为 early-onset / broad-transition 型转换对象。

### 输入

- `outputs/field_transition_progress_timing_v7_e` 中的 W45 whole-field H progress 结果。
- `outputs/w45_H_raw025_threeband_progress_v7_j` 中的 raw025 公平三分区 H progress 结果。

### 输出

- `w45_H_timing_marker_stability_v7_k.csv`
- `w45_H_observed_timing_shape_v7_k.csv`
- `w45_H_window_censoring_audit_v7_k.csv`
- `w45_H_timing_marker_decision_v7_k.csv`
- `w45_H_timing_marker_window_summary_v7_k.csv`
- `w45_H_timing_marker_audit_summary_v7_k.md`

### 边界

- 不重跑 V7-e 或 V7-j。
- 不改变空间区域。
- 不改变统计阈值。
- 不把不可区分解释为同步。
- 不把 early-onset / broad-transition 解释为 confirmed field order。
