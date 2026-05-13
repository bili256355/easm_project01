# UPDATE_LOG_V9_2_DIRECT_YEAR_2D_MEOF_A

## Version

`V9_2_a` / `direct_year_2d_meof_peak_audit_v9_2_a`

## Purpose

This patch opens a new V9.2 diagnostic/control line.  It is intended to test whether real-year 2D multivariate field modes correspond to different V9 peak-order structures, without using bootstrap-resampled year combinations as the sample axis.

## Why this version was added

V9.1_f used bootstrap composite samples and target-guided MCA/SVD.  That method is useful as a bootstrap-space peak-order dependence diagnostic, but its score phases are not direct real-year classes.  V9.2_a provides a more traditional direct-year field diagnosis:

1. samples are real years, 1979-2023;
2. X is built from 2D object fields rather than zonal/profile features;
3. MVEOF/SVD is unsupervised and not target-guided by a peak-difference Y;
4. PC high/low groups are real-year groups;
5. peak timing is computed on PC-group composites using V9/V7 peak semantics.

## Main implementation choices

- New root: `D:\easm_project01\stage_partition\V9_2\`
- Single entry point: `scripts\run_direct_year_2d_meof_peak_audit_v9_2_a.py`
- Single output root: `outputs\direct_year_2d_meof_peak_audit_v9_2_a\`
- Default windows: `W045`, `W081`, `W113`, `W160`
- Default years: 1979-2023
- Default spatial coarsening: 2 degrees
- Default objects and 2D domains:
  - P: precip, 105-125E, 15-39N
  - V: v850, 105-125E, 10-30N
  - H: z500, 110-140E, 15-35N
  - Je: u200, 120-150E, 25-45N
  - Jw: u200, 80-110E, 25-45N
- Feature weighting:
  - sqrt(cos(lat)) spatial weighting;
  - feature z-score across years;
  - object-block equal weighting.

## Explicit boundaries

V9.2_a must not be interpreted as proving physical regimes or causal paths.  It only supplies timing/statistical audit outputs:

- PC high/low groups are score phases of real years, not named physical year types.
- MVEOF maximum-variance modes are field modes, not automatically peak-order mechanisms.
- PC-group peak timing is group-composite timing, not a single-year rule.
- High/low order contrasts are timing diagnostics, not causality.
- NPZ pattern outputs are data carriers, not completed physical interpretation figures.

## Main outputs

Cross-window outputs are written under:

`outputs\direct_year_2d_meof_peak_audit_v9_2_a\cross_window\`

Important files:

- `v9_2_input_field_audit.csv`
- `v9_2_object_2d_domain_registry.csv`
- `v9_2_meof_feature_block_audit.csv`
- `v9_2_meof_mode_summary_all_windows.csv`
- `v9_2_pc_scores_all_windows.csv`
- `v9_2_pc_phase_years_all_windows.csv`
- `v9_2_pc_group_object_peak_all_windows.csv`
- `v9_2_pc_group_pairwise_peak_order_all_windows.csv`
- `v9_2_pc_group_high_low_order_contrast_all_windows.csv`
- `v9_2_mode_stability_leave_one_year_all_windows.csv`
- `v9_2_leave_one_year_group_peak_stability_all_windows.csv`
- `v9_2_peak_relevance_summary_all_windows.csv`
- `V9_2_A_SUMMARY.md`
- `run_meta.json`

## Run command

```bat
python D:\easm_project01\stage_partition\V9_2\scripts\run_direct_year_2d_meof_peak_audit_v9_2_a.py
```

## Environment overrides

Optional:

```bat
set V9_2_SMOOTHED_FIELDS=D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz
set V9_2_V9_OUTPUT_DIR=D:\easm_project01\stage_partition\V9\outputs\peak_all_windows_v9_a
set V9_2_SPATIAL_RES_DEG=2.0
set V9_2_N_MODES_MAIN=3
set V9_2_N_MODES_SAVE=5
```

If the local V7 helper refuses `bootstrap_n=0`, the run will stop by default rather than silently using bootstrap.  For compatibility only, the user may opt in:

```bat
set V9_2_ALLOW_DETECTOR_COMPAT_BOOTSTRAP_RETRY=1
set V9_2_COMPAT_DETECTOR_BOOTSTRAP_N=1
```

When this compatibility path is used, the compatibility bootstrap is ignored by V9.2 outputs and is not evidence.

---

## hotfix02 / performance patch

Purpose:
- Disable the expensive `leave-one-year group peak stability` branch by default.
- Keep the branch available by environment switch, but do not run it in the standard V9.2_a run.
- Reduce default V9/V7 detector calls by only running PC high/low group peaks. PC mid group peak can still be enabled explicitly.
- Skip saving coarsened input field NPZ files by default to reduce disk I/O. The CSV audit tables remain written.

Default changes:
```text
run_leave_one_year_group_peak_stability = False
run_pc_mid_group_peak = False
save_coarsened_input_fields = False
```

Optional switches:
```bat
set V9_2_RUN_LOO_GROUP_PEAK_STABILITY=1
set V9_2_RUN_PC_MID_GROUP_PEAK=1
set V9_2_SAVE_COARSENED_INPUT_FIELDS=1
```

Scientific boundary:
- This patch does not change the MVEOF X construction.
- This patch does not change PC high/low year grouping.
- This patch does not change the group-composite peak detector logic.
- It only skips optional slow audits/optional mid-phase detector calls by default.

---

# V9.2_a result registry A patch

Date: 2026-05-10

## Purpose

Add a read-only candidate-result registry for existing V9.2_a outputs. This patch does not rerun MVEOF, does not rerun V9/V7 peak detection, and does not perform physical interpretation.

## New entry

```bat
python D:\easm_project01\stage_partition\V9_2\scripts\run_summarize_v9_2_a_result_registry_a.py
```

## New source module

```text
src/stage_partition_v9_2/summarize_v9_2_a_result_registry_a.py
```

## New outputs

```text
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_window_mode_registry.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_window_mode_sequence_summary.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_pc_phase_year_list.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_object_peak_shift_summary.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_order_change_summary.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_detector_quality_audit.csv
outputs/v9_2_a_result_registry_a/cross_window/v9_2_a_detector_quality_summary_by_mode.csv
outputs/v9_2_a_result_registry_a/cross_window/V9_2_A_RESULT_REGISTRY_A_SUMMARY.md
outputs/v9_2_a_result_registry_a/cross_window/run_meta.json
```

## Interpretation boundary

All rows are candidate timing-pattern results. They are not final robust tiers, not physical regimes, and not physical mechanisms. Group-peak leave-one-year stability is marked as not_run_or_incomplete unless a non-empty source table is present.
