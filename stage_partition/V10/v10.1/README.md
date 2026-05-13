# V10.1 Joint Main-Window Reproduction Patch

## Purpose

This bundle lives entirely under `V10/v10.1/` and semantically rewrites the historical joint-object main-window discovery chain:

`joint objects -> full-season candidate peaks -> bootstrap support -> candidate point bands -> derived windows -> strict accepted-window lineage`.

It is a prerequisite layer for later object-native peak discovery and sensitivity testing.

## Boundaries

This patch does **not** perform object-native peak discovery, H/Jw sensitivity testing, pair-order analysis, or physical interpretation. It also does not re-decide whether W135 or other non-strict windows should enter the mainline.

The code intentionally does **not** import `stage_partition_v6`, `stage_partition_v6_1`, `stage_partition_v7`, or `stage_partition_v9`. Historical CSV outputs are read only as regression references.

## Run

```bat
python D:\easm_project01\stage_partition\V1010.1\scriptsun_joint_main_window_reproduce_v10_1.py
```

Debug bootstrap run:

```bat
set V10_1_DEBUG_N_BOOTSTRAP=20
python D:\easm_project01\stage_partition\V1010.1\scriptsun_joint_main_window_reproduce_v10_1.py
```

Formal run:

```bat
set V10_1_N_BOOTSTRAP=1000
python D:\easm_project01\stage_partition\V1010.1\scriptsun_joint_main_window_reproduce_v10_1.py
```

## Outputs

All outputs stay inside:

`V10/v10.1/outputs/joint_main_window_reproduce_v10_1/`

Key files:

- `joint_point_layer/joint_candidate_registry_v10_1.csv`
- `joint_point_layer/joint_candidate_bootstrap_summary_v10_1.csv`
- `joint_window_layer/joint_candidate_point_bands_v10_1.csv`
- `joint_window_layer/joint_derived_windows_registry_v10_1.csv`
- `lineage/joint_main_window_lineage_v10_1.csv`
- `audit/v10_1_joint_main_window_regression_audit.csv`
- `JOINT_MAIN_WINDOW_REPRODUCE_V10_1_SUMMARY.md`

## Expected interpretation

If regression audit passes, this confirms that the joint-object discovery lineage has been reproduced in V10.1. Non-strict candidates such as day 18 should then be treated as known joint candidate lineage, not as unknown contamination. This run does not assign physical meaning to any object-specific peak.
