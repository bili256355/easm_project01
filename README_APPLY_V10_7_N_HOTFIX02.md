# V10.7_n HOTFIX02: empty A2/A3 table schema and route-decision guard

## Purpose

Fixes a runtime crash in `decide_routes()`:

```text
KeyError: 'mode'
```

The crash happens when `--experiment-a-policy skip_heavy` intentionally skips experiment A2/A3. In that branch, `increment_df` was an empty DataFrame without columns, while route decision logic still tried to filter `increment_df["mode"]`.

## Files replaced

```text
stage_partition/V10/v10.7/scripts/run_h_zonal_width_background_target_preinfo_v10_7_n.py
stage_partition/V10/v10.7/src/stage_partition_v10_7/h_zonal_width_background_target_preinfo_pipeline.py
```

## What changed

1. Adds explicit empty-table schemas for skipped increment/sliding outputs.
2. Adds `filter_mode()` guard for route-decision filtering.
3. Makes `incremental_cv_table()` return an empty DataFrame with required columns when no rows are generated.
4. Keeps `skip_heavy` as a valid fast-connect policy.

## What did not change

- No scientific question changed.
- No V10.7_l or V10.7_m output is touched.
- `full` mode remains the formal mode.
- `screen` and `skip_heavy` remain screening / connection modes only.

## Run commands

Fast connect:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 37 --n-boot 23 --n-random-windows 50 --group-frac 0.30 --experiment-a-policy skip_heavy --progress
```

Screening:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --experiment-a-policy screen --screen-n-perm 199 --screen-n-boot 199 --progress
```

Formal:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --experiment-a-policy full --progress
```
