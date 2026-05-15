# ROOT_LOG_03 VERSION STATUS AND EXECUTION CHAIN -- V10.7_N APPEND

## Version

```text
V10.7_n = h_zonal_width_background_target_preinfo_v10_7_n
```

## Status

New semantic diagnostic after V10.7_m. This version is not a bug fix and does not overwrite previous outputs.

## Entry script

```text
stage_partition/V10/v10.7/scripts/run_h_zonal_width_background_target_preinfo_v10_7_n.py
```

## Core module

```text
stage_partition/V10/v10.7/src/stage_partition_v10_7/h_zonal_width_background_target_preinfo_pipeline.py
```

## Output directory

```text
stage_partition/V10/v10.7/outputs/h_zonal_width_background_target_preinfo_v10_7_n
```

## Execution chain

```text
run_h_zonal_width_background_target_preinfo_v10_7_n.py
  -> Settings
  -> run_h_zonal_width_background_target_preinfo_v10_7_n(settings)
  -> load smoothed_fields.npz
  -> build H/P daily metrics under raw/anomaly/local_background_removed
  -> build H_zonal_width local and broad-background source features
  -> build P target candidates including P_NS_reorganization_index and lat-profile contrast
  -> run local-vs-background audit
  -> run target redefinition audit
  -> run weak-preinformation / temporal-control audit
  -> write tables, figures, route decision, run_meta, summary
```

## Formal command

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --progress
```

## Engineering boundary

This version does not modify V10.7_l or V10.7_m code/output directories. It uses a new output directory and writes method boundaries into run_meta.

