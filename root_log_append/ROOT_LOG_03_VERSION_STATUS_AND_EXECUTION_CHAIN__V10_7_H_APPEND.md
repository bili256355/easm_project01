# ROOT_LOG_03 append — V10.7_h W45 configuration trajectory audit

## Version status

- Version: V10.7_h
- Entry: `stage_partition/V10/v10.7/scripts/run_w45_configuration_trajectory_v10_7_h.py`
- Output: `stage_partition/V10/v10.7/outputs/w45_configuration_trajectory_v10_7_h`

## Scientific / method status

V10.7_h replaces the invalid V10.7_g-style regression-control framing for W45. It treats W45 as a multi-object configuration made by P/V/H/Je/Jw, not as an external target after controlling away the component objects.

The implemented audit tests:

1. Whether E1-E2, E2-M, and E1-M configurations are coupled within-year beyond shuffled-year null.
2. Whether removing one object dimension changes the configuration coupling.
3. Whether year trajectories form exploratory configuration types.

## Execution-chain status

Single entry script. No old entry is modified. Outputs are written only to the V10.7_h output directory. Root-log append files are provided separately; no subdirectory UPDATE_LOG is written.
