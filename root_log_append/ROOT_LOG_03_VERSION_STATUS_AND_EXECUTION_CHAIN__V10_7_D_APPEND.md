# ROOT_LOG_03 append: V10.7_d

## V10.7_d — H35 residual/anomaly independence audit

- Version: V10.7_d
- Entry:
  - `stage_partition/V10/v10.7/scripts/run_h35_residual_independence_v10_7_d.py`
- Output:
  - `stage_partition/V10/v10.7/outputs/h35_residual_independence_v10_7_d`
- Status:
  - Implemented as a route-decision experiment.
- Research role:
  - Tests whether H35 has stable independent residual content after removing H18-like structure and local/background signal.
  - Only if this gate passes does the code interpret H18 predictive support for H35 residual.
- Not a role:
  - Not a H35 -> W045 cross-object audit.
  - Not causal inference.
  - Not a detector rerun.
  - Not a descriptive H18/H35 similarity audit.
