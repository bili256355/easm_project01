# ROOT_LOG_03 append — V10.7_f

## V10.7_f — W45 H-package to main-cluster cross-object audit

- Status: new patch package prepared.
- Single entry:
  `stage_partition/V10/v10.7/scripts/run_w45_h_package_cross_object_audit_v10_7_f.py`
- Single output directory:
  `stage_partition/V10/v10.7/outputs/w45_h_package_cross_object_audit_v10_7_f`
- Role: route-decision experiment.
- Research question: whether the pre-W45 H adjustment package (`H18/H35`, not H35 alone) has yearwise incremental association with the W45 main-cluster targets `joint45_proxy`, `P45`, `V45`, `Je46`, `Jw41`, and `M_combined`.
- Controls: `P_E2`, `V_E2`, `Je_E2` when available.
- Not a detector rerun, not an H35 single-point test, not causal inference.
- Missing object fields are explicitly skipped and logged; missing fields are not negative evidence.
