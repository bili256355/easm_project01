# V10.7_f patch: W45 H-package to main-cluster cross-object audit

## Purpose

This patch adds a single-entry V10.7_f audit:

- Main question: whether the pre-W45 H adjustment package (`H18/H35`, not H35 alone) has yearwise incremental association with the W45 main-cluster targets.
- Targets: `joint45_proxy`, `P45`, `V45`, `Je46`, `Jw41`, and `M_combined` when the corresponding object fields are available.
- Controls: `P_E2`, `V_E2`, `Je_E2` when available.

This patch does **not** test H35 as a single point. V10.7_d already closed H35 as a stable-independent target.

## Apply

Copy the `stage_partition` folder into your project root so that the new files land under:

```text
D:\easm_project01\stage_partition\V10\v10.7
```

## Run

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_h_package_cross_object_audit_v10_7_f.py
```

## Output

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_h_package_cross_object_audit_v10_7_f
```

Key tables:

```text
tables\w45_cross_object_input_audit_v10_7_f.csv
tables\w45_yearwise_H_package_and_main_cluster_strength_v10_7_f.csv
tables\w45_H_package_to_main_cluster_correlation_v10_7_f.csv
tables\w45_H_package_incremental_explanatory_power_v10_7_f.csv
tables\w45_H_package_route_decision_v10_7_f.csv
```

Key figures:

```text
figures\w45_H_package_vs_main_cluster_targets_v10_7_f.png
figures\w45_H_package_delta_r2_by_target_v10_7_f.png
```

## Interpretation boundary

- Positive results mean yearwise incremental association only.
- They do not imply causality.
- Missing object fields are skipped and logged in the input audit. Missing fields are not negative evidence.
- Object-domain strengths are proxy indices from detected field keys and configured domains. If project-specific object masks/registries exist, they should replace the default domains in a future hardening version.
