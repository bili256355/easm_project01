# V10.7_h patch: W45 E1-E2-M multi-object configuration trajectory audit

## Purpose

This patch adds a new V10.7_h entry point. It replaces the invalid regression-control framing for W45 with a configuration-trajectory audit:

- W45 is treated as a multi-object configuration made by P/V/H/Je/Jw.
- P/V/Je/Jw are not controlled away as covariates.
- E1/E2/M configurations are compared within the same year against shuffled-year nulls.
- Object importance is estimated by mask-one-object contribution, not by residualizing other objects.

## New entry point

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_configuration_trajectory_v10_7_h.py
```

Optional permutation count override:

```bash
set V10_7_H_N_PERM=5000
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_configuration_trajectory_v10_7_h.py
```

Default `n_perm=500` is intended as a practical first run. Use 2000–5000 for a more stable final audit if runtime is acceptable.

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_configuration_trajectory_v10_7_h
```

## Key tables

```text
tables\w45_config_input_audit_v10_7_h.csv
tables\w45_e1_e2_m_object_configuration_by_year_v10_7_h.csv
tables\w45_configuration_coupling_v10_7_h.csv
tables\w45_object_contribution_to_configuration_coupling_v10_7_h.csv
tables\w45_configuration_trajectory_year_clusters_v10_7_h.csv
tables\w45_configuration_route_decision_v10_7_h.csv
```

## Key figures

```text
figures\w45_e1_e2_m_configuration_heatmap_by_year_v10_7_h.png
figures\w45_configuration_coupling_vs_shuffle_null_v10_7_h.png
figures\w45_object_removal_contribution_to_coupling_v10_7_h.png
figures\w45_configuration_trajectory_dendrogram_v10_7_h.png
```

## Interpretation boundary

Do not interpret this as causal inference. It asks whether E1/E2/M object configurations are coupled within-year beyond shuffled-year nulls, and which object dimensions contribute to that coupling.

The critical outputs are:

1. `w45_configuration_coupling_v10_7_h.csv` — whether E1-E2, E2-M, and E1-M configuration similarity exceeds shuffled-year null.
2. `w45_object_contribution_to_configuration_coupling_v10_7_h.csv` — whether removing H/P/V/Je/Jw changes E2-M configuration coupling.
3. `w45_configuration_route_decision_v10_7_h.csv` — route decision for W45 interpretation.
