# V10.7_k patch: W33→W45 structure-transition mapping audit

## Purpose

This patch adds a new route-decision audit for W45. It replaces the activity-amplitude-only E2→M mapping with structure/state/transition metrics.

It asks whether W33/E2 structural state or structural change maps to W45/M structural state or structural change, allowing H_E2 to map to non-H M targets such as P, V, Je, or Jw.

It is not a causal method and it does not control away P/V/Je/Jw as covariates.

## Files to copy

Copy these into the project root `D:\easm_project01`:

```text
stage_partition/V10/v10.7/scripts/run_w45_structure_transition_mapping_v10_7_k.py
stage_partition/V10/v10.7/src/stage_partition_v10_7/w45_structure_transition_pipeline.py
```

The `__init__.py` in the patch is safe to copy only if the package directory does not already contain one. If your current `__init__.py` has existing imports, do not overwrite it unnecessarily.

## Run

Fast pilot run:

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py
```

More stable formal run:

```bash
set V10_7_K_N_PERM=5000
set V10_7_K_N_BOOT=1000
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py
```

If the smoothed fields file is not at the default location, set:

```bash
set V10_7_SMOOTHED_FIELDS=D:\path\to\smoothed_fields.npz
```

## Output

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_structure_transition_mapping_v10_7_k
```

Key tables:

```text
tables/w45_structure_metric_input_audit_v10_7_k.csv
tables/w45_structure_vectors_by_year_v10_7_k.csv
tables/w45_structure_pairwise_mapping_matrix_v10_7_k.csv
tables/w45_e2_to_m_structure_mapping_skill_v10_7_k.csv
tables/w45_e2_structure_object_contribution_v10_7_k.csv
tables/w45_h_e2_structure_to_m_target_mapping_v10_7_k.csv
tables/w45_structure_transition_route_decision_v10_7_k.csv
```

Key figures:

```text
figures/w45_structure_pairwise_mapping_heatmap_v10_7_k.png
figures/w45_structure_mapping_skill_vs_null_v10_7_k.png
figures/w45_structure_object_contribution_v10_7_k.png
figures/w45_h_e2_structure_target_mapping_v10_7_k.png
```

## Interpretation boundary

- Do not interpret mappings as causality.
- Do not treat a negative result here as proof that H has no W45 role.
- This version uses structure metrics computed from the available smoothed fields; if project-native daily index tables are later preferred, this version should be upgraded to read those exact indices.
- Raw mode is included only as a reference; anomaly/local-background-removed modes should be primary.
