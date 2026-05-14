# V10.7_i patch: W33→W45 cross-object transition mapping audit

## Purpose

This patch adds a new V10.7_i entry point for testing whether the W33/E2 multi-object configuration maps to the W45/M multi-object configuration through cross-object transition mapping.

It is designed after V10.7_h showed that same-object E2–M configuration similarity is insufficient. V10.7_i allows, for example, H_E2 to map to M_Jw / M_P / M_V rather than requiring H_E2 to map to H_M.

## New entry point

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_transition_mapping_v10_7_i.py
```

Optional run controls:

```bash
set V10_7_I_N_PERM=5000
set V10_7_I_N_BOOT=1000
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_transition_mapping_v10_7_i.py
```

Default values are intentionally moderate for exploratory runs:

- `V10_7_I_N_PERM`: 200
- `V10_7_I_N_BOOT`: 200

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_transition_mapping_v10_7_i
```

## Key outputs

```text
tables\w45_transition_mapping_input_audit_v10_7_i.csv
tables\w45_transition_mapping_yearwise_vectors_v10_7_i.csv
tables\w45_e2_to_m_pairwise_transition_matrix_v10_7_i.csv
tables\w45_e2_to_m_multivariate_mapping_skill_v10_7_i.csv
tables\w45_e2_source_object_contribution_to_m_mapping_v10_7_i.csv
tables\w45_h_e2_to_m_target_mapping_v10_7_i.csv
tables\w45_transition_mapping_route_decision_v10_7_i.csv
figures\w45_e2_to_m_pairwise_transition_matrix_v10_7_i.png
figures\w45_e2_to_m_mapping_skill_vs_null_v10_7_i.png
figures\w45_e2_source_object_contribution_to_m_mapping_v10_7_i.png
figures\w45_h_e2_to_m_target_mapping_v10_7_i.png
summary_w45_transition_mapping_v10_7_i.md
run_meta\run_meta_v10_7_i.json
```

## Method boundary

- This is not same-object similarity.
- This is not a control-regression experiment.
- This does not control away W45 component objects.
- This is not causal inference.
- This tests scalar cross-object mapping from E2/W33 to M/W45.
- If scalar mapping is negative, it still does not reject position/shape-based H roles.

## Apply instructions

Copy the `stage_partition` folder in this patch over the project root. Existing files are not replaced except `stage_partition_v10_7/__init__.py`, which is a minimal package extension.

The root log append files are provided separately under `root_log_append/` and should be manually appended to the corresponding root logs.
