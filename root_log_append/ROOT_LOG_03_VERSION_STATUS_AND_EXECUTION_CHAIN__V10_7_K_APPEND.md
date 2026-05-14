# ROOT LOG 03 APPEND — V10.7_k

## Version

`V10.7_k = W33→W45 structure-transition mapping audit`

## Engineering status

- New single entry script:
  - `stage_partition/V10/v10.7/scripts/run_w45_structure_transition_mapping_v10_7_k.py`
- New implementation module:
  - `stage_partition/V10/v10.7/src/stage_partition_v10_7/w45_structure_transition_pipeline.py`
- New output directory:
  - `stage_partition/V10/v10.7/outputs/w45_structure_transition_mapping_v10_7_k`

## Purpose

V10.7_k is introduced because V10.7_i only tested E2→M mapping using window activity/amplitude-like scalar proxies. V10.7_j showed those indicators were not obviously unusable due to low SNR, but that does not settle whether W33 connects to W45 through structural variables.

V10.7_k therefore tests structure-state and structure-transition mapping:

- `E2_state → M_state`
- `E2_state → M_transition`
- `E2_transition → M_state`
- `E2_transition → M_transition`

It allows H_E2 structure metrics to map to non-H M metrics, such as Jw axis/strength or P/V structure.

## Method boundary

This version is:

- not a regression-control experiment;
- not a same-object similarity audit;
- not a causal inference method;
- not a proof that W33 causes W45;
- a route-decision audit for structural mapping candidates.

## Key outputs to inspect first

1. `tables/w45_structure_transition_route_decision_v10_7_k.csv`
2. `tables/w45_e2_to_m_structure_mapping_skill_v10_7_k.csv`
3. `tables/w45_e2_structure_object_contribution_v10_7_k.csv`
4. `tables/w45_h_e2_structure_to_m_target_mapping_v10_7_k.csv`
5. `tables/w45_structure_pairwise_mapping_matrix_v10_7_k.csv`

## Execution note

Default permutation/bootstrap counts are intentionally low for pilot speed. For formal output, run with:

```bash
set V10_7_K_N_PERM=5000
set V10_7_K_N_BOOT=1000
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_structure_transition_mapping_v10_7_k.py
```
