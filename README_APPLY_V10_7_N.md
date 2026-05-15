# README_APPLY_V10_7_N

## 1. Patch identity

Version:

```text
V10.7_n
```

Task:

```text
h_zonal_width_background_target_preinfo_v10_7_n
```

This patch adds a new narrow diagnostic after V10.7_m. It does **not** overwrite V10.7_l or V10.7_m outputs.

## 2. Scientific purpose

V10.7_n answers only three second-round questions:

```text
Qn1. Is the E2 H_zonal_width signal local to E2, or is it better understood as broader pre-M H background / memory?
Qn2. Is the P target rainband-only, position-only, or a position-rainband north-south structural reorganization?
Qn3. Does E2 / pre-M H_zonal_width carry weak preinformation value, or is the relation better explained as common-year/background coupling?
```

It does **not** re-test scalarized-transition loss from V10.7_m, and it does **not** expand to a full V/Je/Jw mechanism network.

## 3. Files added

```text
stage_partition/V10/v10.7/scripts/run_h_zonal_width_background_target_preinfo_v10_7_n.py
stage_partition/V10/v10.7/src/stage_partition_v10_7/h_zonal_width_background_target_preinfo_pipeline.py
README_APPLY_V10_7_N.md
root_log_append/ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN__V10_7_N_APPEND.md
root_log_append/ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS__V10_7_N_APPEND.md
PATCH_MANIFEST_V10_7_N.txt
```

No `__init__.py` replacement is included. The entry script imports the new module directly from `stage_partition_v10_7`.

## 4. Output directory

```text
stage_partition/V10/v10.7/outputs/h_zonal_width_background_target_preinfo_v10_7_n
```

Subdirectories:

```text
tables/
figures/
run_meta/
```

## 5. Formal run command

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 5000 --n-boot 1000 --n-random-windows 1000 --group-frac 0.30 --progress
```

## 6. Small connectivity run

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h_zonal_width_background_target_preinfo_v10_7_n.py --n-perm 37 --n-boot 23 --n-random-windows 50 --group-frac 0.30 --progress
```

## 7. Important outputs

### Qn1: local E2 vs broad background

```text
tables/h_zonal_width_local_vs_background_relation_v10_7_n.csv
tables/h_zonal_width_background_increment_v10_7_n.csv
tables/h_zonal_width_sliding_background_rank_v10_7_n.csv
```

### Qn2: P target redefinition

```text
tables/p_target_candidates_v10_7_n.csv
tables/p_ns_reorganization_index_components_v10_7_n.csv
tables/p_lat_profile_contrast_v10_7_n.csv
tables/p_target_redefinition_v10_7_n.csv
tables/p_target_family_route_decision_v10_7_n.csv
tables/p_ns_composite_component_dominance_v10_7_n.csv
```

### Qn3: weak preinformation / common-year background

```text
tables/time_order_negative_control_v10_7_n.csv
tables/year_influence_preinformation_v10_7_n.csv
tables/decade_block_sensitivity_v10_7_n.csv
```

### Overall route decision

```text
tables/route_decision_v10_7_n.csv
run_meta/run_meta_v10_7_n.json
summary_h_zonal_width_background_target_preinfo_v10_7_n.md
```

## 8. Method boundary

V10.7_n is not:

```text
- causal inference
- full W33-to-W45 mapping
- full P/V/H/Je/Jw object-network audit
- proof that transition windows represent most object information
- a control-regression that removes P/V/Je/Jw from W45
- a re-test of V10.7_m scalarized-transition loss
```

## 9. Forbidden interpretations

Do not write:

```text
H causes P.
H controls W45.
E2 is the unique source window unless the route decision supports it.
Transition window represents most H information.
P target is rainband-only if position/composite target is stronger.
Scalarized transition failure means strength-class failure.
```

## 10. Allowed interpretation examples

If supported by `route_decision_v10_7_n.csv`, safe wording may be:

```text
E2-local H_zonal_width information remains informative beyond the tested broad-background features.
```

or:

```text
The H_zonal_width relation is better interpreted as broad pre-M H background / memory than a purely local E2 transition.
```

or:

```text
The P target is better defined as position-rainband north-south structural reorganization than rainband-only.
```

