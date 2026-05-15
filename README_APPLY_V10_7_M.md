# README_APPLY_V10_7_M

## Version

`v10.7_m = window_anchor_information_decomposition_v10_7_m`

This patch adds a new semantic diagnostic layer after V10.7_l. It does **not** overwrite V10.7_l and does **not** reuse the V10.7_l output directory.

## Scientific purpose

V10.7_m answers the first-round proof questions that emerged after V10.7_l:

1. **Q2 — information form:** Is the H_zonal_width -> P rainband relationship carried by E2 signed transition, E2 pre/post state, window mean, slope, or a background-like proxy?
2. **Q1/Q4 — anchor specificity:** Is E2/W33 a useful source-window anchor for this H -> P relation, or do E1 / shifted / sliding / random windows perform similarly?
3. **Q3 — representation:** Did the earlier scalarized-transition mapping fail because it compressed away direction/component/structure information?
4. **Q5-min — metric specificity:** Is the route source/target specific inside the H -> P narrow channel?

This patch is intentionally **not** a full V/Je/Jw mechanism network and **not** a causal design.

## Files added

```text
stage_partition/V10/v10.7/scripts/run_window_anchor_information_decomposition_v10_7_m.py
stage_partition/V10/v10.7/src/stage_partition_v10_7/window_anchor_information_decomposition_pipeline.py
root_log_append/ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN__V10_7_M_APPEND.md
root_log_append/ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS__V10_7_M_APPEND.md
README_APPLY_V10_7_M.md
```

No existing V10.7_i / V10.7_l code is replaced.

## Run command

Formal run:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_window_anchor_information_decomposition_v10_7_m.py --n-perm 5000 --n-boot 1000 --group-frac 0.30 --n-random-windows 1000 --progress
```

Small connection test:

```bat
cd /d D:\easm_project01
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_window_anchor_information_decomposition_v10_7_m.py --n-perm 37 --n-boot 23 --group-frac 0.30 --n-random-windows 50 --progress
```

Optional explicit input:

```bat
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_window_anchor_information_decomposition_v10_7_m.py --smoothed-fields D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz --n-perm 5000 --n-boot 1000 --progress
```

## Output directory

```text
stage_partition/V10/v10.7/outputs/window_anchor_information_decomposition_v10_7_m
```

## Main output tables

```text
tables/input_audit_v10_7_m.csv
tables/source_anchor_definitions_v10_7_m.csv
tables/yearly_source_features_v10_7_m.csv
tables/yearly_target_features_v10_7_m.csv
tables/information_form_decomposition_v10_7_m.csv
tables/h_zonal_width_information_form_route_decision_v10_7_m.csv
tables/anchor_specificity_all_windows_v10_7_m.csv
tables/e2_anchor_rank_summary_v10_7_m.csv
tables/random_window_null_summary_v10_7_m.csv
tables/yearly_representation_source_features_v10_7_m.csv
tables/representation_comparison_v10_7_m.csv
tables/scalar_vs_signed_component_route_decision_v10_7_m.csv
tables/source_metric_specificity_v10_7_m.csv
tables/target_metric_specificity_v10_7_m.csv
tables/metric_specificity_route_decision_v10_7_m.csv
run_meta/run_meta_v10_7_m.json
summary_window_anchor_information_decomposition_v10_7_m.md
```

## Interpretation boundary

Allowed:

- V10.7_m can say whether the V10.7_l H_zonal_width -> P rainband route is transition-dominant, state-dominant, broad-background-like, or unclear.
- V10.7_m can say whether E2 is a useful anchor relative to E1 / shifted / sliding windows.
- V10.7_m can say whether scalarized-transition representation appears weaker than signed structural-component representation.

Forbidden:

- Do not say H causes P.
- Do not say H controls W45.
- Do not generalize H -> P to full W33 -> W45.
- Do not say “strength class failed.” The earlier failure is about scalarized transition mapping, not all strength variables.
- Do not say transition windows represent most object information.

## Notes

- Formal permutation/bootstrap is used for named/fixed relation tables and representation comparison.
- Sliding/random anchor specificity is rank-based by default. It is designed as an anchor-screening null, not as a final causal/significance layer.
- The default input remains `foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz`, which is expected to exist locally and may not be present on GitHub because large scientific data files are ignored.
