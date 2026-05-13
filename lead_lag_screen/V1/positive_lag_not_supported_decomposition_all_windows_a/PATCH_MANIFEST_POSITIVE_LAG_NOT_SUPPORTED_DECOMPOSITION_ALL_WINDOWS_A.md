# PATCH MANIFEST: positive_lag_not_supported_decomposition_all_windows_a

## Purpose

Add a read-only V1 audit for decomposing `positive_lag_not_supported` rows across **all windows**, not only T3/T4. This is intended to diagnose why `lead_lag_yes` is low by comparing T3/T4 against all other windows under the same diagnostic buckets.

## Files added

```text
lead_lag_screen/V1/positive_lag_not_supported_decomposition_all_windows_a/run_v1_positive_lag_not_supported_decomposition_all_windows_a.py
lead_lag_screen/V1/positive_lag_not_supported_decomposition_all_windows_a/PATCH_MANIFEST_POSITIVE_LAG_NOT_SUPPORTED_DECOMPOSITION_ALL_WINDOWS_A.md
```

## Default input

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv
```

## Default output

```text
D:\easm_project01\lead_lag_screen\V1\positive_lag_not_supported_decomposition_all_windows_a\outputs
```

## Run command

```bat
python D:\easm_project01\lead_lag_screen\V1\positive_lag_not_supported_decomposition_all_windows_a\run_v1_positive_lag_not_supported_decomposition_all_windows_a.py
```

## Main outputs

```text
tables/positive_lag_not_supported_decomposition_long.csv
tables/positive_lag_not_supported_decomposition_counts_by_window.csv
tables/positive_lag_not_supported_decomposition_counts_by_family_direction.csv
tables/positive_lag_not_supported_blocker_counts.csv
tables/positive_lag_support_profile_counts.csv
tables/positive_lag_not_supported_secondary_dominance_counts.csv
tables/positive_lag_not_supported_metric_distribution_summary.csv
tables/positive_lag_not_supported_near_pass_candidates.csv
tables/positive_lag_not_supported_decomposition_diagnosis_table.csv
summary/summary.json
summary/run_meta.json
logs/RUN_LOG.md
```

## Notes

- This audit does not rerun V1 lead-lag screening.
- It does not modify V1 outputs.
- Diagnostic buckets are derived summaries of rows already labeled `positive_lag_not_supported`; they are not new physical conclusions.
- Default audit windows are all nine windows: S1, T1, S2, T2, S3, T3, S4, T4, S5.
