# W045 Jw/H Peak Sensitivity Cause Audit v1_1 Summary

Generated at: 2026-05-12T03:50:41.217355+00:00

## 1. What v1_1 corrects

v1_1 is a correction audit, not a physical sub-peak classification. It corrects three v1 risks: all-config pair-order inflation, missing Jw axis/core-latitude check, and insufficient H candidate admissibility screening.

## 2. Revised diagnosis

| target   | v1_primary_cause                          | v1_1_primary_cause                                    | interpretation_level                        | recommended_next_step                                                                                         |   confidence |
|:---------|:------------------------------------------|:------------------------------------------------------|:--------------------------------------------|:--------------------------------------------------------------------------------------------------------------|-------------:|
| H        | SEARCH_WINDOW_MIXING                      | SEARCH_WINDOW_MIXING_OR_OUTER_CANDIDATE               | Level_1_rule_search_sensitivity             | Exclude non-admissible H clusters from W045 subpeak/order interpretation; audit H core-window candidate only. |          0.8 |
| Jw       | STATISTICAL_MULTICLUSTER_NOT_YET_PHYSICAL | AXIS_SHIFT_CANDIDATE_WITHOUT_FULL_PHYSICAL_PROOF      | Level_3_single_physical_dimension_candidate | Do a narrow Jw-only axis/core-latitude follow-up before any full subpeak physical classification.             |          0.7 |
| Jw-H     | ORDER_RELATIVELY_STABLE                   | ORDER_INFLATED_BY_INADMISSIBLE_OR_RULE_LOCKED_CONFIGS | Level_1_filter_sensitive_order              | Do not use all-config Jw-H order as result; report filtered order as mixed/near-tie/filter-sensitive.         |          0.8 |

## 3. Filtered Jw-H order

| filter_name                     |   n_config_pairs |   frac_jw_after_h |   frac_jw_before_h |   frac_near_tie |   median_lag | order_stability_status   | interpretation_allowed   |
|:--------------------------------|-----------------:|------------------:|-------------------:|----------------:|-------------:|:-------------------------|:-------------------------|
| F0_all_configs                  |               72 |          0.861111 |          0.0833333 |       0.0555556 |           11 | ORDER_STABLE_JW_AFTER_H  | True                     |
| F1_exclude_outside_system       |               14 |          0.285714 |          0.428571  |       0.285714  |            0 | NEAR_TIE_DOMINATED       | False                    |
| F2_exclude_max_score            |               54 |          0.814815 |          0.111111  |       0.0740741 |           10 | ORDER_STABLE_JW_AFTER_H  | True                     |
| F3_exclude_max_score_or_outside |               14 |          0.285714 |          0.428571  |       0.285714  |            0 | NEAR_TIE_DOMINATED       | False                    |
| F4_core_configs_only            |               14 |          0.285714 |          0.428571  |       0.285714  |            0 | NEAR_TIE_DOMINATED       | False                    |
| F5_strict_core_configs          |               14 |          0.285714 |          0.428571  |       0.285714  |            0 | NEAR_TIE_DOMINATED       | False                    |

## 4. Jw axis/core-latitude proxy audit

| cluster_pair                 |   max_lat_difference |   core_lat_difference | axis_shift_status   | physical_distinctness_revision   | interpretation_allowed   | core_lat_proxy_method                   |
|:-----------------------------|---------------------:|----------------------:|:--------------------|:---------------------------------|:-------------------------|:----------------------------------------|
| C01_041_043__vs__C02_048_049 |                2.625 |                 2.625 | AXIS_SHIFT_CLEAR    | AXIS_DISTINCT_ONLY               | True                     | max_lat_from_v1_profile_component_audit |

## 5. H candidate admissibility

| cluster_id   |   day_min |   day_max |   inside_system_fraction |   outside_system_fraction |   max_score_fraction | candidate_legality_status              | recommended_use                                    |
|:-------------|----------:|----------:|-------------------------:|--------------------------:|---------------------:|:---------------------------------------|:---------------------------------------------------|
| C01_017_019  |        17 |        19 |                     0    |                      1    |                  1   | NOT_ADMISSIBLE_FOR_W045_INTERPRETATION | exclude_from_W045_subpeak_and_order_interpretation |
| C02_030_048  |        30 |        48 |                     0.25 |                      0.75 |                  0.1 | TRANSITION_EDGE_CANDIDATE              | weak_admissible_only                               |

## 6. Cluster distinctness revision

| object   | cluster_pair                 | v1_distinctness_status   |   max_lat_difference |   core_lat_difference | candidate_legality_pair_status   | axis_shift_status   | revised_distinctness_status   | interpretation_allowed   |
|:---------|:-----------------------------|:-------------------------|---------------------:|----------------------:|:---------------------------------|:--------------------|:------------------------------|:-------------------------|
| H        | C01_017_019__vs__C02_030_048 | NOT_DISTINCT             |                1.25  |                 1.25  | NOT_ADMISSIBLE_PAIR              | NA_NOT_JW           | NOT_ADMISSIBLE_PAIR           | False                    |
| Jw       | C01_041_043__vs__C02_048_049 | NOT_DISTINCT             |                2.625 |                 2.625 | ADMISSIBLE_PAIR                  | AXIS_SHIFT_CLEAR    | AXIS_DISTINCT_ONLY            | False                    |

## 7. Interpretation constraints

- Do not treat all-config Jw-H order as a final W045 order unless the filtered/core layers support the same order.
- Do not interpret H outside-system or max-score locked clusters as W045 subpeaks.
- A Jw `AXIS_SHIFT_CLEAR` flag only supports a follow-up axis/core-latitude audit; it is not by itself a confirmed physical subpeak.
- `core_lat` in this patch is a proxy based on v1 `max_lat`, because v1 did not save full profile vectors.