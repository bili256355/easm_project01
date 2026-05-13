# W45 feature process resolution V7-q

Created: 2026-05-03T14:09:53

## Purpose

V7-q raises W45 from whole-field scalar process curves to field x feature/component-level process ensembles. It rebuilds from the current 2-degree interpolated profile/state base and does not read V7-m/n/o/p derived result tables as input.

## Base rebuild

| field   |   n_days |   n_features | feature_metadata_status   | feature_axis_type   |   feature_coordinate_min |   feature_coordinate_max |   state_nan_fraction | state_builder_source           | smoothed_fields_path                                                              | state_rebuild_status   |
|:--------|---------:|-------------:|:--------------------------|:--------------------|-------------------------:|-------------------------:|---------------------:|:-------------------------------|:----------------------------------------------------------------------------------|:-----------------------|
| P       |      183 |           13 | available                 | feature_index       |                        0 |                       12 |            0.0437158 | stage_partition_v7.field_state | D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz | success                |
| V       |      183 |           11 | available                 | feature_index       |                        0 |                       10 |            0.0437158 | stage_partition_v7.field_state | D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz | success                |
| H       |      183 |           11 | available                 | feature_index       |                        0 |                       10 |            0.0437158 | stage_partition_v7.field_state | D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz | success                |
| Je      |      183 |           11 | available                 | feature_index       |                        0 |                       10 |            0.0437158 | stage_partition_v7.field_state | D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz | success                |
| Jw      |      183 |           11 | available                 | feature_index       |                        0 |                       10 |            0.0437158 | stage_partition_v7.field_state | D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz | success                |

## Field feature timing distributions

| field   | marker       |   feature_timing_median |   feature_timing_iqr | heterogeneity_label                 | field_internal_process_label        |
|:--------|:-------------|------------------------:|---------------------:|:------------------------------------|:------------------------------------|
| H       | departure90  |                    37   |                 0    | coherent_feature_transition         | coherent_feature_transition         |
| H       | t25          |                    37   |                 0    | coherent_feature_transition         | coherent_feature_transition         |
| H       | t50          |                    39   |                 0    | coherent_feature_transition         | coherent_feature_transition         |
| H       | t75          |                    44   |                 9.5  | strongly_heterogeneous_transition   | strongly_heterogeneous_transition   |
| H       | peak_smooth3 |                    36   |                 2.75 | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Je      | departure90  |                    37   |                 8    | strongly_heterogeneous_transition   | strongly_heterogeneous_transition   |
| Je      | t25          |                    36   |                 1    | coherent_feature_transition         | coherent_feature_transition         |
| Je      | t50          |                    41   |                 6    | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Je      | t75          |                    52   |                 3    | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Je      | peak_smooth3 |                    32   |                14    | multi_component_transition          | multi_component_transition          |
| Jw      | departure90  |                    37   |                 0    | coherent_feature_transition         | coherent_feature_transition         |
| Jw      | t25          |                    39   |                 3.5  | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Jw      | t50          |                    45.5 |                 5.75 | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Jw      | t75          |                    49   |                 5.5  | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| Jw      | peak_smooth3 |                    45.5 |                11    | strongly_heterogeneous_transition   | strongly_heterogeneous_transition   |
| P       | departure90  |                    37   |                 2    | coherent_feature_transition         | coherent_feature_transition         |
| P       | t25          |                    39   |                 2    | coherent_feature_transition         | coherent_feature_transition         |
| P       | t50          |                    42   |                 4    | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| P       | t75          |                    46   |                 1    | coherent_feature_transition         | coherent_feature_transition         |
| P       | peak_smooth3 |                    42   |                 5    | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| V       | departure90  |                    38   |                 2.75 | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| V       | t25          |                    41   |                 1    | coherent_feature_transition         | coherent_feature_transition         |
| V       | t50          |                    43.5 |                 2.75 | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| V       | t75          |                    46.5 |                 4    | moderately_heterogeneous_transition | moderately_heterogeneous_transition |
| V       | peak_smooth3 |                    45   |                 2.75 | moderately_heterogeneous_transition | moderately_heterogeneous_transition |

## H/Jw feature relation detail

| marker           |   median_delta_Jw_minus_H |   overlap_score |   fraction_H_features_earlier |   fraction_Jw_features_earlier |   fraction_near_equal | relation_label                 |
|:-----------------|--------------------------:|----------------:|------------------------------:|-------------------------------:|----------------------:|:-------------------------------|
| departure90      |                       0   |        0        |                      0        |                      0         |             0.743802  | feature_relation_mixed         |
| departure95      |                       0   |        0        |                      0        |                      0         |             0.743802  | feature_relation_mixed         |
| t10              |                       2   |        0        |                      1        |                      0.0909091 |             0.165289  | A_feature_distribution_earlier |
| t15              |                       2   |        0        |                      1        |                      0.0909091 |             0.132231  | A_feature_distribution_earlier |
| t20              |                       3   |        0        |                      1        |                      0.0909091 |             0.0826446 | A_feature_distribution_earlier |
| t25              |                       2   |        0        |                      1        |                      0.0909091 |             0.14876   | A_feature_distribution_earlier |
| t30              |                       2.5 |        0        |                      1        |                      0.0909091 |             0.107438  | A_feature_distribution_earlier |
| t35              |                       3   |        0        |                      1        |                      0.0909091 |             0.14876   | A_feature_distribution_earlier |
| t40              |                       4.5 |        0        |                      0.818182 |                      0.0909091 |             0.157025  | A_feature_distribution_earlier |
| t45              |                       6.5 |        0        |                      0.818182 |                      0         |             0.123967  | A_feature_distribution_earlier |
| t50              |                       6.5 |        0        |                      0.818182 |                      0         |             0.0991736 | A_feature_distribution_earlier |
| t75              |                       5   |        0.578947 |                      0.636364 |                      0.272727  |             0.190083  | feature_distributions_overlap  |
| peak_raw         |                      12   |        0.517857 |                      0.636364 |                      0         |             0.123967  | feature_distributions_overlap  |
| peak_smooth3     |                       9.5 |        0        |                      0.909091 |                      0         |             0.0661157 | A_feature_distribution_earlier |
| duration_25_75   |                      -0.5 |        0.386364 |                      0.454545 |                      0.454545  |             0.14876   | B_partial_feature_advantage    |
| tail_50_75       |                      -1   |        0.333333 |                      0.363636 |                      0.727273  |             0.404959  | B_feature_distribution_earlier |
| early_span_25_50 |                       2   |        0.166667 |                      0.727273 |                      0         |             0.305785  | A_feature_distribution_earlier |

## Early group feature organization

| field   | marker       | feature_timing_rank_distribution        |   prob_field_feature_distribution_earliest |   prob_field_feature_distribution_latest |   overlap_with_H |   overlap_with_Jw | early_group_role_label      | interpretation                                                                                |
|:--------|:-------------|:----------------------------------------|-------------------------------------------:|-----------------------------------------:|-----------------:|------------------:|:----------------------------|:----------------------------------------------------------------------------------------------|
| P       | departure90  | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0         | broad_or_mixed_feature_role | P departure90: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.  |
| P       | t25          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.222222  | broad_or_mixed_feature_role | P t25: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.          |
| P       | t50          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.392857  | broad_or_mixed_feature_role | P t50: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.          |
| P       | t75          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0.105263 |         0.0833333 | broad_or_mixed_feature_role | P t75: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.          |
| P       | peak_smooth3 | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.454545  | broad_or_mixed_feature_role | P peak_smooth3: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw. |
| V       | departure90  | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0         | broad_or_mixed_feature_role | V departure90: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.  |
| V       | t25          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.285714  | broad_or_mixed_feature_role | V t25: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.          |
| V       | t50          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.478261  | broad_or_mixed_feature_role | V t50: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw.          |
| V       | t75          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0.421053 |         0.583333  | early_group_overlap         | V t75: early_group_overlap; based on feature distribution overlap with H/Jw.                  |
| V       | peak_smooth3 | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         0.25      | broad_or_mixed_feature_role | V peak_smooth3: broad_or_mixed_feature_role; based on feature distribution overlap with H/Jw. |
| H       | departure90  | computed_by_feature_timing_distribution |                                        nan |                                      nan |         1        |         0         | early_group_overlap         | H departure90: early_group_overlap; based on feature distribution overlap with H/Jw.          |
| H       | t25          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         1        |         0         | early_group_overlap         | H t25: early_group_overlap; based on feature distribution overlap with H/Jw.                  |
| H       | t50          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         1        |         0         | early_group_overlap         | H t50: early_group_overlap; based on feature distribution overlap with H/Jw.                  |
| H       | t75          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         1        |         0.578947  | early_group_overlap         | H t75: early_group_overlap; based on feature distribution overlap with H/Jw.                  |
| H       | peak_smooth3 | computed_by_feature_timing_distribution |                                        nan |                                      nan |         1        |         0         | early_group_overlap         | H peak_smooth3: early_group_overlap; based on feature distribution overlap with H/Jw.         |
| Jw      | departure90  | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         1         | early_group_overlap         | Jw departure90: early_group_overlap; based on feature distribution overlap with H/Jw.         |
| Jw      | t25          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         1         | early_group_overlap         | Jw t25: early_group_overlap; based on feature distribution overlap with H/Jw.                 |
| Jw      | t50          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         1         | early_group_overlap         | Jw t50: early_group_overlap; based on feature distribution overlap with H/Jw.                 |
| Jw      | t75          | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0.578947 |         1         | early_group_overlap         | Jw t75: early_group_overlap; based on feature distribution overlap with H/Jw.                 |
| Jw      | peak_smooth3 | computed_by_feature_timing_distribution |                                        nan |                                      nan |         0        |         1         | early_group_overlap         | Jw peak_smooth3: early_group_overlap; based on feature distribution overlap with H/Jw.        |

## Je feature consistency as background reference

| marker       |   Je_feature_median |   Je_iqr |   late_feature_fraction |   dominant_late_feature_contribution | Je_late_consistency_label   | interpretation                                                                                     |
|:-------------|--------------------:|---------:|------------------------:|-------------------------------------:|:----------------------------|:---------------------------------------------------------------------------------------------------|
| departure90  |                  37 |        8 |                0.545455 |                                  nan | Je_feature_mixed_late       | Je departure90: Je_feature_mixed_late; this is a consistency check, not the main research target.  |
| t25          |                  36 |        1 |                0.363636 |                                  nan | Je_feature_late_consistent  | Je t25: Je_feature_late_consistent; this is a consistency check, not the main research target.     |
| t50          |                  41 |        6 |                0.363636 |                                  nan | Je_feature_mixed_late       | Je t50: Je_feature_mixed_late; this is a consistency check, not the main research target.          |
| t75          |                  52 |        3 |                0.363636 |                                  nan | Je_feature_late_consistent  | Je t75: Je_feature_late_consistent; this is a consistency check, not the main research target.     |
| peak_smooth3 |                  32 |       14 |                0.545455 |                                  nan | Je_feature_mixed_late       | Je peak_smooth3: Je_feature_mixed_late; this is a consistency check, not the main research target. |

## Prohibited interpretations

- Feature-level relation is not causality.
- Feature timing overlap is not synchronization unless a separate equivalence test is defined.
- Low-contribution or noisy features are retained and labelled, not deleted.
- Weighted and unweighted distributions must not be mixed.
- If feature metadata are insufficient, do not write physical-region conclusions.
