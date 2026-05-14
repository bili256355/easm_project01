# V10.7_c H W045 Event-Content Audit Summary
## 1. Method boundary
This run audits what H changes occur around H18/H35/H45/H57. It is not an influence test, not a causal test, not a lead-lag test, and not a detector rerun.
It should be used to decide what to test next, not to claim H18→H35 or H35→W045.

HOTFIX01 note: full-domain spatial composites are retained as background/context. H-object-domain spatial metrics and similarity are added and used as the primary spatial evidence for role classification. H18/H19 scale-context aliasing is also fixed.
## 2. Input status
- profile_status: reconstructed_from_smoothed_fields
- spatial_status: loaded
- v10_7_b_scale_context: loaded
- yearwise_status: computed

## 3. Event-content role summary
| event_id   | profile_strength_class    | spatial_strength_class    | yearwise_consistency_class   | scale_ridge_context_from_v10_7_b                                                        | content_role_class                    | recommended_next_test_target                                                                    |
|:-----------|:--------------------------|:--------------------------|:-----------------------------|:----------------------------------------------------------------------------------------|:--------------------------------------|:------------------------------------------------------------------------------------------------|
| H18        | profile_strong_relative   | spatial_strong_relative   | moderately_consistent        | ridge_near_H18;v10_7_b_target=H19;role_hint=candidate_scale_structure;persistence=1.0   | strong_early_H_adjustment             | Use H18/H18-26 package as primary early-H target for later yearwise/spatial relationship tests. |
| H35        | profile_moderate_relative | spatial_moderate_relative | moderately_consistent        | ridge_near_H35;v10_7_b_target=H35;role_hint=candidate_scale_structure;persistence=0.375 | same_type_second_H_adjustment         | Test H18-H35 package rather than H35 alone.                                                     |
| H45        | profile_weak_relative     | spatial_weak_relative     | moderately_consistent        | no_clear_scale_ridge_near_target                                                        | H_absent_in_W045_main_cluster         | Use as negative/control target for H absence in W045 main cluster.                              |
| H57        | profile_moderate_relative | spatial_moderate_relative | moderately_consistent        | no_clear_scale_ridge_near_target                                                        | weak_or_unclear_post_window_reference | Use only as post-window reference target.                                                       |

## 4. H18/H35/H45/H57 similarity — H-object domain primary
| comparison   | spatial_domain_used   |   profile_pearson_correlation |   spatial_pattern_correlation |   profile_top_feature_overlap_top5 | interpretation_hint                            |
|:-------------|:----------------------|------------------------------:|------------------------------:|-----------------------------------:|:-----------------------------------------------|
| H18_vs_H35   | h_object_domain       |                     0.956207  |                      0.948877 |                                1   | similar_content_candidate_same_type_adjustment |
| H18_vs_H45   | h_object_domain       |                    -0.0437291 |                     -0.239233 |                                0.6 | low_similarity_candidate_distinct_content      |
| H35_vs_H45   | h_object_domain       |                    -0.195451  |                     -0.268854 |                                0.6 | low_similarity_candidate_distinct_content      |
| H35_vs_H57   | h_object_domain       |                     0.965018  |                      0.954081 |                                1   | similar_content_candidate_same_type_adjustment |
| H18_vs_H57   | h_object_domain       |                     0.988499  |                      0.940887 |                                1   | similar_content_candidate_same_type_adjustment |

## 5. H18/H35/H45/H57 similarity — full-domain context
| comparison   | spatial_domain_used   |   profile_pearson_correlation |   spatial_pattern_correlation |   profile_top_feature_overlap_top5 | interpretation_hint                            |
|:-------------|:----------------------|------------------------------:|------------------------------:|-----------------------------------:|:-----------------------------------------------|
| H18_vs_H35   | full_domain_context   |                     0.956207  |                      0.830904 |                                1   | similar_content_candidate_same_type_adjustment |
| H18_vs_H45   | full_domain_context   |                    -0.0437291 |                      0.442587 |                                0.6 | low_similarity_candidate_distinct_content      |
| H35_vs_H45   | full_domain_context   |                    -0.195451  |                      0.330469 |                                0.6 | low_similarity_candidate_distinct_content      |
| H35_vs_H57   | full_domain_context   |                     0.965018  |                      0.822752 |                                1   | similar_content_candidate_same_type_adjustment |
| H18_vs_H57   | full_domain_context   |                     0.988499  |                      0.81005  |                                1   | similar_content_candidate_same_type_adjustment |

## 6. Spatial metrics — H-object domain primary
| event_id   | domain_label    |   domain_lat_min |   domain_lat_max |   domain_lon_min |   domain_lon_max |   field_diff_abs_mean |   field_diff_max |   field_diff_min |   dominant_positive_lat |   dominant_positive_lon |   dominant_negative_lat |   dominant_negative_lon |
|:-----------|:----------------|-----------------:|-----------------:|-----------------:|-----------------:|----------------------:|-----------------:|-----------------:|------------------------:|------------------------:|------------------------:|------------------------:|
| H18        | h_object_domain |               15 |               35 |              110 |              140 |               8.32923 |         21.3461  |         0.462315 |                   35    |                  129.75 |                    16   |                  140    |
| H35        | h_object_domain |               15 |               35 |              110 |              140 |               6.38815 |         14.5521  |        -2.64949  |                   33.75 |                  132.75 |                    15.5 |                  124.25 |
| H45        | h_object_domain |               15 |               35 |              110 |              140 |               1.65001 |          7.3475  |        -4.634    |                   35    |                  110    |                    32   |                  129.25 |
| H57        | h_object_domain |               15 |               35 |              110 |              140 |               3.12078 |          9.77622 |        -0.977543 |                   35    |                  134.75 |                    15   |                  119.5  |

## 7. Spatial metrics — full-domain background/context
| event_id   | domain_label        |   domain_lat_min |   domain_lat_max |   domain_lon_min |   domain_lon_max |   field_diff_abs_mean |   field_diff_max |   field_diff_min |   dominant_positive_lat |   dominant_positive_lon |   dominant_negative_lat |   dominant_negative_lon |
|:-----------|:--------------------|-----------------:|-----------------:|-----------------:|-----------------:|----------------------:|-----------------:|-----------------:|------------------------:|------------------------:|------------------------:|------------------------:|
| H18        | full_domain_context |               10 |               60 |               20 |              150 |              10.5766  |          32.2144 |         -1.68837 |                    60   |                   117.5 |                    13.5 |                   50    |
| H35        | full_domain_context |               10 |               60 |               20 |              150 |              10.8531  |          37.4424 |         -2.64949 |                    60   |                    79   |                    15.5 |                  124.25 |
| H45        | full_domain_context |               10 |               60 |               20 |              150 |               8.58307 |          28.1144 |         -9.83224 |                    48.5 |                   123.5 |                    55.5 |                   66.75 |
| H57        | full_domain_context |               10 |               60 |               20 |              150 |               8.59578 |          35.2121 |         -6.55377 |                    60   |                   119   |                    16.5 |                   63.25 |

## 8. Yearwise consistency
| event_id   |   n_years |   median_pattern_corr |   fraction_positive_pattern_corr | yearwise_consistency_class   |
|:-----------|----------:|----------------------:|---------------------------------:|:-----------------------------|
| H18        |        45 |              0.269946 |                         0.844444 | moderately_consistent        |
| H35        |        45 |              0.328886 |                         0.733333 | moderately_consistent        |
| H45        |        45 |              0.293368 |                         0.844444 | moderately_consistent        |
| H57        |        45 |              0.336559 |                         0.844444 | moderately_consistent        |

## 9. Forbidden interpretations
- Do not claim H18 influences H35 from this run.
- Do not claim H35 influences W045 from this run.
- Do not call H35 a confirmed weak precursor based on this run alone.
- Do not infer causality or a physical pathway from profile/spatial composite content alone.
- Do not treat full-domain z500 composites as H-object evidence without checking the H-object-domain metrics.

## 10. Output notes
Profile diffs are computed from H object profile/state reconstruction. Spatial maps are computed only if H/z500 field and lat/lon are detected in smoothed_fields.npz. Yearwise results are computed only if a year dimension is detected.
