# V10.7_m window-anchor information decomposition summary

This file is a diagnostic summary, not a paper conclusion. It separates direct outputs, derived judgments, allowed interpretations, forbidden interpretations, and next-step implications.


## Q2 direct output: information form route decision

| mode                     | information_form   |   mean_primary_score |   n_support |   mean_abs_spearman | targets                                                                                                                  |   rank_by_mode | route_decision      | top_information_form   |
|:-------------------------|:-------------------|---------------------:|------------:|--------------------:|:-------------------------------------------------------------------------------------------------------------------------|---------------:|:--------------------|:-----------------------|
| anomaly                  | abs_transition     |            0.0257327 |           0 |           0.0232581 | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              6 | slope_dominant      | slope                  |
| anomaly                  | post_state         |            2.406     |           4 |           0.341686  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              3 | slope_dominant      | slope                  |
| anomaly                  | pre_state          |            0.336056  |           0 |           0.300362  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              4 | slope_dominant      | slope                  |
| anomaly                  | signed_transition  |            2.53777   |           4 |           0.466447  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              2 | slope_dominant      | slope                  |
| anomaly                  | slope              |            2.55922   |           4 |           0.481223  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              1 | slope_dominant      | slope                  |
| anomaly                  | window_mean        |            0.0487944 |           0 |           0.0465507 | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              5 | slope_dominant      | slope                  |
| local_background_removed | abs_transition     |            0.703001  |           1 |           0.18155   | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              4 | post_state_dominant | post_state             |
| local_background_removed | post_state         |            2.30544   |           4 |           0.254633  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              1 | post_state_dominant | post_state             |
| local_background_removed | pre_state          |            0.142655  |           0 |           0.139043  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              5 | post_state_dominant | post_state             |
| local_background_removed | signed_transition  |            2.02238   |           3 |           0.228786  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              2 | post_state_dominant | post_state             |
| local_background_removed | slope              |            0.999985  |           1 |           0.219012  | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              3 | post_state_dominant | post_state             |
| local_background_removed | window_mean        |            0.0688836 |           0 |           0.0653513 | P_centroid_lat_transition;P_main_band_share_transition;P_main_minus_south_transition;P_south_band_share_18_24_transition |              6 | post_state_dominant | post_state             |



## Q1/Q4 direct output: E2 anchor rank summary

| mode    | anchor_name         | anchor_type   |   anchor_start |   anchor_end | information_form   |   mean_score |   mean_abs_spearman |   n_targets |   E2_mean_score |   E2_percentile_among_sliding_windows |   E2_better_than_n_named_controls |   n_named_controls | anchor_decision                       |
|:--------|:--------------------|:--------------|---------------:|-------------:|:-------------------|-------------:|--------------------:|------------:|----------------:|--------------------------------------:|----------------------------------:|-------------------:|:--------------------------------------|
| anomaly | E1_W18              | named         |             12 |           23 | abs_transition     |    0.103148  |           0.101354  |           3 |       0.0307866 |                               6.89655 |                                 0 |                  3 | not_E2_specific                       |
| anomaly | E1_W18              | named         |             12 |           23 | post_state         |    0.0200567 |           0.0189647 |           3 |       0.332531  |                              89.6552  |                                 2 |                  3 | broad_or_unclear_preseason_background |
| anomaly | E1_W18              | named         |             12 |           23 | pre_state          |    0.188316  |           0.182493  |           3 |       0.316464  |                              93.1034  |                                 3 |                  3 | E2_anchor_partial                     |
| anomaly | E1_W18              | named         |             12 |           23 | signed_transition  |    0.154953  |           0.149281  |           3 |       0.469702  |                              96.5517  |                                 2 |                  3 | E2_anchor_supported                   |
| anomaly | E1_W18              | named         |             12 |           23 | slope              |    0.173003  |           0.166107  |           3 |       0.484905  |                             100       |                                 3 |                  3 | E2_anchor_supported                   |
| anomaly | E1_W18              | named         |             12 |           23 | window_mean        |    0.177616  |           0.170703  |           3 |       0.0493862 |                              10.3448  |                                 1 |                  3 | not_E2_specific                       |
| anomaly | E2_W33              | named         |             27 |           38 | abs_transition     |    0.0307866 |           0.0301323 |           3 |       0.0307866 |                               6.89655 |                                 0 |                  3 | not_E2_specific                       |
| anomaly | E2_W33              | named         |             27 |           38 | post_state         |    0.332531  |           0.32064   |           3 |       0.332531  |                              89.6552  |                                 2 |                  3 | broad_or_unclear_preseason_background |
| anomaly | E2_W33              | named         |             27 |           38 | pre_state          |    0.316464  |           0.308908  |           3 |       0.316464  |                              93.1034  |                                 3 |                  3 | E2_anchor_partial                     |
| anomaly | E2_W33              | named         |             27 |           38 | signed_transition  |    0.469702  |           0.456553  |           3 |       0.469702  |                              96.5517  |                                 2 |                  3 | E2_anchor_supported                   |
| anomaly | E2_W33              | named         |             27 |           38 | slope              |    0.484905  |           0.470286  |           3 |       0.484905  |                             100       |                                 3 |                  3 | E2_anchor_supported                   |
| anomaly | E2_W33              | named         |             27 |           38 | window_mean        |    0.0493862 |           0.0491314 |           3 |       0.0493862 |                              10.3448  |                                 1 |                  3 | not_E2_specific                       |
| anomaly | E2_shift_left       | named         |             22 |           33 | abs_transition     |    0.12256   |           0.120619  |           3 |       0.0307866 |                               6.89655 |                                 0 |                  3 | not_E2_specific                       |
| anomaly | E2_shift_left       | named         |             22 |           33 | post_state         |    0.259684  |           0.250343  |           3 |       0.332531  |                              89.6552  |                                 2 |                  3 | broad_or_unclear_preseason_background |
| anomaly | E2_shift_left       | named         |             22 |           33 | pre_state          |    0.164073  |           0.159007  |           3 |       0.316464  |                              93.1034  |                                 3 |                  3 | E2_anchor_partial                     |
| anomaly | E2_shift_left       | named         |             22 |           33 | signed_transition  |    0.171922  |           0.167144  |           3 |       0.469702  |                              96.5517  |                                 2 |                  3 | E2_anchor_supported                   |
| anomaly | E2_shift_left       | named         |             22 |           33 | slope              |    0.167748  |           0.161054  |           3 |       0.484905  |                             100       |                                 3 |                  3 | E2_anchor_supported                   |
| anomaly | E2_shift_left       | named         |             22 |           33 | window_mean        |    0.287578  |           0.281296  |           3 |       0.0493862 |                              10.3448  |                                 1 |                  3 | not_E2_specific                       |
| anomaly | E2_shift_right_safe | named         |             28 |           39 | abs_transition     |    0.108766  |           0.102277  |           3 |       0.0307866 |                               6.89655 |                                 0 |                  3 | not_E2_specific                       |
| anomaly | E2_shift_right_safe | named         |             28 |           39 | post_state         |    0.382622  |           0.37319   |           3 |       0.332531  |                              89.6552  |                                 2 |                  3 | broad_or_unclear_preseason_background |



## Q3 direct output: scalarized vs signed route decision

| mode                     | representation_type         |   mean_primary_score |   n_support |   mean_abs_spearman | representation_route_decision   |
|:-------------------------|:----------------------------|---------------------:|------------:|--------------------:|:--------------------------------|
| anomaly                  | scalarized_transition       |             0.140309 |           0 |           0.137974  | scalarization_loss_supported    |
| anomaly                  | signed_component_transition |             0.768687 |           4 |           0.205991  | scalarization_loss_supported    |
| local_background_removed | scalarized_transition       |             0.101075 |           0 |           0.0991802 | scalarization_loss_supported    |
| local_background_removed | signed_component_transition |             0.50424  |           2 |           0.165129  | scalarization_loss_supported    |



## Q5-min direct output: metric specificity decision

| mode                     | top_source_metric   | top_target_family   | metric_specificity_decision   |
|:-------------------------|:--------------------|:--------------------|:------------------------------|
| anomaly                  | H_zonal_width       | P_position          | target_not_rainband_specific  |
| local_background_removed | H_zonal_width       | P_position          | target_not_rainband_specific  |



## Global forbidden interpretations

- Do not state H causes P.

- Do not state H controls W45.

- Do not generalize to full W33-to-W45 mapping.

- Do not say the strength class failed; V10.7_i is a scalarized-transition limitation.

- Do not say transition windows represent most object information.
