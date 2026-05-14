# V10.7_f W45 H-package to main-cluster cross-object audit

## Method boundary
- This is a yearwise cross-object association / incremental-explanatory audit.
- It does not test H35 as a single point; H35 was already closed as a stable-independent target by V10.7_d.
- It does not infer causality. Positive results mean incremental yearwise association only.
- Field-domain event strengths are proxy indices; missing object fields are skipped and logged.

## Input coverage
| object   | status   | field_key       | lat_range    | lon_range      | note                                                |   n_years |   n_days |   n_lat |   n_lon |   added_year_axis |
|:---------|:---------|:----------------|:-------------|:---------------|:----------------------------------------------------|----------:|---------:|--------:|--------:|------------------:|
| H        | loaded   | z500_smoothed   | (15.0, 35.0) | (110.0, 140.0) |                                                     |        45 |      183 |      81 |     121 |                 0 |
| P        | loaded   | precip_smoothed | (15.0, 35.0) | (110.0, 140.0) |                                                     |        45 |      183 |      81 |     121 |                 0 |
| V        | loaded   | v850_smoothed   | (15.0, 35.0) | (110.0, 140.0) |                                                     |        45 |      183 |      81 |     121 |                 0 |
| Je       | missing  |                 | (25.0, 45.0) | (110.0, 150.0) | No matching field key. This object will be skipped. |       nan |      nan |     nan |     nan |               nan |
| Jw       | missing  |                 | (25.0, 45.0) | (60.0, 110.0)  | No matching field key. This object will be skipped. |       nan |      nan |     nan |     nan |               nan |

## Route decision
| decision_item   | status                               | evidence                                                                                 | route_implication                                                                                           |
|:----------------|:-------------------------------------|:-----------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------|
| H package line  | downgrade_H_for_W45_main_explanation | No target shows incremental support from H_package after E2 controls under primary mode. | Shift W45 explanation toward P/V/Je/Jw main-cluster structure; do not continue H package as a primary line. |
| input coverage  | partial_inputs                       | Je:missing; Jw:missing                                                                   | Interpret missing-object targets/controls as unavailable, not as negative evidence.                         |

## Primary-mode incremental table
| target                 |   n_years_valid |   delta_r2 |   delta_cv_rmse_positive_means_improvement |   H_package_coef_loo_sign_stability |   permutation_p_delta_r2 | decision               |
|:-----------------------|----------------:|-----------:|-------------------------------------------:|------------------------------------:|-------------------------:|:-----------------------|
| joint45_strength_proxy |              45 | 0.0103964  |                                 -0.0146536 |                                   1 |                 0.48503  | no_incremental_support |
| P_P45_strength         |              45 | 0.0153475  |                                 -0.0102578 |                                   1 |                 0.449102 | no_incremental_support |
| V_V45_strength         |              45 | 0.00426803 |                                 -0.0205868 |                                   1 |                 0.674651 | no_incremental_support |
| M_combined_strength    |              45 | 0.0103964  |                                 -0.0146536 |                                   1 |                 0.516966 | no_incremental_support |