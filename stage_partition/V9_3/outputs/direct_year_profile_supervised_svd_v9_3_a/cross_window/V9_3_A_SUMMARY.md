# V9.3_a result summary

This is a direct-year profile-evolution supervised SVD / PLS1 audit.
It uses yearly peak_B - peak_A as Y, so yearly peak quality is a core risk.

## Method boundaries
- No 2D-field X is used.
- No bootstrap year-resampling is used as samples.
- The supervised direction is target-guided, not an unsupervised natural mode.
- Result extraction should use result_usability_level, not raw corr_score_y alone.

## Settings
- target_mode: priority_and_full_pairs
- perm_n: 500
- split_half_n: 100

## Usability counts
- Level_C_exploratory: 41
- Level_B_candidate: 2

## Top candidate rows by corr_score_y

| window_id   | target_pair   | target_set            |   corr_score_y |    perm_p | result_usability_level   | downgrade_reasons                                                                |
|:------------|:--------------|:----------------------|---------------:|----------:|:-------------------------|:---------------------------------------------------------------------------------|
| W113        | V-Je          | priority              |       0.823625 | 0.0179641 | Level_B_candidate        | warning_many_low_quality_peaks;split_half_marginal                               |
| W160        | P-V           | priority              |       0.813732 | 0.0359281 | Level_B_candidate        | warning_many_low_quality_peaks;split_half_unstable_or_unclear                    |
| W045        | V-Jw          | priority              |       0.776181 | 0.159681  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W081        | P-V           | priority              |       0.775405 | 0.0678643 | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W113        | P-V           | priority              |       0.770159 | 0.161677  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W160        | V-Jw          | exploratory_full_pair |       0.766199 | 0.179641  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W160        | Jw-V          | priority              |       0.766199 | 0.181637  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W160        | P-H           | exploratory_full_pair |       0.763018 | 0.215569  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W160        | P-Jw          | exploratory_full_pair |       0.760053 | 0.211577  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W045        | P-H           | exploratory_full_pair |       0.757892 | 0.253493  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W045        | P-Jw          | priority              |       0.756301 | 0.251497  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W160        | H-Jw          | priority              |       0.754518 | 0.259481  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W081        | V-Je          | exploratory_full_pair |       0.743788 | 0.171657  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W113        | H-Je          | priority              |       0.740197 | 0.397206  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W045        | H-Jw          | exploratory_full_pair |       0.734225 | 0.411178  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W160        | Je-Jw         | exploratory_full_pair |       0.734158 | 0.419162  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W113        | V-Jw          | exploratory_full_pair |       0.733328 | 0.469062  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W113        | Jw-V          | priority              |       0.733328 | 0.459082  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_unstable_or_unclear |
| W081        | V-Jw          | priority              |       0.731404 | 0.253493  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |
| W045        | H-Je          | exploratory_full_pair |       0.729523 | 0.479042  | Level_C_exploratory      | warning_many_low_quality_peaks;permutation_not_95;split_half_marginal            |

## Interpretation restrictions
- Do not call score high/low groups physical year types without separate physical audit.
- Do not interpret dominant_object as a physical driver.
- Do not use Level C/D rows as main results.