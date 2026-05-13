# V1 all-window positive_lag_not_supported decomposition audit log

Input: `D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv`
Output: `D:\easm_project01\lead_lag_screen\V1\positive_lag_not_supported_decomposition_all_windows_a\outputs`

This audit decomposes rows already labelled `positive_lag_not_supported` for all windows, so T3/T4 can be compared against S/T windows under the same diagnostic buckets.
It is a read-only statistical/diagnostic localization layer and does not prove the physical cause of window-level relationship drop.

Default diagnostic thresholds:
- weak_abs_r < 0.1
- moderate_abs_r >= 0.2
- surrogate_alpha = 0.1
- fdr_alpha = 0.1
- audit_surrogate_alpha = 0.1

Interpretation caution: blocker labels are derived diagnostic buckets, not original V1 decisions and not physical explanations.