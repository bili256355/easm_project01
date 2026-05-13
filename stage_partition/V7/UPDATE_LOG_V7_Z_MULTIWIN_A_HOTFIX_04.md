# UPDATE_LOG_V7_Z_MULTIWIN_A_HOTFIX_04

## Purpose

Fix a non-equivalent detector-input change introduced during multiwindow extraction. Earlier multiwin-a hotfixes restored `ruptures.Window` but still passed raw climatological profile matrices directly into the detector. Original V7-z W45 raw/profile detection used feature-wise z-scored climatological profiles.

## Changes

- Added `_zscore_features_v7z()`.
- Added `_raw_state_matrix_v7z_from_year_cube()`.
- Observed detector input is now `zscore(nanmean(year_cube, axis=0))`.
- Bootstrap detector input uses the same sampled-year climatology then feature-wise z-score.
- Tightened `_is_system_relevant_candidate()` so peak-day relevance controls current-window main selection. Broad candidate-band overlap alone no longer makes a far-pre/far-post peak system-relevant.
- Changed output tag to `accepted_windows_multi_object_prepost_v7_z_multiwin_a_hotfix04_w45_regression` to avoid stale files from older runs.
- Added W045 regression audit against original V7-z W45 reference peak days.

## Not changed

- Hardcoded significant windows.
- Window scope rules.
- C0/C1/C2 baseline definitions.
- early/core/late segment rules.
- profile pre-post S_dist/R_diff/S_pattern definitions.
- 2D mirror definitions.
- hardened evidence gate definitions.

## Required verification

Run default W045 profile-only mode first and inspect:

- `per_window/W045/object_profile_window_registry_W045.csv`
- `per_window/W045/main_window_selection_W045.csv`
- `cross_window/W045_regression_against_v7z_reference_v7_z_multiwin_a_hotfix04.csv`

Only after W045 profile regression is acceptable should `V7_MULTI_WINDOW_MODE=all` or `V7_MULTI_RUN_2D=1` be used.
