# V10-a hotfix02 main-treed independent subpeak/peak reproduction summary

version: `v10_peak_subpeak_reproduce_a_hotfix02_main`
output_tag: `peak_subpeak_reproduce_v10_a`

## Purpose
V10-a hotfix02 main-treed audit independently reimplements the V9 subpeak/peak extraction layer.
It does not import or call V9 or V7 code. Existing V9 CSVs are read only for regression audit.

## Windows processed
- W045: anchor=45, system=40-48, detector=0-69
- W081: anchor=81, system=75-87, detector=49-102
- W113: anchor=113, system=108-118, detector=88-149
- W160: anchor=160, system=155-165, detector=119-182

## Method boundary
- This is a reproduction/extraction layer, not a new sensitivity audit.
- This is not a physical subpeak classification layer.
- This does not redefine accepted windows.
- It should first be judged by regression against V9 outputs.

## Expected core outputs
- object_profile_window_registry_all_windows.csv: all detected candidate subpeaks.
- main_window_selection_all_windows.csv: V9-equivalent selected main candidate per object/window.
- raw_profile_detector_scores_all_windows.csv: detector score landscape.
- bootstrap_selected_peak_days_all_windows.csv: paired-year bootstrap selected days.
- audit/v10_vs_v9_subpeak_regression_audit.csv: read-only comparison against V9 CSVs when available, with diff details stored under the same bundle.