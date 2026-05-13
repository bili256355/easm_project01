# UPDATE_LOG_V7_L

## w45_allfield_timing_marker_audit_v7_l hotfix 01

Purpose: fix output labeling and summary interpretation for W45 all-field timing-marker audit.

Changes:

1. Marker ties are no longer silently promoted to onset.
   - Adds `marker_selection_type`.
   - Adds `tie_markers`.
   - Adds `unique_best_marker`.
   - Uses `recommended_marker = marker_tie` when onset/midpoint/finish tie.

2. Marker comparability now treats tied-marker fields as not directly comparable for onset-order or midpoint-order.

3. Window summary now reports:
   - `n_fields_onset_unique_best`
   - `n_fields_midpoint_unique_best`
   - `n_fields_finish_unique_best`
   - `n_fields_marker_tie`
   - `H_statistically_confirmed_broad_transition`

4. Broad-transition interpretation is split into:
   - observed/candidate transition shape;
   - statistically confirmed duration/finish-tail broadness.

5. H-specific duration/finish-tail diagnostics are added to avoid overstating broad-transition significance.

No change:

- Does not recompute V7-e progress.
- Does not change onset/midpoint/finish extraction.
- Does not change 90%/95% CI logic.
- Does not change input/output directory names.
- Does not add new scientific claims.
