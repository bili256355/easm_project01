# W45 2D-field pre-post metric mirror (V7-z-2d-a)
## Purpose
This is a minimal 2D-field mirror of the profile-based W45 clean mainline. It computes 2D analogues of S_dist, R_diff/S_pattern, and growth metrics. It does not run 2D change-point detection and does not overwrite clean-mainline final claims.
## Method boundary
- Timing anchor / clean mainline: profile-based.
- This audit basis: full object-specific 2D regional fields.
- Outputs here are comparison and support diagnostics, not final timing claims.
## Input status
- Clean profile curves status: `loaded`.
## Main output files
- `W45_2d_state_progress_curves_v7_z_2d_a.csv`
- `W45_2d_growth_speed_curves_v7_z_2d_a.csv`
- `W45_2d_single_object_metric_summary_v7_z_2d_a.csv`
- `W45_2d_pairwise_metric_summary_v7_z_2d_a.csv`
- `W45_profile_vs_2d_prepost_metric_comparison_v7_z_2d_a.csv`
## Profile-vs-2D agreement overview
- consistent: 203
- similar_tendency: 34
- unresolved: 30
- same_side_different_magnitude: 2
- profile_2d_offset: 1
## Notes for interpretation
- `S_pattern_2d` is a weighted 2D field correlation-based pre/post similarity progress, not a lat-profile correlation.
- Disagreement between profile and 2D metrics should be treated as an audit signal, not as an automatic refutation of the profile clean mainline.
- No 2D object-window or final claim is generated in this version.
