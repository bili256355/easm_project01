# UPDATE_LOG_V7_Z_2D_A

## V7-z-2d-a — W45 2D-field pre-post metric mirror

Purpose: add a minimal 2D-field mirror of the clean profile-based W45 mainline.

Implemented:

- New entry script:
  - `stage_partition/V7/scripts/run_W45_2d_prepost_metric_mirror_v7_z_2d_a.py`
- New module:
  - `stage_partition/V7/src/stage_partition_v7/W45_2d_prepost_metric_mirror_v7_z_2d_a.py`
- Computes 2D-field analogues of the clean mainline pre-post metrics:
  - `S_dist_2d`
  - `R_pre_2d`, `R_post_2d`, `R_diff_2d`
  - `S_pattern_2d`
  - `V_dist_2d`, `V_pattern_2d`
- Uses the same W45 baseline definitions as the clean mainline:
  - C0: day0–39 / day49–74
  - C1: day0–34 / day54–69
  - C2: day25–34 / day54–69
- Uses the same object regions as the clean mainline:
  - P, V, H, Je, Jw
- Uses paired year-bootstrap for 2D metric uncertainty.
- Writes a profile-vs-2D comparison table when clean profile outputs are available.

Explicitly not implemented:

- No 2D change-point detection.
- No 2D before-after spatial maps.
- No spatial center/variance/deep physical decomposition beyond the mirror metrics.
- No final claim rewrite.
- No replacement of the clean profile-based mainline.

Interpretation boundary:

- Clean profile mainline remains the timing/claim source.
- V7-z-2d-a only audits whether the profile-based pre-post metrics have corresponding 2D-field metric behavior.
