# UPDATE_LOG_V7_Y

## V7-y: H/Jw pattern-only detector with bootstrap

Purpose: test whether the Jw day30–39 early pattern signal can be detected from the original profile's shape layer, rather than only from the pre/post-derived R_diff trajectory.

Implemented:
- New entry script: `scripts/run_H_Jw_pattern_only_detector_v7_y.py`
- New module: `src/stage_partition_v7/H_Jw_pattern_only_detector_v7_y.py`
- Output directory: `outputs/H_Jw_pattern_only_detector_v7_y`
- Uses H-only and Jw-only 2° profiles over day0–70.
- Converts raw profiles to shape-normalized profiles by removing the weighted latitudinal mean and normalizing by the weighted latitudinal norm for each year/day.
- Reuses V6/V7-w style `ruptures.Window`, local peaks, candidate bands/windows, and bootstrap matching.

Scope:
- Does not use pre/post progress, S_dist/S_pattern, or R_diff as the main detector input.
- Does not add P/V/Je.
- Does not infer pathway/causality.
