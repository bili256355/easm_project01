# W45 H raw025 three-band progress V7-j
Created: 2026-05-02T17:30:35
## Purpose
This audit avoids the current 2-degree interpolation profile and builds three equal-count latitude regions directly from raw latitude points in the H/z500 field. It checks whether W45-H stability improves when the analysis unit is a fair raw-resolution region-vector rather than a post-hoc 2-degree feature.
## Window and construction
- Window: W002 / anchor=45
- Analysis: 30–60
- Pre: 30–37
- Post: 53–60
- Uses 2-degree interpolation: **False**
- Region construction: raw025_equal_count_lat_partition_after_lon_mean

## Region definitions
- R1_low: 15.0–21.5°N, n_raw_lat_points=27
- R2_mid: 21.75–28.25°N, n_raw_lat_points=27
- R3_high: 28.5–35.0°N, n_raw_lat_points=27

## Bootstrap summary
- R1_low: midpoint median=39.0, q05–q95=37.0–52.0, q90_width=15.0, quality=nonmonotonic_progress
- R2_mid: midpoint median=39.0, q05–q95=37.0–49.0, q90_width=12.0, quality=nonmonotonic_progress
- R3_high: midpoint median=41.0, q05–q95=37.0–51.0, q90_width=14.0, quality=monotonic_clear_progress

## Pairwise diagnostic
- R1_low vs R2_mid: median Δ(B-A)=0.0, q05–q95=-13.0–3.0, pass90=False, pass95=False
- R1_low vs R3_high: median Δ(B-A)=1.0, q05–q95=-11.0–12.0, pass90=False, pass95=False
- R2_mid vs R3_high: median Δ(B-A)=1.0, q05–q95=-1.0–12.0, pass90=False, pass95=False

## Upstream implication
- raw025_threeband_improves_midpoint_width: no — median_q90_width_raw025_threeband=14.0; v7f_interp=14.0; v7i_latbin=14.0
- raw025_threeband_progress_quality: limited — n_nonmonotonic_regions=2 of 3

## Interpretation guardrails
- This audit does not use the 2-degree interpolation profile.
- This audit does not infer causality or pathway.
- Passing or failing here diagnoses W45-H implementation, not other windows or fields.
- If raw025 three-region results remain unstable, the problem is unlikely to be solved by simply replacing interpolation with raw-lat fair regional averaging.
