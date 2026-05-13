# UPDATE_LOG_V7_W

## V7-w: H/Jw object-specific V6_1-style window detection

### Purpose
This patch adds a new V7-w diagnostic branch to reuse the original W45 detection skeleton while changing only:

1. `object_scope`: from all-object system state matrix to H-only and Jw-only object runs.
2. `detector_day_range`: day0–70.

### What is intentionally reused from the W45/V6_1 method
- 2-degree latitudinal profile representation.
- Full-season day-wise z-score feature standardization.
- Block equal contribution scaling; with one object this becomes the same object-block scaling logic.
- `ruptures.Window` detector with `model=l2`, `width=20`, `min_size=2`, `jump=1`, `pen=4.0`.
- Local peak extraction and candidate registry.
- Candidate band construction with V6_1-style band rules.
- Window merging and bootstrap peak matching logic.

### What is not included
- No pre/post progress metrics.
- No `S_dist`, `S_pattern`, or growth speed curves.
- No raw025 2D field-speed detector.
- No P/V/Je.
- No spatial/lat-band pairing.
- No causality or physical interpretation.

### Output tag
`w45_H_Jw_object_specific_v6style_window_detection_v7_w`

### Main entry
`stage_partition/V7/scripts/run_w45_H_Jw_object_specific_v6style_window_detection_v7_w.py`


## V7-w hotfix_01: feature_table duplicate object column

### Fixed
- Prevented duplicate insertion of the `object` column in `_build_object_outputs()`.
- `_build_object_state()` already records `object` in `feature_table`; the hotfix now checks the column before insertion and preserves column order.

### Impact
- Execution-chain hotfix only.
- No change to detector inputs, parameters, peak logic, band logic, bootstrap logic, or output semantics.
