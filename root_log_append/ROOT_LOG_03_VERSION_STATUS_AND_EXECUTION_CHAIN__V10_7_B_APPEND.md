# ROOT_LOG_03 append: V10.7_b H W045 dedicated scale diagnostic

## V10.7_b status

- Version: `V10.7_b`
- Entry: `stage_partition/V10/v10.7/scripts/run_h_w045_scale_diagnostic_v10_7_b.py`
- Output: `stage_partition/V10/v10.7/outputs/h_w045_scale_diagnostic_v10_7_b`
- Scope: H only, W045 and pre-W045 region.
- Method role: dedicated Gaussian derivative scale-space diagnostic on the H object state matrix.
- State: implemented as a heuristic diagnostic / follow-up target selection layer.

## Execution semantics

V10.7_b does **not** rerun `ruptures.Window`, does not scan `detector_width`, and does not reinterpret detector-width sensitivity as physical scale. It rebuilds the H object state matrix using the same H profile/state construction as V10.7_a, applies Gaussian smoothing along the day axis at several `sigma` values, computes multifeature temporal-derivative energy, and extracts scale-space ridge families.

## Boundary

- Not a breakpoint detection rerun.
- Not detector-width sensitivity.
- Not yearwise validation.
- Not cartopy spatial-field validation.
- Not causal or quasi-causal inference.
- Output should be used only to choose whether later tests should target H35, H19-H35 as a package, H45 absence, or H57 as a post-W045 reference.
