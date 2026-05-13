# UPDATE_LOG_V9_PEAK_SELECTION_SENSITIVITY_A

## Version

`V9_peak_selection_sensitivity_a`

## Purpose

This patch adds a sensitivity-test layer for the V9 peak-only baseline. It does **not** re-run changepoint detection and does **not** replace V9. It fixes the four V7/V9 accepted main windows:

- W045
- W081
- W113
- W160

and tests whether the selected object peak day, selected peak band, pairwise peak order, and five-object peak sequence are stable when the peak-detection and peak-selection settings are perturbed.

## Fixed elements

- Main windows are fixed from the V9/V7 accepted-window scope registry.
- Object definitions remain the same as V9:
  - P: precip, 105–125E, 15–39N
  - V: v850, 105–125E, 10–30N
  - H: z500, 110–140E, 15–35N
  - Je: u200, 120–150E, 25–45N
  - Jw: u200, 80–110E, 25–45N
- No physical interpretation is included.

## Perturbation axes

The sensitivity grid includes:

1. Input smoothing scale
   - smooth9: `foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz`
   - smooth5: `foundation/V1/outputs/baseline_smooth5_a/preprocess/smoothed_fields.npz`

2. Detector scale
   - narrow: `detector_width=16`, `band_half_width=8`
   - baseline: `detector_width=20`, `band_half_width=10`
   - wide: `detector_width=24`, `band_half_width=12`

3. Search range
   - narrow_search: anchor ±15 days, clipped to the analysis range
   - baseline_search: V9 detector search range
   - wide_search: V9 analysis range

4. Candidate selection rule
   - baseline_rule
   - max_score
   - closest_anchor
   - max_overlap

Total default configurations: `2 × 3 × 3 × 4 = 72`.

## Outputs

The main output directory is:

`D:\easm_project01\stage_partition\V9\outputs\peak_selection_sensitivity_v9_a\`

Core outputs:

- `cross_window/object_peak_selection_by_config.csv`
- `cross_window/object_peak_sensitivity_summary.csv`
- `cross_window/pairwise_order_sensitivity_summary.csv`
- `cross_window/window_sequence_by_config.csv`
- `cross_window/window_sequence_sensitivity_summary.csv`
- `cross_window/smooth9_vs_smooth5_peak_comparison.csv`
- `cross_window/baseline_reproduction_audit.csv`
- `cross_window/object_profile_build_audit.csv`
- `cross_window/sensitivity_config_grid.csv`
- `cross_window/V9_PEAK_SELECTION_SENSITIVITY_A_SUMMARY.md`
- `cross_window/run_meta.json`

Per-window slices are also written under:

`outputs/peak_selection_sensitivity_v9_a/per_window/<window_id>/`

## Interpretation boundary

This patch produces sensitivity diagnostics only. A stable selected peak day can be used as a V9 reference peak. If the peak day is sensitive but object ordering is stable, only coarse order should be used. If selected peak bands are sensitive, they should not be interpreted as physical sub-windows. If object ordering is sensitive, the V9 peak sequence for that window/object should be downgraded.

## Implementation note

For exact V9 baseline reproduction, the baseline configuration reads V9's original candidate and selection tables. For non-baseline sensitivity configurations, the module builds the same object domains from the selected smoothed field input and computes a local pre/post profile-contrast score. This is recorded in `run_meta.json` as `uses_local_prepost_contrast_score_for_nonbaseline_configs=true`. The baseline reproduction audit must be checked before interpreting sensitivity outputs.
