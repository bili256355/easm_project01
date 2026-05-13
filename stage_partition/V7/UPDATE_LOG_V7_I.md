# UPDATE_LOG_V7_I

## w45_H_latbin_profile_progress_v7_i

Purpose: test the implementation-level correction raised after W45-H V7-f/g/h diagnostics.

The current V6/V7 profile builder constructs 2-degree latitude features by:

1. averaging over the configured longitude range;
2. interpolating the resulting latitude profile to the 2-degree destination latitude grid.

That procedure is not a true 2-degree latitude-bin mean. V7-i keeps the W45-H progress method unchanged, but replaces only the H profile construction for this diagnostic with:

1. averaging over the configured H longitude range;
2. averaging raw latitude points within each 2-degree latitude bin.

This patch does not modify V7-e / V7-e1 / V7-e2 / V7-f / V7-g / V7-h outputs. It is a diagnostic branch for deciding whether the 2-degree feature construction itself is a source of instability.

Key outputs:

- `w45_H_latbin_feature_provenance_v7_i.csv`
- `w45_H_profile_construction_comparison_v7_i.csv`
- `w45_H_latbin_progress_observed_v7_i.csv`
- `w45_H_latbin_progress_bootstrap_samples_v7_i.csv`
- `w45_H_latbin_progress_bootstrap_summary_v7_i.csv`
- `w45_H_interp_vs_latbin_progress_comparison_v7_i.csv`
- `w45_H_latbin_upstream_implication_v7_i.csv`

Interpretation boundary:

- If lat-bin mean improves stability, the previous W45-H feature instability may partly come from the interpolation-based 2-degree construction.
- If lat-bin mean does not improve stability, the instability should not be attributed primarily to missing latitude-bin denoising.
- This run does not prove field order, synchrony, or causality.
