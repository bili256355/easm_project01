# PATCH MANIFEST: index_validity V1_b joint family field coverage enhancement

This additive patch adds family-level mixed coverage outputs to `index_validity/V1_b_window_family_guardrail`.

## New outputs

- `tables/window_family_joint_field_coverage.csv`
- `tables/t3_window_family_joint_field_coverage.csv`

## Purpose

Quantify how much all indices of a given object family jointly cover that family’s own smoothed field in each window.

## Scientific boundary

This is still index-to-own-field representativeness. It does not test lead-lag, pathway, causality, mediation, or mechanism establishment.

## Files changed

- `src/index_validity_v1_b/metrics.py`
- `src/index_validity_v1_b/pipeline.py`
- `README_WINDOW_FAMILY_GUARDRAIL_V1_B.md`
- `PATCH_MANIFEST_WINDOW_FAMILY_GUARDRAIL_V1_B.md`
