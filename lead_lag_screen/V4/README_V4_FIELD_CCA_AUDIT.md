# lead_lag_screen/V4 — smooth5 field lagged-CCA audit

## Purpose

This layer is a **field-to-field coupling-mode audit** after V1/V3.
It is designed to test whether an apparent index-level weakening, especially around T3 / meiyu-ending, is caused by index applicability rather than a true collapse of field-to-field coupling.

It does **not** replace V1 lead-lag results, does **not** perform PCMCI, and does **not** establish pathway or causal mechanisms.

## Main design

- Input: `foundation/V1/outputs/baseline_smooth5_a/preprocess/anomaly_fields.npz`
- Field objects: `P, V, H, Je, Jw`
- Object domains: reused from `foundation/V1` / V3.
- Window definition: same 9 windows as V1/V3.
- Dimensionality reduction: window-wise EOF scores for each object.
- Headline CCA dimension: `k=5`; sensitivity dimension: `k=3`.
- Core pair directions only:
  - `V→P`
  - `H→P`
  - `H→V`
  - `Jw→Je`
  - `Je→H`
- Lag design: `X(t-lag)` vs `Y(t)`, `lag=0..5`.
- Diagnostics:
  - in-sample canonical correlation;
  - year-block cross-validated canonical correlation;
  - year-block target permutation p-value for max train canonical r;
  - year-block bootstrap stability at observed best lag.

## Main outputs

- `cca_eof_scores_long.csv`
- `cca_eof_mode_quality.csv`
- `cca_lag_long.csv`
- `cca_pair_summary.csv`
- `cca_pair_summary_main_k5.csv`
- `cca_permutation_summary.csv`
- `cca_bootstrap_stability.csv`
- `v1_v3_v4_cca_comparison.csv`
- `t3_meiyu_end_cca_audit.csv`
- `summary.json`
- `run_meta.json`

## Interpretation boundary

CCA finds a pair of field-coupling modes that maximizes relationship between two fields under the chosen lag design. It is useful for identifying field-to-field coupling that may be missed by fixed indices or by PC1-only EOF audits. It does not by itself prove a causal pathway, mediator chain, or physical mechanism.

If the result is mostly `tau0_dominant`, it should be interpreted as same-day / fast-adjustment coupling, not as a failure of coupling.
