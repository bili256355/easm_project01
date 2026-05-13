# lead_lag_screen/V1 stability judgement post-processing

## Purpose

This patch adds a post-processing layer inside `lead_lag_screen/V1`. It does **not** rerun the V1 lead-lag screen. It reads existing V1 result tables and formalizes a stricter interpretation label for:

- whether positive lag is stably stronger than lag0;
- whether positive lag is stably stronger than reverse lag;
- whether a V1 core candidate should be treated as stable lag-dominant, tau0-coupled, tau0-dominant, audit-sensitive, or not supported.

## Run

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1_stability_judgement.py
```

Default input:

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a
```

Default output:

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b
```

## Main outputs

```text
tables/lead_lag_pair_summary_stability_judged.csv
tables/window_stability_rollup.csv
tables/window_family_stability_rollup.csv
tables/evidence_tier_vs_stability_rollup.csv
tables/lag_vs_tau0_stability_rollup.csv
tables/t3_stability_audit.csv
tables/v1_core_candidate_pool_stability_judged.csv
summary/summary.json
summary/run_meta.json
```

## Interpretation boundary

This layer can separate `stable_lag_dominant` from `significant_lagged_but_tau0_coupled` and `stable_tau0_dominant_coupling`. It still does not prove causality, mechanism, or a pathway. It only refines the V1 temporal eligibility interpretation.
