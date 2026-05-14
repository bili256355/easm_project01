# V10.7_d H35 residual/anomaly independence audit patch

## Purpose

This patch adds a single route-decision experiment:

**Question 1:** Is H35 a stable independent residual after removing the H18-like component and local/background signal?

**Question 2:** Only if H35 passes the residual-independence gate, does H18 have yearwise predictive support for the H35 residual?

This patch is intentionally not a descriptive H18/H35 similarity audit. It is meant to decide whether the H35 single-point line should continue or stop.

## Apply

Copy the `stage_partition` directory into the project root so the new files land under:

```text
D:\easm_project01\stage_partition\V10\v10.7
```

## Run

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h35_residual_independence_v10_7_d.py
```

Optional project-root argument:

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h35_residual_independence_v10_7_d.py D:\easm_project01
```

## Outputs

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h35_residual_independence_v10_7_d
```

Main decision table:

```text
tables\h35_route_decision_v10_7_d.csv
```

Key evidence tables:

```text
tables\h35_independence_decision_v10_7_d.csv
tables\h35_projection_residual_by_year_v10_7_d.csv
tables\h35_pseudo_event_null_v10_7_d.csv
tables\h18_predicts_h35_residual_v10_7_d.csv
```

Core figures:

```text
figures\h35_residual_decomposition_object_domain_v10_7_d.png
figures\h35_residual_decomposition_profile_v10_7_d.png
figures\h35_residual_against_pseudo_null_v10_7_d.png
```

## Interpretation boundary

This experiment does not test H35 -> W045 / P / V / Je / Jw. It does not infer causality. It only decides whether H35 has enough independent residual content to justify treating H35 as a single-point target in later cross-object audits.

If `H35 single-point line = not_independent`, then the H18 -> H35 question is marked not meaningful / not tested by gate.
