# V10.7_e H35 existence attribution audit patch

## Purpose

This patch adds **V10.7_e**, a route-decision audit for why the H35 feature exists in the H-only W045 analysis.

It does **not** test H35 → W045, H35 → P/V/Je/Jw, or any causal mechanism. It answers a narrower attribution question:

> If H35 is not a stable independent event, why does it still appear as an H-only candidate?

The audit tests whether H35 is better attributed to:

- H18-like second-stage structure;
- seasonal/background or local curvature;
- a few-year-driven climatological feature;
- a method-level score shoulder/background candidate;
- or an unresolved possible independent component.

E2 multi-object attribution is explicitly **not tested** in V10.7_e.

## Added entrypoint

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_h35_existence_attribution_v10_7_e.py
```

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\h35_existence_attribution_v10_7_e
```

## Key output tables

```text
tables\h35_existence_attribution_decision_v10_7_e.csv
tables\h35_ablation_score_audit_v10_7_e.csv
tables\h35_score_feature_contribution_v10_7_e.csv
tables\h35_h18_like_projection_residual_v10_7_e.csv
tables\h35_attribution_pseudo_null_v10_7_e.csv
tables\h35_year_contribution_v10_7_e.csv
```

The most important table is:

```text
tables\h35_existence_attribution_decision_v10_7_e.csv
```

## Key figures

```text
figures\h35_ablation_score_audit_v10_7_e.png
figures\h35_existence_attribution_decomposition_object_domain_v10_7_e.png
```

## Interpretation boundary

Do not use this output to claim:

- H35 causes W045;
- H35 conditions P/V/Je/Jw;
- H35 is a confirmed weak precursor;
- E2 multi-object adjustment has been tested.

Use this output only to decide whether H35 is likely a H18-like second stage, background/local curvature, few-year-driven feature, method shoulder, or unresolved.
