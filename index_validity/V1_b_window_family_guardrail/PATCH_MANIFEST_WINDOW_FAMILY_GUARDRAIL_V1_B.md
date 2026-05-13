# PATCH MANIFEST — index_validity/V1_b_window_family_guardrail smoothed-default correction

## Purpose

This patch corrects the data semantics of the V1_b window-family guardrail.

Previous draft defaulted to:

```text
anomaly_fields.npz + index_anomalies.csv
```

That was inappropriate for the main `index_validity` question. This layer asks whether an index still indicates its own field object, so the main audit must use:

```text
smoothed_fields.npz + index_values_smoothed.csv
```

Anomaly mode remains available only as an optional auxiliary audit.

## Files replaced / added

```text
index_validity/V1_b_window_family_guardrail/README_WINDOW_FAMILY_GUARDRAIL_V1_B.md
index_validity/V1_b_window_family_guardrail/PATCH_MANIFEST_WINDOW_FAMILY_GUARDRAIL_V1_B.md
index_validity/V1_b_window_family_guardrail/scripts/run_index_validity_window_family_guardrail_v1_b.py
index_validity/V1_b_window_family_guardrail/src/index_validity_v1_b/settings.py
index_validity/V1_b_window_family_guardrail/src/index_validity_v1_b/data_io.py
index_validity/V1_b_window_family_guardrail/src/index_validity_v1_b/pipeline.py
```

Other source files are included unchanged to keep the patch self-contained.

## Main run command

```bat
cd /d D:\easm_project01
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py
```

## Main output

```text
D:\easm_project01\index_validity\V1_b_window_family_guardrail\outputs\window_family_guardrail_v1_b_smoothed_a
```

## Optional auxiliary anomaly run

```bat
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py --data-mode anomaly
```

## Guardrail

Do not interpret anomaly-mode results as the main index_validity conclusion. The main conclusion about whether indices indicate their own fields must come from the smoothed-mode output.


## Joint coverage enhancement

Adds family-level mixed coverage outputs:

- `tables/window_family_joint_field_coverage.csv`
- `tables/t3_window_family_joint_field_coverage.csv`

These outputs are additive and do not change the original single-index representativeness metrics or `window_family_guardrail.csv` decision logic.
