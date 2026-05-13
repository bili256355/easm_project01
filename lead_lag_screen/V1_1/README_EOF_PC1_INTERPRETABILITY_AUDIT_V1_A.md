# V1_1 EOF-PC1 interpretability audit v1_a

## Purpose

This is a diagnostic audit inside `lead_lag_screen/V1_1`. It asks why an EOF-PC1 lead-lag route may fail to show the sharp T3 V→P weakening seen by the V1/V1_1 index-pair route.

The audit does **not** replace the V1_1 structural V→P screen. It tests whether EOF-PC1 actually carries the T3 high-latitude / boundary / retreat structures under dispute.

## Main question

> Does EOF-PC1 fail to show T3 weakening because T3 is not weak, or because PC1 does not represent the T3 structures that V1_1 found?

## What it does

1. Computes top EOF modes for P and V850 over the East Asian domain.
2. Summarizes EOF loadings over main / south / high-latitude regions.
3. Correlates PC1–PC5 with V1_1 structural indices.
4. Tests whether PC1-only and PC1–3 reconstructions reproduce S3→T3 and T3→S4 regional changes.
5. Runs a lightweight PC-level lead-lag comparison.
6. Outputs a diagnosis table stating whether EOF-PC1 can or cannot be used to refute V1_1 T3 weakening.

## What it does not do

- It does not rerun the full V1_1 structural V→P screen.
- It does not add new P/V indices.
- It does not perform pathway or causal inference.
- It does not use EOF-PC1 as the final arbiter.
- It does not edit V1 or V1_1 main outputs.

## Run

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_eof_pc1_interpretability_audit_v1_a.py
```

No figures:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_eof_pc1_interpretability_audit_v1_a.py --no-figures
```

If Cartopy is unavailable:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_eof_pc1_interpretability_audit_v1_a.py --no-cartopy
```

If memory is tight, explicitly request coarser spatial stride:

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_eof_pc1_interpretability_audit_v1_a.py --spatial-stride 2
```

The default is `--spatial-stride 1`.

## Output directory

```text
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_eof_pc1_interpretability_audit_v1_a
```

## Key tables

```text
tables/p_eof_loading_region_summary.csv
tables/v_eof_loading_region_summary.csv
tables/eof_pc_structural_index_correlation.csv
tables/eof_reconstruction_region_change_skill.csv
tables/eof_pc_lead_lag_by_window.csv
tables/eof_vs_structural_index_lead_lag_comparison.csv
tables/eof_pc1_interpretability_diagnosis.csv
```

## Key figures

```text
figures/P_EOF_loading_modes_1_3.png
figures/V_EOF_loading_modes_1_3.png
figures/P_observed_vs_PC_reconstruction_T3_minus_S3.png
figures/V_observed_vs_PC_reconstruction_T3_minus_S3.png
figures/P_observed_vs_PC_reconstruction_S4_minus_T3.png
figures/V_observed_vs_PC_reconstruction_S4_minus_T3.png
```

## Interpretation guardrail

If PC1 does not reconstruct T3 high-latitude / boundary / retreat structures, then EOF-PC1 lead-lag continuity cannot be used to refute V1_1's T3 pair-level weakening.
