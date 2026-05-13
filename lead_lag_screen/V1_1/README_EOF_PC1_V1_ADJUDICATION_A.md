# V1_1 EOF-PC1 V1 Adjudication Audit A

This patch adds an independent V1_1 audit layer:

```text
lead_lag_screen_v1_1_eof_pc1_v1_adjudication_a
```

It does **not** use the high-latitude V1_1 branch as the adjudication criterion.  It asks a narrower question:

> Is EOF-PC1 actually eligible to adjudicate the V1 old-index T3 V→P weakening?

## Run

```bat
python D:\easm_project01\lead_lag_screen\V1_1\scripts\run_v1_1_eof_pc1_v1_adjudication_a.py
```

## Required prior outputs

This audit expects the previous EOF-PC1 interpretability audit to have produced:

```text
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_eof_pc1_interpretability_audit_v1_a\tables\p_eof_pc_scores.csv
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_eof_pc1_interpretability_audit_v1_a\tables\v_eof_pc_scores.csv
```

It also reads V1_1 structural VP outputs:

```text
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_structural_vp_a\indices\v1_1_index_values_doy_anomaly.csv
D:\easm_project01\lead_lag_screen\V1_1\outputs\lead_lag_screen_v1_1_structural_vp_a\tables\v1_1_v_to_p_classified_pairs.csv
```

## Main outputs

```text
tables/pc1_old_index_alignment_by_window.csv
tables/old_index_pc1_leadlag_by_window.csv
tables/pc1_seasonal_progression_control.csv
tables/eof_pc1_v1_style_classification.csv
tables/eof_pc1_v1_adjudication_diagnosis.csv
summary/summary.json
summary/run_meta.json
```

## Interpretation rules

EOF-PC1 can only be used as a serious counterpoint to V1 T3 weakening if all three prerequisites pass:

1. V_PC1 and P_PC1 align with the V1 old-index spaces.
2. EOF-PC1 relation survives day-of-year / window-centering controls.
3. EOF-PC1 is `stable_lag_dominant` under the V1-style lag-vs-tau0 diagnostic.

If any prerequisite fails, EOF-PC1 should not be used to refute V1 T3 weakening.
