# T3 V→P physical hypothesis audit patch

## Purpose

This patch adds a V1-side physical-hypothesis audit for the T3 V→P contraction.

It does **not** rerun the V1 lead-lag screen and does **not** claim pathway causality. It reads existing V1 / V1 stability / index_validity outputs plus smooth5 field/index products and produces diagnostic evidence for five hypotheses:

- H1: rain-band spatial reorganization
- H2: V component shift from strength toward NS-difference / position
- H3: T3 internal state mixing / dilution
- H4: P target component shift
- H5: synchronous multi-family reorganization

## Run

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_t3_v_to_p_physical_hypothesis_audit.py
```

Without figures:

```bat
python lead_lag_screen\V1\scripts\run_t3_v_to_p_physical_hypothesis_audit.py --no-figures
```

Without cartopy:

```bat
python lead_lag_screen\V1\scripts\run_t3_v_to_p_physical_hypothesis_audit.py --no-cartopy
```

## Default output

```text
lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_physical_hypothesis_audit
```

## Main outputs

```text
tables/t3_subwindow_v_to_p_dilution_classification.csv
tables/t3_subwindow_dilution_summary.csv
tables/window_subwindow_regional_precip_contribution.csv
tables/t3_early_late_regional_precip_shift.csv
tables/t3_v_index_to_regional_p_response.csv
tables/t3_v_index_subwindow_response_shift.csv
tables/t3_p_target_group_v_to_p_summary.csv
tables/t3_p_target_component_shift_evidence.csv
tables/s3_t3_s4_multi_family_stability_shift.csv
tables/t3_synchronous_reorganization_evidence.csv
tables/t3_physical_hypothesis_evidence_summary.csv
summary/T3_PHYSICAL_HYPOTHESIS_AUDIT_README.md
summary/summary.json
summary/run_meta.json
```

## Interpretation

This audit is designed to distinguish statistical/diagnostic support levels for hypotheses. It should not be treated as a direct mechanism proof. The physical interpretation still requires reviewing the spatial maps and connecting the results back to the broader EASM seasonal-transition problem.
