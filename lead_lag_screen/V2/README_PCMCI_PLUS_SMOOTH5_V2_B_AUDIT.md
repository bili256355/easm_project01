# lead_lag_screen/V2_b: PCMCI+ smooth5 audit layer

## Role

This patch adds a **diagnostic audit layer** for the already-run V2_a output:

```text
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_a
```

It does not rerun PCMCI+. It does not replace V1. It does not create pathways, mediators, or physical-mechanism claims.

The purpose is to explain why the V2_a headline supported-edge layer is much narrower than the V1 smooth5 lead-lag eligibility pool.

## Entry

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_lead_lag_screen_v2_b_audit.py
```

## Output

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_smooth5_v2_b_audit
```

## Inputs read

```text
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_a/pcmci_plus_edges_long.csv
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_a/pcmci_plus_tau0_contemporaneous_diagnostic.csv
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_a/pcmci_plus_v1_overlap.csv
lead_lag_screen/V1/outputs/lead_lag_screen_v1_smooth5_a/lead_lag_evidence_tier_summary.csv
```

If the V1 evidence table is not found, the audit falls back to the V1 columns already merged in `pcmci_plus_v1_overlap.csv`.

## Main outputs

```text
pcmci_plus_lagged_layer_audit_long.csv
pcmci_plus_window_threshold_audit.csv
pcmci_plus_family_threshold_audit.csv
pcmci_plus_family_threshold_audit_all_windows.csv
pcmci_plus_pair_level_tau0_lagged_v1_audit.csv
v1_lead_lag_yes_fate_under_pcmci_plus.csv
v1_tier_to_pcmci_fate_audit.csv
v1_family_to_pcmci_fate_audit.csv
tau0_vs_lagged_joint_summary.csv
manual_review_priority_examples.csv
v1_to_pcmci_fate_summary.csv
pcmci_lagged_layer_count_summary.csv
INTERPRETATION_GUARDRAILS.md
SAME_FAMILY_CONDITIONING_SENSITIVITY_NOTE.md
summary.json
run_meta.json
settings_summary.json
```

## Diagnostic layers

The lagged PCMCI+ tests are divided into these layers:

```text
L4_supported_graph_and_window_fdr
L3_graph_selected_raw_p_lost_by_fdr
L2_graph_selected_not_raw_p05
L1_raw_p05_not_graph_selected
L0_not_selected_not_raw_p05
```

This makes it possible to see whether V2_a is narrow because:

```text
1. PCMCI+ did not select graph edges;
2. graph-selected edges had weak raw p-values;
3. graph-selected/raw-p edges were removed by window-level BH-FDR;
4. tau=0 contemporaneous coupling dominates instead of lagged tau=1..5 support.
```

## Important boundary

Same-family conditioning sensitivity is **not tested** in this fast audit, because that requires rerunning PCMCI+ with a changed conditioning design. The audit writes a note explaining this and prepares the basis for a possible later `v2_c` sensitivity run.
