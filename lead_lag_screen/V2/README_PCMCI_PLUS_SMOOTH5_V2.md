# lead_lag_screen/V2: PCMCI+ smooth5 direct-edge control

## Role

This layer is a PCMCI+ control group for the existing `lead_lag_screen/V1` 5-day lead-lag temporal eligibility screen.

It is **not** a pathway reconstruction layer, **not** a mediator layer, and **not** a physical-mechanism interpretation layer.

## Fixed decisions

```text
Input: foundation/V1/outputs/baseline_smooth5_a/indices/index_anomalies.csv
Variables: same 20 variables as V1
Windows: same 9 target-side windows as V1
Method: PCMCI+
Conditional independence test: ParCorr
Main lagged output: tau = 1..5
Contemporaneous tau=0: separate diagnostic output only
Reported edges: cross-family only
Conditioning pool: all 20 variables, including same-family variables
FDR: Benjamini-Hochberg within each window
Fallback: forbidden; tigramite import/API failure stops the run
```

## Entry

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_lead_lag_screen_v2.py
```

## Main output directory

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_smooth5_v2_a
```

## Main outputs

```text
pcmci_plus_edges_long.csv
pcmci_plus_tau0_contemporaneous_diagnostic.csv
pcmci_plus_candidate_pairs.csv
pcmci_plus_window_panel_meta.csv
pcmci_plus_window_summary.csv
pcmci_plus_window_family_rollup.csv
pcmci_plus_v1_overlap.csv
pcmci_plus_v1_overlap_summary.csv
runtime_task_status.csv
runtime_task_timing.csv
runtime_failed_tasks.csv  # only if a window fails
run_meta.json
settings_summary.json
summary.json
_task_cache/
```

## Interpretation boundary

`pcmci_plus_supported=True` means the edge was retained by PCMCI+ graph extraction and passed the window-level BH q-value rule. It should be interpreted as a **conditional direct-edge control result** for V1 lead-lag, not as a pathway or mechanism claim.

`tau=0` links are written separately as contemporaneous diagnostics and should not be merged into the main lagged direct-edge table.
