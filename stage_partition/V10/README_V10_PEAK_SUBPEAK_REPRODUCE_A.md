# V10-a independent V9 subpeak/peak reproduction

## Purpose

This patch creates an independent V10 reproduction of the V9 subpeak/peak extraction layer.

It is intentionally **not** a continuation of the discarded v1/v1_1 sensitivity-cause audits. It does not inherit their filters, judgments, admissibility logic, or interpretations.

## Hard boundary

V10-a:

- does **not** import `stage_partition_v9`;
- does **not** import `stage_partition_v7`;
- does **not** call any V9/V7 helper/interface;
- reimplements the required V9 extraction semantics inside `stage_partition_v10`;
- reads existing V9 CSV outputs only for a final read-only regression audit.

## What it reproduces

- hardcoded accepted windows used by V9: W045, W081, W113, W160;
- object profile construction for P/V/H/Je/Jw;
- V7-z/V9 detector input: year-mean profile and feature-wise day-standardization;
- `ruptures.Window(width=20, model='l2', min_size=2, jump=1)` detector score;
- local candidate subpeak extraction;
- selected support band construction;
- paired-year bootstrap support and selected peak distributions;
- selected main peak candidate per object/window;
- pairwise peak-order and synchrony outputs for parity with V9 peak layer;
- read-only V10 vs V9 regression audit.

## What it does not do

- no sensitivity audit;
- no physical subpeak typing;
- no H/Jw admissibility filtering;
- no accepted-window redefinition;
- no state/growth/process outputs;
- no scientific interpretation.

## Run

Copy this `V10` directory into:

```bat
D:\easm_project01\stage_partition\V10
```

Run:

```bat
python D:\easm_project01\stage_partition\V10\scripts\run_peak_subpeak_reproduce_v10_a.py
```

Optional debug run:

```bat
set V10_PEAK_DEBUG_N_BOOTSTRAP=20
python D:\easm_project01\stage_partition\V10\scripts\run_peak_subpeak_reproduce_v10_a.py
```

Optional custom smoothed fields path:

```bat
set V10_PEAK_SMOOTHED_FIELDS=D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz
python D:\easm_project01\stage_partition\V10\scripts\run_peak_subpeak_reproduce_v10_a.py
```

## Outputs

```text
V10/outputs/peak_subpeak_reproduce_v10_a/
```

Main cross-window outputs:

```text
cross_window/object_profile_window_registry_all_windows.csv
cross_window/cross_window_subpeak_candidate_registry.csv
cross_window/main_window_selection_all_windows.csv
cross_window/cross_window_object_peak_registry.csv
cross_window/raw_profile_detector_scores_all_windows.csv
cross_window/bootstrap_selected_peak_days_all_windows.csv
cross_window/v10_vs_v9_subpeak_regression_audit.csv
cross_window/V10_PEAK_SUBPEAK_REPRODUCE_A_SUMMARY.md
```

## How to judge success

First check:

```text
cross_window/v10_vs_v9_subpeak_regression_audit.csv
```

If regression passes for `main_window_selection`, `object_profile_window_registry`, `raw_profile_detector_scores`, and `bootstrap_selected_peak_days`, then V10 has reproduced the V9 extraction layer at the CSV-output level.

If differences appear, do **not** interpret V10 scientifically. Inspect the regression audit first.
