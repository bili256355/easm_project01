# V10 peak_subpeak_reproduce_v10_a hotfix02 main-tree patch

## What this patch does

This patch installs the confirmed V10 peak/subpeak reproduction corrections into the normal V10 project tree.

It replaces the temporary audit-bundle execution path with the normal main entry:

```bat
python D:\easm_project01\stage_partition\V10\scripts\run_peak_subpeak_reproduce_v10_a.py
```

## What was fixed

- The selector now matches V9 by using candidate/system-window `overlap_fraction`, not raw overlap days.
- Raw detector score output now includes `window_id`.
- Raw score regression now reads V9 per-window raw score files and infers `window_id` for a valid cross-window comparison.
- Regression diff outputs are cleaned on each run to avoid stale old difference files.

## Output

The run writes to:

```text
D:\easm_project01\stage_partition\V10\outputs\peak_subpeak_reproduce_v10_a
D:\easm_project01\stage_partition\V10\logs\peak_subpeak_reproduce_v10_a
```

The temporary folder below is not required by this main-tree patch and may be deleted after this patch is installed:

```text
D:\easm_project01\stage_partition\V10\peak_subpeak_reproduce_v10_a1_audit_bundle
```

## First file to inspect

```text
D:\easm_project01\stage_partition\V10\outputs\peak_subpeak_reproduce_v10_a\audit\v10_vs_v9_subpeak_regression_audit.csv
```

Expected successful state:

```text
main_window_selection: pass
object_profile_window_registry: pass
raw_profile_detector_scores: pass
bootstrap_selected_peak_days: pass
```

If all four pass, V10 main-tree `peak_subpeak_reproduce_v10_a` has reproduced the V9 peak/subpeak extraction outputs.
