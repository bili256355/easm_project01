# V10.7_j patch: W45 E2–M yearwise SNR / reliability audit

## Purpose

This patch adds a new V10.7_j entry point for auditing whether the yearwise scalar indicators used by V10.7_i have enough signal-to-noise and window stability to support E2→M mapping tests.

It is designed after V10.7_i returned a non-detection in the current scalar mapping framework. V10.7_j does **not** reinterpret that non-detection as a scientific negative result. Instead, it checks whether the underlying yearwise E2/M object-window indicators are reliable enough for such a mapping test.

## New entry point

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_snr_reliability_v10_7_j.py
```

Optional bootstrap control:

```bash
set V10_7_J_N_BOOT=1000
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_snr_reliability_v10_7_j.py
```

Default bootstrap iterations: `500`.

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_snr_reliability_v10_7_j
```

## Key outputs

```text
tables\w45_snr_input_audit_v10_7_j.csv
tables\w45_object_window_daily_reliability_by_year_v10_7_j.csv
tables\w45_object_window_snr_summary_v10_7_j.csv
tables\w45_yearwise_strength_bootstrap_reliability_v10_7_j.csv
tables\w45_mapping_leave_days_out_sensitivity_v10_7_j.csv
tables\w45_mapping_window_shift_grid_v10_7_j.csv
tables\w45_mapping_window_shift_sensitivity_v10_7_j.csv
tables\w45_snr_route_decision_v10_7_j.csv
figures\w45_object_window_snr_summary_v10_7_j.png
figures\w45_yearwise_rank_reliability_v10_7_j.png
figures\w45_window_shift_sign_stability_v10_7_j.png
summary_w45_snr_reliability_v10_7_j.md
run_meta\run_meta_v10_7_j.json
```

## What it audits

1. Object-window daily reliability for E1/E2/M and P/V/H/Je/Jw.
2. Between-year variance relative to within-window daily noise.
3. Bootstrap stability of yearwise rankings.
4. Leave-one-day-out sensitivity of E2→M pairwise mapping signs.
5. ±2-day E2/M window shift sensitivity.
6. A route decision for whether V10.7_i's scalar non-detection is interpretable or should be downgraded to low-power / non-decisive.

## Method boundary

- This is not a new W45 mechanism test.
- This is not a causal inference method.
- This does not test H's role directly.
- This audits whether the current yearwise scalar indicators are reliable enough to support V10.7_i-style mapping.
- Low SNR means non-detection should be treated as non-decisive, not as evidence that E2 and M are unrelated.

## Apply instructions

Copy the `stage_partition` folder in this patch over the project root. Existing files are not replaced except `stage_partition_v10_7/__init__.py`, which is a minimal package extension.

The root log append files are provided under `root_log_append/` and should be manually appended to the corresponding root logs.
