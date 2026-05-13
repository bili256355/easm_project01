# UPDATE_LOG — V9.1_c bootstrap_year_influence_audit

## Purpose

Add a read-only V9.1_c audit branch to reverse-audit V9 peak bootstrap instability:
which years, when sampled or oversampled in paired-year bootstrap replicates,
systematically push object peak days earlier/later or push pairwise order toward
A-earlier / B-earlier outcomes.

## Boundary

- Does **not** modify V9 source files.
- Does **not** overwrite V9 outputs.
- Does **not** use single-year peak as evidence.
- Does **not** perform year clustering or physical-type naming.
- Does **not** add state, growth, process_a, or climatological-phase layers.

## Added files

- `V9_1/scripts/run_bootstrap_year_influence_audit_v9_1_c.py`
- `V9_1/src/stage_partition_v9_1/bootstrap_year_influence_audit_v9_1_c.py`

## Output root

`V9_1/outputs/bootstrap_year_influence_audit_v9_1_c/`

## Main outputs

Per window:

- `bootstrap_sample_year_counts_Wxxx.csv`
- `bootstrap_object_peak_samples_Wxxx.csv`
- `bootstrap_pairwise_order_samples_Wxxx.csv`
- `v9_replay_bootstrap_regression_audit_Wxxx.csv`
- `object_peak_year_influence_Wxxx.csv`
- `pairwise_order_year_influence_Wxxx.csv`
- `influential_year_sets_Wxxx.csv`
- `bootstrap_year_influence_summary_Wxxx.csv`

Cross-window:

- `object_peak_year_influence_all_windows.csv`
- `pairwise_order_year_influence_all_windows.csv`
- `influential_year_sets_all_windows.csv`
- `bootstrap_year_influence_summary_all_windows.csv`
- `v9_replay_bootstrap_regression_audit_all_windows.csv`
- `bootstrap_sample_year_counts_all_windows.csv`

## Method note

If V9 did not save bootstrap year composition, V9.1_c replays the V9 paired-year
bootstrap using the same V7/V9 peak detector and random-index generator, while
recording year counts.  A replay regression audit compares replayed bootstrap
peak/order summaries against existing V9 outputs.

## Interpretation

Influential years are statistical candidates only.  They are not physical year
types and should be followed by field/background anomaly checks before any
physical interpretation.
