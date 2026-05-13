# UPDATE_LOG_V7_Z_PREPOST_PROCESS_A

## Purpose

Add an isolated W45 profile-only pre/post process-diagnostic line: `v7_z_prepost_process_a`.

This is not a detector hotfix. It does not change the V7-z detector, accepted-window scopes, C0/C1/C2 baseline definitions, or the S_dist / S_pattern / G_dist / G_pattern curve definitions.

## Main change

The previous winner-style middle layer is not used for main interpretation. The new line diagnoses curve structures from paired-year bootstrap samples of:

- `Delta S_AB(t) = S_A(t) - S_B(t)` for state progress;
- `Delta G_AB(t) = G_A(t) - G_B(t)` for growth process.

## New entry

```bash
python V7/scripts/run_prepost_process_curve_v7_z_process_a.py
```

## New output root

```text
V7/outputs/prepost_process_curve_v7_z_process_a/
V7/logs/prepost_process_curve_v7_z_process_a/
```

## Main outputs

```text
per_window/W045/object_curve_validity_W045.csv
per_window/W045/object_growth_episode_bootstrap_W045.csv
per_window/W045/pairwise_state_curve_bootstrap_W045.csv
per_window/W045/pairwise_growth_curve_bootstrap_W045.csv
per_window/W045/pairwise_branch_baseline_consensus_W045.csv
per_window/W045/pairwise_process_interpretation_W045.csv
per_window/W045/legacy_output_status_W045.csv
per_window/W045/prepost_process_curve_summary_W045.md
```

## Support semantics

- `supported`: bootstrap structure support >= 0.95
- `tendency`: 0.90 <= support < 0.95
- `exploratory_signal`: 0.80 <= support < 0.90, audit only
- `unresolved`: support < 0.80

## Deprecated-for-mainline outputs

The hotfix06 pairwise summary / winner-style outputs are not deleted, but are marked legacy audit only by `legacy_output_status_W045.csv`.

## Scope guard

This patch intentionally runs W45 profile-only. Do not promote to all windows until the process-a W45 output has been audited.
