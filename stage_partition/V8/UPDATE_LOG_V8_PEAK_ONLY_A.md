# UPDATE_LOG_V8_PEAK_ONLY_A

## Purpose

Create a clean V8 peak-only baseline extracted from the audited V7-z multiwindow peak layer.

This is not a new scientific method and not a detector rewrite. The purpose is to isolate the peak layer from the mixed V7 state/growth/pre-post/process branches before any future state redesign.

## Scope included

- Accepted/significant window loading using the original V7 helper.
- V7-z raw/profile object-window detector.
- Paired-year bootstrap selected peak days.
- Timing-resolution audit and tau_sync estimate.
- Pairwise peak-order test.
- Pairwise peak synchrony/equivalence test.
- Pairwise selected-window overlap audit.
- Regression audit against existing V7 hotfix06 peak outputs when available.

## Explicitly excluded

- State curves: S_dist / S_pattern.
- Growth curves: G_dist / G_pattern.
- Pairwise state/growth differences.
- W45 pre-post process summaries.
- process_a ΔS/ΔG curve-structure diagnostics.
- Evidence gates and final-claim registries.
- 2D mirror and Je physical variance audit outputs.

## Implementation rule

V8-a reuses the original V7 peak-layer helper functions directly. It may reorganize output files into a clean V8 directory, but it must not alter detector semantics, accepted-window semantics, paired-year bootstrap semantics, peak-order semantics, or peak synchrony semantics.

## Run command

```bat
python D:\easm_project01\stage_partition\V8\scripts\run_peak_only_v8_a.py
```

Debug run:

```bat
set V8_PEAK_DEBUG_N_BOOTSTRAP=50
python D:\easm_project01\stage_partition\V8\scripts\run_peak_only_v8_a.py
```

Formal default run:

```bat
set V8_PEAK_N_BOOTSTRAP=1000
python D:\easm_project01\stage_partition\V8\scripts\run_peak_only_v8_a.py
```

## Output directory

```text
D:\easm_project01\stage_partition\V8\outputs\peak_only_v8_a\
```

## Validation expectation

For W045, V8 peak-only outputs should match the V7 hotfix06 peak-layer outputs. Check:

```text
V8\outputs\peak_only_v8_a\cross_window\v8_vs_v7_hotfix06_peak_regression_audit.csv
```

Any difference should be treated as an implementation-regression item, not as a scientific finding.
