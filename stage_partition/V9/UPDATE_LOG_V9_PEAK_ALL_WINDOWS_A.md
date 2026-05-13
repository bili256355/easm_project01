# UPDATE_LOG V9 peak_all_windows_v9_a

## Purpose

Create a clean V9 peak-only baseline by extracting the audited V8/V7 peak layer and applying it to the strict accepted/significant windows:

- W045
- W081
- W113
- W160

W135 is explicitly not included because it is not in the strict accepted 95% mainline window set.

## Scope

Included:

- object peak registry per window;
- paired-year bootstrap peak-day distribution;
- timing resolution and tau_sync estimate;
- pairwise peak-order test;
- pairwise peak-synchrony/equivalence test;
- pairwise window-overlap helper;
- peak valid-day / boundary-NaN audit;
- cross-window peak summaries;
- W045 regression audit against V8 peak_only_v8_a when available.

Excluded:

- state curves and state relation diagnostics;
- growth curves;
- catch-up / rollback / multi-stage / process_a diagnostics;
- coordinate-meaning audit;
- final physical claims.

## Interpretation boundary

V9 outputs are event-time peak diagnostics only. They may support statements about peak timing, peak-order, and peak synchrony inside accepted windows. They must not be used alone to infer state-front, growth-front, catch-up, rollback, physical causality, or process-stage claims.
