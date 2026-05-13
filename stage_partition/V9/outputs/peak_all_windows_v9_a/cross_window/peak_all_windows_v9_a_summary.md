# V9 peak-all-windows summary

version: `v9_peak_all_windows_a`
output_tag: `peak_all_windows_v9_a`
windows_processed: 4
run_mode: list; targets: W045,W081,W113,W160

## Method boundary
- V9 is peak-only: it extracts object transition-event timing across accepted windows.
- It reuses the audited V7/V8 peak-layer semantics and does not compute state/growth/process outputs.
- It should be interpreted as an event-time skeleton, not a state-process explanation.

## Windows processed
- W045: anchor day 45; system day40-48; detector day0-69
- W081: anchor day 81; system day75-87; detector day49-102
- W113: anchor day 113; system day108-118; detector day88-149
- W160: anchor day 160; system day155-165; detector day119-182

## Mainline exclusion
- W135 is not included in V9 because it is not part of the strict accepted 95% window set.

## Peak relation counts
- peak_order_decision=B_peak_earlier_tendency: 22
- peak_order_decision=A_peak_earlier_tendency: 15
- peak_order_decision=peak_order_unresolved: 3
- synchrony_decision=synchrony_indeterminate: 37
- synchrony_decision=synchrony_tendency: 3

## Forbidden interpretations
- Do not infer state-front, growth-front, catch-up, rollback, multi-stage, or pre-post process claims from V9 alone.
- Peak order is event timing only; it is not causality.