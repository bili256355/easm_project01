# V8 peak-only baseline summary

version: `v8_peak_only_a`
output_tag: `peak_only_v8_a`
windows_processed: 1
run_mode: w45; targets: W045,45

## Method boundary
- This V8 line is a peak-only extraction from the original V7-z multiwindow peak layer.
- It does not compute or write state curves, growth curves, pre-post process summaries, 2D mirror outputs, or final claims.
- Peak-order and peak-synchrony outputs keep the original V7 helper semantics.

## Windows processed
- W045: system day40-48; detector day0-69

## Peak relation counts
- peak_order_decision=A_peak_earlier_tendency: 5
- peak_order_decision=B_peak_earlier_tendency: 5
- synchrony_decision=synchrony_indeterminate: 8
- synchrony_decision=synchrony_tendency: 2

## Forbidden interpretations
- Do not infer state-front, growth-front, catch-up, rollback, multi-stage, or pre-post process claims from this V8 output alone.