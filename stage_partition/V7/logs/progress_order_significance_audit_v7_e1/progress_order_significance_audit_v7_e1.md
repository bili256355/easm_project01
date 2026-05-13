# progress_order_significance_audit_v7_e1 audit log

## Scope

This audit reads V7-e progress-timing outputs and tests pairwise progress-midpoint differences.
It does not recompute progress curves, field states, bootstrap samples, or LOYO samples.

## Source

- Source V7-e output directory: `D:\easm_project01\stage_partition\V7\outputs\field_transition_progress_timing_v7_e`
- Output directory: `D:\easm_project01\stage_partition\V7\outputs\progress_order_significance_audit_v7_e1`

## Test object

For each window and field pair A/B:

```text
Delta = midpoint_B - midpoint_A
```

- Delta > 0 means A's progress midpoint is earlier than B's.
- Delta < 0 means B's progress midpoint is earlier than A's.

No minimum effective day threshold is used.

## Statistical components

1. Bootstrap Delta distribution and 95% percentile interval.
2. Exact sign-direction test under a sign-flip null: non-zero Delta signs are treated as exchangeable to test directional asymmetry.
3. BH-FDR over pairwise sign-flip p-values.
4. LOYO direction audit as stability evidence, not as a primary significance test.

## Parameters

```json
{
  "source_v7e_output_tag": "field_transition_progress_timing_v7_e",
  "output_tag": "progress_order_significance_audit_v7_e1",
  "random_seed": 20260430,
  "fdr_alpha": 0.05
}
```

## Evidence label counts

```json
{
  "supported_directional_tendency": 35,
  "not_distinguishable": 4,
  "confirmed_directional_order": 1
}
```

## Interpretation limits

- This is timing/progress order, not causality.
- `not_distinguishable` does not prove synchrony; equivalence/synchrony would require a justified equivalence margin, which is intentionally not introduced here.
- `supported_directional_tendency` means direction is significant by sign-flip, but the bootstrap 95% CI still crosses zero.
