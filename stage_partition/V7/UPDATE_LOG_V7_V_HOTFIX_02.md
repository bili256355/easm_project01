# UPDATE_LOG_V7_V_HOTFIX_02

## Patch
`v7_v_hotfix_02_unified_expanded_search`

## Purpose
Fix the C0 event-search design exposed by the first V7-v result review.

C0 previously used:

```text
pre    = day0–39
post   = day49–74
search = day40–48
```

This made the accepted W45 core window also serve as the complete event-search window. The result produced many `left_censored`, `right_censored`, and `boundary_peak` events, making it impossible to distinguish true event invalidity from an overly narrow search window.

## Change
Only C0 search is changed:

```text
C0_full_stage search: day40–48 -> day35–53
```

C0 pre/post remain unchanged:

```text
pre  = day0–39
post = day49–74
```

C1 and C2 are unchanged:

```text
C1 search = day35–53
C2 search = day35–53
```

## Scope
This hotfix does not change:

- baseline pre/post definitions
- S_dist definition
- S_pattern definition
- growth-speed definition `V = dS/dt`
- event validity logic
- bootstrap logic
- order/synchrony decision rules
- H/Jw-only scope
- no P/V/Je rule
- no spatial/latband pairing rule

## Interpretation
This hotfix isolates whether the poor C0 validity in V7-v came from using the accepted transition core as the full event-search window.

If C0 still remains invalid/unresolved after this patch, the issue is less likely to be only search-window truncation and more likely reflects baseline/metric/event-definition sensitivity.
