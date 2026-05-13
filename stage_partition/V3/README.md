# stage_partition/V3

Current clean mainline for stage partitioning.

## Scope of V3 phase-1
- use `foundation/V1` preprocess outputs as upstream dependency
- use raw smoothed field profiles for `P / V / H / Je / Jw`
- use `ruptures.Window` as the only active detector
- build unique main windows by a window-native construction layer
- produce original window-level evidence mapping, support audit, and retention audit

## Explicit non-goals in V3 phase-1
- no anomaly-based state expression
- no MovingWindow mainline
- no window-internal timing / contribution / type / pathway layers
- no expanded physical interpretation layer

## Main output table
`stage_partition_main_windows.csv` is the only formal window-object entry for downstream use.


Current default run label after point-layer B-track patch: mainline_v3_i.

## Point-layer B-track patch (mainline_v3_i)
- keeps the existing point/bootstrap/yearwise/parameter-path generation chain
- adds point-layer B-track support audit outputs on top of the existing point evidence products
- does not promote window-layer support-profile outputs into the current patch scope
- keeps `stage_partition_main_windows.csv` as the only formal downstream window-object entry


## Current point-layer B-track refinement (mainline_v3_j)

- Refines point-layer B-track outputs on top of `mainline_v3_i`.
- Hard-separates `formal_primary` headline points from `neighbor_competition` audit points.
- Downgrades parameter-path support to an auxiliary caution/support column rather than a hard headline gate.
- Makes the current backend mode explicit: point-layer B-track still reuses existing point stability outputs in this refinement.
