# V9.1_b transition-type mixture audit summary

version: `v9_1_b_transition_type_mixture_audit`

## Method boundary
- This audit is read-only relative to V9 and does not modify V9 outputs.
- It does not cluster single-year peak days.
- It clusters whole-window multi-object behaviour features, then reruns V9 peak logic inside type groups.
- It does not assign physical meanings to clusters.

## Configuration
- windows: W045, W081, W113, W160
- k_values: [2, 3, 4, 5, 6]
- type_bootstrap_n: 500
- min_cluster_size_for_type_peak: 8

## Evidence counts
- unstable_clustering: 200

## Interpretation rules
- strong/moderate mixture candidates are statistical transition-type hypotheses, not physical regimes.
- no V9 peak result is replaced by V9.1_b.
- If type-level peak/order does not stabilize, the mixture hypothesis is not supported in the tested feature space.