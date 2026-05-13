# V8 state relation summary

version: `v8_state_relation_a_hotfix01`
window: `W045`

## Method boundary
- State only: S_dist / S_pattern and pairwise ΔS_AB(t).
- No growth, rollback, multi-stage growth, process_a labels, or final causal/process claims.
- No fixed |ΔS| strength thresholds are used for near/dominant classification.
- Strength is compared against same-object same-size bootstrap reproducibility nulls.
- Segment duration and matching parameters are construction parameters and must be sensitivity-audited.
- Smoothed-field boundary NaNs are audited explicitly; finite pairwise domains are used for segmentation.

## Segment construction parameters
- null_n: 1000
- min_atomic_segment_len: 3
- min_near_block_len: 3
- max_short_gap_for_merge: 1
- min_overlap_ratio: 0.5
- support_supported: 0.95
- support_tendency: 0.9
- support_exploratory: 0.8
- random_seed_offset: 9137
- write_null_daily: False

## Segment bootstrap support counts
- unresolved: 308
- exploratory_signal: 51
- tendency: 25
- supported: 5

## Observed block counts
- uncertain_block: 85
- B_dominant_block: 64
- A_dominant_block: 43
- near_block: 3

## Interpretation boundary
- A-dominant/B-dominant/near/uncertain are state-relation segment classes, not peak timing claims.
- near means within same-object state reconstruction resolution, not perfect physical synchrony.