# stage_partition/V2

## Scope
This version implements **window finding only** for stage partitioning.

It does **not** implement:
- type layer
- lead/lag or contribution diagnostics inside windows
- pathway analysis
- anomaly-based state expression

## Hard constraints
- Input must come from `foundation/V1` raw smoothed fields only.
- `field_profiles` are the only mainline input representation.
- State expression must not subtract daily climatology.
- Two mature detectors run in parallel: `MovingWindow` and `ruptures.Window`.
- Detector-native output is primary. Only when interval support is missing do we reconstruct windows from the detector's own support curve.
- Edge windows come from the same backend that produced them.
- V2 does not inherit code from `stage_partition/V1`.

## Standardization
- Build `X[day, feature]` from raw smoothed field profiles.
- Apply per-feature z-score along the day axis.
- Apply block-equal-contribution scaling after feature-wise z-score.

## Support audits included in V2
- parameter-path stability
- year bootstrap stability
- backend agreement
- optional permutation significance support
