# T3 V→P disappearance audit

This audit does not rerun V1. It reads existing V1 smooth5 outputs and the V1 stability judgement layer.

Focus question:
Why do V→P candidates shrink in T3 relative to adjacent windows?

Key interpretation:
- `passed_stable_lag` means the pair survives as stable lag-dominant.
- `passed_tau0_coupled` means the pair survives, but lag0 is close/competitive.
- `positive_not_surrogate_significant` means positive-lag signal did not clear the AR(1) surrogate background.
- `audit_not_stable` means main support weakens under the audit null.
- `reverse_competitive` / `direction_uncertain` mean the V→P direction is not stable.
- `tau0_competitive` means the relation is better interpreted as synchronous/rapid adjustment than stable lag.

This is an audit of attrition mechanisms, not a new pathway result.
