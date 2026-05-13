# UPDATE_LOG_V7_Z_JE_AUDIT_B

## v7_z_b — Je raw-supported early signal validation

Purpose:
- Add a Je-only follow-up audit after `V7-z-je-audit-a` found that the Je shape-pattern day33 peak is normalization-sensitive.
- Test whether Je day26–33 is still supported by raw-profile evidence, or whether it is only a shape-normalization-sensitive artifact / weak precursor to the day46 raw-profile main peak.

Scope:
- Adds a new standalone entry script and module.
- Does not modify the V7-z main multi-object workflow.
- Does not modify V7-z evidence gates or final claims.

Main outputs:
- Raw Je profile and raw feature time series.
- Raw feature peak audit.
- Day26/day33/day46 raw before-after bootstrap.
- Early-only / late-excluded raw detector sensitivity.
- Raw cumulative-change / peak-vs-ramp audit.
- A final Je day26–33 status decision.
