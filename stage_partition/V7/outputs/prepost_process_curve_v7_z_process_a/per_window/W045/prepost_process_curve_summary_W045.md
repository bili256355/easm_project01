# V7-z prepost process-a summary

## Method boundary
- This run is W45 profile-only.
- Detector, W45 scope, C0/C1/C2 baselines, and S/G definitions are not modified.
- Main diagnostics use paired-year bootstrap curve-structure classifications of Delta S_AB(t) and Delta G_AB(t).
- early/core/late winner outputs are deprecated for main interpretation.
- Supported threshold is 0.95; 0.90-0.95 is tendency only; 0.80-0.90 is audit-only exploratory signal.

bootstrap_n: 1000

## Interpretation readiness counts
- audit_only_or_unresolved: 10

## Process interpretation rows
- H-Je: process_structure_supported_or_audit_only | risk=branch_split;C2_sensitive
- H-Jw: H_distance_state_growth_front_with_Jw_catchup | risk=branch_split;C2_sensitive
- Je-Jw: process_structure_supported_or_audit_only | risk=branch_split;C2_sensitive
- P-H: process_structure_supported_or_audit_only | risk=C2_sensitive
- P-Je: process_structure_supported_or_audit_only | risk=C2_sensitive
- P-Jw: process_structure_supported_or_audit_only | risk=C2_sensitive
- P-V: process_structure_supported_or_audit_only | risk=branch_split;C2_sensitive
- V-H: process_structure_supported_or_audit_only | risk=branch_split;C2_sensitive
- V-Je: process_structure_supported_or_audit_only | risk=C2_sensitive
- V-Jw: process_structure_supported_or_audit_only | risk=branch_split;C2_sensitive