# V9.1_d EOF transition-mode audit hotfix01

## Scope

This hotfix only repairs a runtime argument/unpacking bug in `eof_transition_mode_audit_v9_1_d.py`.

## Fix

- Removed an invalid loop in `_pc_vs_influence()` that attempted to unpack `(obj, mode)` from `pc.groupby(["mode"])`.
- The PC score table is grouped only by `mode`; object-level grouping is handled by the V9.1_c influence table inside the nested loop.

## Non-changes

- Does not change V9.
- Does not change V9.1_b/c.
- Does not change the EOF/MEOF method, PC grouping, type-level peak logic, or output structure.
