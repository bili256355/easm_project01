# ROOT LOG APPEND — V10.7_g

## Version

V10.7_g — W45 multisource method-control audit.

## Status

Implemented as a new V10.7 entry, not a hotfix to V10.7_f.

## Single entry

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_multisource_method_control_v10_7_g.py
```

## Output directory

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_multisource_method_control_v10_7_g
```

## Role

This is a method-control / cross-object incremental-audit layer. It tests whether the W45 cross-object audit method can detect signals from multiple source objects, and whether the H-package negative result is meaningful.

## Key implementation note

Je and Jw are derived from the shared `u200` field using sector definitions:

- Je = 120–150E, 25–45N, q90 jet strength
- Jw = 80–110E, 25–45N, q90 jet strength

They are not expected to exist as separate smoothed fields.
