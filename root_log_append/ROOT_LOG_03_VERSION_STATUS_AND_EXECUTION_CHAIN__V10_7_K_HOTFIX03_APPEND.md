# ROOT_LOG_03 append: V10.7_k HOTFIX03

V10.7_k HOTFIX03 adds runtime visibility and controls after the previous run stalled for about an hour without visible progress.

Changes:
- Added stage-level progress prints.
- Added pairwise task progress prints.
- Added `--progress-every`.
- Added `--pairwise-scope` with options `all`, `h-source`, `h-related`.
- Added `--pairwise-bootstrap-policy` with options `all`, `candidate`, `none`.

Purpose:
- Avoid silent long-running executions.
- Allow a practical H-focused formal run before attempting the full all-pairs run.

Scientific note:
- `h-source` is a scoped diagnostic for the current H_E2 structural mapping question. It is not a full all-pairs audit.
