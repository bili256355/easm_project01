# UPDATE LOG — V9.1_b transition type mixture audit

## Purpose

V9.1_b is added inside the existing `V9_1` branch as an independent audit entry. It tests whether V9 peak/order instability may arise from mixtures of multiple year-level transition behaviour types.

## Key methodological correction from V9.1_a

V9.1_b does **not** use single-year peak days as clustering inputs. Single-year peak detection is too noisy for this project because the original peak detector is designed for multi-year/common-signal estimation. Instead, V9.1_b clusters each year by whole-window multi-object behaviour features, then returns to the V9 multi-year peak detector inside each candidate type group.

## Boundary

- Added under `V9_1`; it does not create or require a new `V9_1_b` root.
- Does not modify V9.
- Does not overwrite V9 outputs.
- Does not modify V9.1_a outputs.
- Does not add state/growth/process_a.
- Does not assign physical meanings to clusters.
- Does not replace V9 peak results.

## Main evidence chain

whole-window multi-object behaviour features → candidate year grouping → type-group multi-year peak → type-group bootstrap → comparison with V9 full sample.
