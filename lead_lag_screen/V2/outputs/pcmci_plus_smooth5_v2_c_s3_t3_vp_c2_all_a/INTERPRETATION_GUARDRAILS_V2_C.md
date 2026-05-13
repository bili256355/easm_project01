# V2_c PCMCI+ sensitivity guardrails

This output is a sensitivity rerun layer, not a final scientific result.

## What V2_c tests

1. Whether V2_a lost physically expected V->P / H->P signals because same-source-family and same-target-family derived indices were allowed to enter the conditioning pool.
2. Whether the 20-index full system is too collinear for a stable PCMCI+ graph-selection result.

## What V2_c does not prove

- It does not establish causal pathways.
- It does not replace V1 lead-lag temporal eligibility.
- It does not make mediator or chain claims.
- It does not turn PCMCI+ edges into physical mechanism conclusions.

## Variant meanings

- `c1_targeted_no_same_family_controls`: targeted diagnostic for (('V', 'P'),). For each concrete source->target variable pair, the PCMCI+ variable set is source + target + variables from other families. This exactly excludes other variables from the source family and target family for that concrete test, but it is not a full-system graph.
- `c2_family_representative_standard`: standard PCMCI+ on a smaller representative variable pool. This tests whether strong within-family redundancy in the 20-index pool caused graph collapse.

## How to judge V->P

If V->P remains absent even in C1 and C2, then PCMCI+ is likely not a useful control for this index/window design without a deeper method redesign. If V->P recovers in C1 or C2, then V2_a should be treated as an over-conditioned lower bound, not a scientific contradiction of V1 or physical expectation.
