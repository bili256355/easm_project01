# lead_lag_screen/V2_b audit interpretation guardrails

This audit is a diagnostic layer for `pcmci_plus_smooth5_v2_a`. It does **not** rerun PCMCI+ and does **not** replace V1.

## What this audit can tell you

1. Whether V2_a is narrow because PCMCI+ did not select many graph edges, or because the additional within-window BH-FDR layer removed graph-selected/raw-p signals.
2. Where V1 Tier1/Tier2/Tier3 candidates went under PCMCI+ layers.
3. Whether a relation is more visible as tau=0 contemporaneous coupling than as a tau=1..5 lagged direct edge.
4. Which family directions survive only as a strict direct-edge lower bound.

## What this audit cannot tell you

1. It cannot prove a physical pathway.
2. It cannot prove that V1-only relations are false.
3. It cannot test same-family conditioning sensitivity, because that requires rerunning PCMCI+ with a changed conditioning design.
4. It cannot convert mediator-like or chain-like patterns into established mediation.

## Recommended wording

Use: `under the current strict PCMCI+ direct-edge + window-FDR definition, only a narrow lower-bound set remains`.

Do not use: `PCMCI+ proves only these pathways matter`.
