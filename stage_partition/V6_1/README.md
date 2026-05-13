# Stage Partition V6_1

V6_1 is a child version built on top of V6 point-layer outputs.

Purpose:
- derive a lightweight window layer from V6 candidate points
- attach bootstrap return-day uncertainty to each derived window main peak

V6_1 reads V6 outputs as inputs and may reuse stable V6 utility interfaces, but keeps its own:
- entry script
- config
- pipeline
- outputs
- logs

V6_1 does **not** introduce:
- competition
- parameter-path
- final judgement
- yearwise gating for windows
- a second window system

Uncertainty in V6_1 is only a property of the window main peak.
