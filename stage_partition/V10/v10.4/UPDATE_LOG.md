# UPDATE_LOG — V10.4 object_order_sensitivity_v10_4

## v10.4 initial patch

Purpose: add candidate-lineage-aware object-to-object order sensitivity audit.

Scope:

- reads V10.1 / V10.2 / V10.3 outputs
- assigns P/V/H/Je/Jw candidates to each joint lineage and each V10.3 sensitivity config
- computes pairwise object order at tau=0, tau=2, tau=5
- summarizes pairwise stability
- emits whole-sequence object order per lineage/config
- emits reversal inventory

Non-goals:

- no peak discovery rerun
- no bootstrap rerun
- no physical interpretation
- no accepted-window redefinition
