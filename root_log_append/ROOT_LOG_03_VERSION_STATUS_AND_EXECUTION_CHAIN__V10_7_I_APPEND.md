# V10.7_i append: W33→W45 cross-object transition mapping audit

## Version status

V10.7_i adds a new single entry point:

```bash
python D:\easm_project01\stage_partition\V10\v10.7\scripts\run_w45_transition_mapping_v10_7_i.py
```

Output directory:

```text
D:\easm_project01\stage_partition\V10\v10.7\outputs\w45_transition_mapping_v10_7_i
```

## Why this version exists

V10.7_h tested same-object E2–M configuration similarity. That is insufficient for W45 because W33→W45 may involve object role reorganization rather than same-object continuation.

V10.7_i tests E2-to-M cross-object transition mapping. It allows H_E2 to map to non-H M targets such as M_Jw, M_P, or M_V.

## Execution chain

Input:

```text
foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz
```

Objects:

- P from precip, 15–35N, 110–140E
- V from v850, 15–35N, 110–140E
- H from z500, 15–35N, 110–140E
- Je from u200, 25–45N, 120–150E, q90 jet strength
- Jw from u200, 25–45N, 80–110E, q90 jet strength

Windows:

- E1 = day12–23
- E2 = day27–38
- M = day40–48

Core outputs:

- E2 object → M object pairwise transition matrix
- E2→M multivariate mapping skill
- remove-one-source-object contribution
- H_E2 target-specific mapping
- route decision

## Status boundary

V10.7_i is a route-decision diagnostic, not final mechanism evidence. It should be used to decide whether W33-to-W45 scalar cross-object mapping exists and whether H_E2 contributes to that mapping.
