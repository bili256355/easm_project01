# UPDATE_LOG_V7_V

## V7-v: W45 H/Jw baseline-sensitive state-growth framework

Purpose: rebuild the H/Jw state-growth method after V7-u exposed event-definition problems.

Key changes:

1. Only H and Jw are included.
2. Three baseline configurations are tested:
   - C0_full_stage: pre day0-39, post day49-74, search day40-48.
   - C1_buffered_stage: pre day0-34, post day54-69, search day35-53.
   - C2_immediate_pre: pre day25-34, post day54-69, search day35-53.
3. State progress has two parallel branches:
   - distance branch: S_dist = D_pre / (D_pre + D_post).
   - pattern branch: S_pattern = normalized R_diff where R_diff = corr(x, post) - corr(x, pre).
4. Growth is defined only as dS/dt, the first difference of each state-progress branch.
5. P_proj is reference-only and does not enter main decisions.
6. Every event passes a validity gate before pairwise H/Jw order or synchrony adjudication.
7. Invalid events are not treated as unresolved or synchronous.
8. No spatial pairing, no lat-band pairing, and no P/V/Je extension is performed in this branch.

Run:

```bash
cd D:\easm_project01
python stage_partition\V7\scripts\run_w45_H_Jw_baseline_sensitive_state_growth_v7_v.py
```

Debug run:

```bash
cd D:\easm_project01
set V7V_DEBUG_N_BOOTSTRAP=20
set V7V_SKIP_FIGURES=1
python stage_partition\V7\scripts\run_w45_H_Jw_baseline_sensitive_state_growth_v7_v.py
```
