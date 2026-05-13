# UPDATE_LOG_V9_1_D_EOF_TRANSITION_MODE_AUDIT

## Version
`v9_1_d_eof_transition_mode_audit`

## Purpose
新增 V9.1_d EOF/MEOF 转换模态审计支线，用于检验 V9 peak/order 不稳定是否与窗口内多对象年际转换模态有关。

## Boundary
- 只读 V9，不修改 V9 源码或 V9 输出。
- 不使用单年 peak day 作为 EOF 输入。
- EOF 输入来自固定窗口内的多对象、整段、年际异常 profile，并做对象 block 等权。
- EOF/PC 位相只是统计转换模态候选，不是物理年份型。
- PC 位相分组后，会重新调用 V9 peak 逻辑做组内 peak/order 审计。
- 不加入 state、growth、process_a。

## New files
- `V9_1/scripts/run_eof_transition_mode_audit_v9_1_d.py`
- `V9_1/src/stage_partition_v9_1/eof_transition_mode_audit_v9_1_d.py`

## Output
`V9_1/outputs/eof_transition_mode_audit_v9_1_d/`

Key outputs:
- `meof_input_feature_matrix_Wxxx.csv`
- `object_block_variance_contribution_Wxxx.csv`
- `eof_mode_summary_Wxxx.csv`
- `eof_mode_stability_Wxxx.csv`
- `pc_scores_Wxxx.csv`
- `pc_year_leverage_Wxxx.csv`
- `pc_vs_v9_instability_targets_Wxxx.csv`
- `pc_vs_pairwise_order_influence_Wxxx.csv`
- `pc_phase_group_pairwise_order_Wxxx.csv`
- `eof_transition_mode_peak_order_evidence_Wxxx.csv`
- `eof_phase_composite_profiles_Wxxx.csv`

## Interpretation
V9.1_d 只判断 EOF/PC 年际模态是否能够解释 V9 peak/order 不稳定。即使某个 PC 位相下 order 变强，也不能直接命名为物理型；必须后续再检查 composite 场型/剖面与物理背景。
