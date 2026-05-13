# UPDATE_LOG_V7_H

## w45_H_coherent_region_progress_v7_h

新增 W45/H transition-coherent region progress 诊断分支。

定位：
- 只针对 W45 / anchor=45 / H(z500)。
- 不覆盖 V7-e / V7-e1 / V7-e2 / V7-f / V7-g 结果。
- 不重新解释为因果，不接 downstream pathway / lead-lag。

核心变化：
- 不再使用 single-lat feature 或机械 sliding 3-lat band 作为区域单元。
- 先审计 H 的 pre/post transition vector：dH(lat)=H_post(lat)-H_pre(lat)。
- 使用 bootstrap dH 的 90% CI 对每个纬向 feature 标注 stable_positive_change / stable_negative_change / ambiguous_change。
- 将连续、同一 dH sign class 的纬向 features 构造成 transition-coherent regions。
- 对 coherent region 重新计算 progress onset / midpoint / finish / duration 及 bootstrap 分布。
- 输出 whole-field / single-feature / 3-lat-band / coherent-region 稳定性对照和上游影响诊断。

主要输出：
- w45_H_transition_vector_audit_v7_h.csv
- w45_H_coherent_regions_v7_h.csv
- w45_H_coherent_region_progress_v7_h.csv
- w45_H_unit_comparison_v7_h.csv
- w45_H_unit_stability_summary_v7_h.csv
- w45_H_coherent_region_upstream_implication_v7_h.csv
- w45_H_coherent_region_progress_summary_v7_h.md

运行入口：
```bat
python D:\easm_project01\stage_partition\V7\scripts\run_w45_H_coherent_region_progress_v7_h.py
```

## hotfix_01 - ProgressTimingConfig threshold-name compatibility

- Fixed `AttributeError: 'ProgressTimingConfig' object has no attribute 'onset_threshold'` in `w45_H_coherent_region_progress.py`.
- Added config compatibility helpers so V7-h accepts the existing V7-e names `threshold_onset`, `threshold_midpoint`, and `threshold_finish`, while remaining compatible with earlier draft names if present.
- No algorithm, input, output, window, or statistical interpretation semantics were changed.
