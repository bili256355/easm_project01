# PATCH MANIFEST: lead_lag_screen/V3_b EOF-PC1 Stability Judgement

## 新增文件

```text
lead_lag_screen/V3/scripts/run_lead_lag_screen_v3_b_stability.py
lead_lag_screen/V3/src/lead_lag_screen_v3b/__init__.py
lead_lag_screen/V3/src/lead_lag_screen_v3b/settings_b.py
lead_lag_screen/V3/src/lead_lag_screen_v3b/stability_core.py
lead_lag_screen/V3/src/lead_lag_screen_v3b/pipeline_stability.py
lead_lag_screen/V3/README_V3_B_STABILITY.md
lead_lag_screen/V3/PATCH_MANIFEST_V3_B_STABILITY.md
```

## 设计说明

- 不覆盖 `eof_pc1_smooth5_v3_a`。
- 新输出目录为 `eof_pc1_smooth5_v3_b_stability_a`。
- 复用 V3_a 的 EOF-PC1 提取、AR(1) surrogate max-stat 和 V1 对照逻辑。
- 新增正式稳定性判定层，避免把“正滞后显著”误写成“稳定超前滞后”。

## 默认 bootstrap

```text
relation stability bootstrap = 1000
PC1 mode stability bootstrap = 500
```

## 解释边界

本补丁用于重新生成 V3 的正式稳定性判定结果。V3_a 结果应保留为探索性结果，不再作为正式 EOF-PC1 lead-lag 结论。
