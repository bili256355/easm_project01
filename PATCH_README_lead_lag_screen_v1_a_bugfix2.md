# lead_lag_screen_v1_a_bugfix2

## 修复 / 增强内容

本补丁针对 surrogate 口径做显式化与审计增强，不改变 lead-lag 主方法语义。

新增内容：

1. 保留主 null：
   `pooled_window_variable_ar1`

   含义：
   在每个 window × variable 上跨年份 pooled 估计一组 AR(1) 参数，
   再为每一年独立生成 synthetic series。

2. 新增审计 null：
   `pooled_phi_yearwise_scale_ar1`

   含义：
   `phi` 仍使用 window × variable pooled phi，
   但 synthetic series 的均值/尺度使用各 year × variable 自身的 mean/std。
   这用于检查主 null 是否对 yearwise amplitude 差异敏感。

3. 新增 AR(1) 参数审计输出：

```text
lead_lag_surrogate_ar1_params.csv
lead_lag_surrogate_yearwise_scale.csv
```

4. 新增审计 null 对照输出：

```text
lead_lag_audit_surrogate_null_summary.csv
```

该表包含主 null 与审计 null 的 p 值、95% null 阈值、support flag 是否改变等。

## 重要边界

- 审计 null 不改变主判定结果；
- 主 `lead_lag_label` 仍由 `pooled_window_variable_ar1` 生成；
- 审计 null 用于判断结果对 yearwise scale 差异是否敏感；
- 本补丁同时包含 bugfix1 的 bootstrap 年份抽样维度修复。

## 替换文件

覆盖：

```text
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\core.py
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\settings.py
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

## 输出目录

仍为：

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_a
```

如果不想覆盖旧结果，请先手动备份旧输出目录，或在 `settings.py` 中修改 `output_tag`。
