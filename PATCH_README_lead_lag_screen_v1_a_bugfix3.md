# lead_lag_screen_v1_a_bugfix3

## 修复 / 增强内容

本补丁新增 AR(1) persistence / phi clipping 审计，不改变主判定规则。

### 背景

上一轮结果显示大量 `phi = 0.95`，但原输出只给出了裁剪后的 phi，无法判断：

```text
raw_phi 只是略高于 0.95；
还是大量接近 0.98 / 0.99 / 1.0 甚至超过 1。
```

这会影响对 pooled AR(1) surrogate null 是否偏乐观的判断。

### 新增内容

`lead_lag_surrogate_ar1_params.csv` 新增字段：

```text
raw_phi_before_clip
phi_after_clip
phi_clip_limit
phi_clipped_flag
phi_clip_amount
phi_clip_direction
phi_clip_severity
n_pair_for_phi
```

`lead_lag_surrogate_yearwise_scale.csv` 新增字段：

```text
pooled_raw_phi_before_clip
pooled_phi_used
pooled_phi_clip_amount
pooled_phi_clip_severity
```

`summary.json` 新增：

```text
n_phi_clipped_total
phi_clip_severity_counts
raw_phi_max
raw_phi_min
```

### clip severity 定义

```text
none      : 未裁剪
minor     : clip_amount < 0.01
moderate  : 0.01 <= clip_amount < 0.03
severe    : clip_amount >= 0.03
```

## 是否改变方法语义

不改变。

本补丁只增强审计输出：

```text
主 null 仍是 pooled_window_variable_ar1；
审计 null 仍是 pooled_phi_yearwise_scale_ar1；
lead_lag_label 的判定规则不变。
```

## 替换文件

覆盖：

```text
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\core.py
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\settings.py
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\stats_utils.py
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

## 注意

如果不想覆盖旧输出目录，请先备份：

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_a
```

或修改 `settings.py` 中的 `output_tag`。
