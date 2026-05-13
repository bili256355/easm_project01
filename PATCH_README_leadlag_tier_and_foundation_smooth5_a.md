# lead-lag evidence-tier + foundation smooth5 联合补丁

## 目的

本补丁落实两个独立但相关的任务：

1. 在 `lead_lag_screen/V1` 当前结果基础上新增 evidence-tier 分层审计输出；
2. 在 `foundation/V1` 增加 5 日滑动版本的独立运行入口，输出 5 日滑动基础层与异常指数。

---

## A. lead_lag_screen/V1 evidence-tier 分层

### 替换 / 新增文件

```text
lead_lag_screen/V1/src/lead_lag_screen_v1/core.py
lead_lag_screen/V1/src/lead_lag_screen_v1/settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stats_utils.py
lead_lag_screen/V1/src/lead_lag_screen_v1/evidence_tier.py
```

### 新增输出

```text
lead_lag_evidence_tier_summary.csv
lead_lag_window_family_tier_rollup.csv
lead_lag_phi_risk_audit.csv
lead_lag_method_warning_flags.csv
```

### 语义边界

Evidence-tier 是 post-processing / 分层审计：

```text
不改变 lead_lag_label；
不改变主 null；
不改变方向稳健判定；
只是把 main yes / audit null / same-day / high-persistence risk 合并成可用等级。
```

### 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

---

## B. foundation/V1 smooth5

### 新增文件

```text
foundation/V1/scripts/run_foundation_v1_smooth5_a.py
```

### 作用

运行 5 日滑动版本的 foundation/V1，输出到独立 runtime tag：

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a
D:\easm_project01\foundation\V1\logs\baseline_smooth5_a\mainline
```

不会覆盖原来的 9 日滑动 baseline：

```text
D:\easm_project01\foundation\V1\outputs\baseline_a
```

### 运行

```bat
cd /d D:\easm_project01
python foundation\V1\scripts\run_foundation_v1_smooth5_a.py
```

### 5 日异常指数输出

运行完成后查看：

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv
```

---

## 推荐运行顺序

1. 先运行 5 日 foundation：

```bat
cd /d D:\easm_project01
python foundation\V1\scripts\run_foundation_v1_smooth5_a.py
```

2. 重新运行 lead-lag 当前 9 日结果以生成 evidence-tier：

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

如后续要用 5 日异常指数跑 lead-lag，需要把 `lead_lag_screen/V1/settings.py` 中：

```text
input_index_anomalies
```

切换到：

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv
```

建议后续另开 `lead_lag_screen_v1_smooth5_a` 输出目录，以免覆盖 9 日 lead-lag 结果。
