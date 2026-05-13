# 当前工程新增 / 修改清单

## 1. lead_lag_screen/V1

### 修改文件

```text
lead_lag_screen/V1/src/lead_lag_screen_v1/core.py
lead_lag_screen/V1/src/lead_lag_screen_v1/settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stats_utils.py
```

### 新增文件

```text
lead_lag_screen/V1/src/lead_lag_screen_v1/evidence_tier.py
```

### 新增输出

```text
lead_lag_evidence_tier_summary.csv
lead_lag_window_family_tier_rollup.csv
lead_lag_phi_risk_audit.csv
lead_lag_method_warning_flags.csv
```

## 2. foundation/V1

### 新增入口

```text
foundation/V1/scripts/run_foundation_v1_smooth5_a.py
```

### 新增输出目录

运行后生成：

```text
foundation/V1/outputs/baseline_smooth5_a
foundation/V1/logs/baseline_smooth5_a/mainline
```

## 3. 不改变的内容

```text
不覆盖 foundation/V1/outputs/baseline_a
不改变 stage_partition
不恢复旧 pathway
不改变 lead_lag_screen 主判定语义
```
