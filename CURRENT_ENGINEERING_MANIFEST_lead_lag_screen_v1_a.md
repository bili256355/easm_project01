\
# 当前工程新增清单：lead_lag_screen_v1_a

## 新增层

```text
lead_lag_screen/V1
```

## 角色

```text
pathway 重建前的第一层时间资格筛查；
不继承旧 pathway V1/V2；
不替代 stage_partition；
读取 foundation/V1 的 anomaly index。
```

## 单一入口

```text
lead_lag_screen/V1/scripts/run_lead_lag_screen_v1.py
```

## 单一默认输出目录

```text
lead_lag_screen/V1/outputs/lead_lag_screen_v1_a
```

## 核心实现文件

```text
lead_lag_screen/V1/src/lead_lag_screen_v1/settings.py
lead_lag_screen/V1/src/lead_lag_screen_v1/core.py
lead_lag_screen/V1/src/lead_lag_screen_v1/stats_utils.py
lead_lag_screen/V1/src/lead_lag_screen_v1/data_io.py
lead_lag_screen/V1/src/lead_lag_screen_v1/pipeline.py
lead_lag_screen/V1/src/lead_lag_screen_v1/logging_utils.py
```

## 输出表

```text
lead_lag_curve_long.csv
lead_lag_null_summary.csv
lead_lag_directional_robustness.csv
lead_lag_pair_summary.csv
lead_lag_temporal_pools.csv
summary.json
run_meta.json
```

## 关键边界

```text
lead_lag_yes != pathway_established
lag=0 != lead evidence
strength != fixed-r hard gate
direction != fixed-ratio hard gate
ambiguous must be preserved
```
