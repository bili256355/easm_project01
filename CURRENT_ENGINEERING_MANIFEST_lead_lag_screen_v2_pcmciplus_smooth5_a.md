# 当前工程补丁清单：lead_lag_screen_v2_pcmciplus_smooth5_a

## 修改性质

新增独立层：

```text
lead_lag_screen/V2
```

该补丁不修改 `lead_lag_screen/V1`，不修改 `foundation/V1`，不继承旧 `pathway` 工程。

## 单一入口

```text
lead_lag_screen/V2/scripts/run_lead_lag_screen_v2.py
```

## 单一输出目录

```text
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_a
```

## 新增文件

```text
lead_lag_screen/V2/README_PCMCI_PLUS_SMOOTH5_V2.md
lead_lag_screen/V2/scripts/run_lead_lag_screen_v2.py
lead_lag_screen/V2/src/lead_lag_screen_v2/__init__.py
lead_lag_screen/V2/src/lead_lag_screen_v2/settings.py
lead_lag_screen/V2/src/lead_lag_screen_v2/data_io.py
lead_lag_screen/V2/src/lead_lag_screen_v2/stats_utils.py
lead_lag_screen/V2/src/lead_lag_screen_v2/logging_utils.py
lead_lag_screen/V2/src/lead_lag_screen_v2/tigramite_adapter.py
lead_lag_screen/V2/src/lead_lag_screen_v2/graph_extract.py
lead_lag_screen/V2/src/lead_lag_screen_v2/compare_v1.py
lead_lag_screen/V2/src/lead_lag_screen_v2/rollups.py
```

## 已落实裁决

```text
只考虑 5 日指数；
PCMCI+ 作为 V1 lead-lag 的条件直接边对照；
ParCorr；
主输出 tau=1..5；
tau=0 单独诊断；
输出/报告层不允许同族 source-target 边；
同族变量允许进入 conditioning pool；
每个窗口内 BH-FDR；
只输出 direct edge，不生成 pathway/mediator/chain；
tigramite 缺失或 multiple-dataset API 不支持时硬报错，不做 fallback。
```

## 工程边界

```text
V2 不替代 V1；
V2 不产生 pathway established；
V2 不解释物理机制；
V2 不读取旧 pathway 工程；
V2 不跑 9 日版本。
```
