# 当前工程修改清单：lead_lag_screen_v2_pcmciplus_direction_d

## 修改层

```text
lead_lag_screen/V2
```

## 入口脚本

保持不变：

```text
lead_lag_screen/V2/scripts/run_pcmciplus_lpcmci_v2.py
```

## 新输出目录

```text
lead_lag_screen/V2/outputs/pcmci_plus_v2_d
```

## 修改文件

```text
lead_lag_screen/V2/src/lead_lag_screen_v2/settings.py
lead_lag_screen/V2/src/lead_lag_screen_v2/graph_extract.py
lead_lag_screen/V2/src/lead_lag_screen_v2/pipeline.py
lead_lag_screen/V2/README_PCMCI_PLUS_LPCMCI_V2.md
```

## 关键修复

```text
Tigramite graph[source,target,lag] is exported as source(t-lag)->target(t)
```

## 不改变内容

```text
PCMCI+ only
LPCMCI disabled
20 variables in conditioning pool
9 windows
tau 0..5
same-family output exclusion
```
