# lead_lag_screen_v2_pcmciplus_direction_d 修复补丁

## 修复内容

本补丁修复 V2-c 的 graph 方向错误。

V2-c 错误规则：

```text
graph[target, source, lag] -> source(t-lag) -> target(t)
```

V2-d 正确规则：

```text
graph[source, target, lag] -> source(t-lag) -> target(t)
```

依据是 Tigramite 官方说明：

```text
graph[i,j,tau] = '-->' for tau > 0 denotes a directed lagged link from i to j at lag tau.
```

## 保持不变

```text
LPCMCI disabled
PCMCI+ only
20 variables remain in conditioning pool
9 windows
tau 0..5
ParCorr
analysis_mode multiple
same-family source-target edges excluded from exported results
```

## 新输出目录

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_v2_d
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_pcmciplus_lpcmci_v2.py
```
