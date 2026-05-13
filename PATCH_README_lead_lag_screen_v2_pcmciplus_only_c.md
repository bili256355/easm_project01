# lead_lag_screen_v2_pcmciplus_only_c 修复补丁

## 修复内容

1. 放弃 LPCMCI 执行链；
2. 只保留 PCMCI+；
3. 修复 Tigramite graph 方向导出问题；
4. 禁止同族 source-target 边进入输出和 V1-V2 对照；
5. 保留 20 个指数作为 PCMCI+ 条件变量池；
6. 使用新输出目录，避免覆盖旧的 V2a/V2b 结果。

## 关键修复

旧风险：

```text
graph[source, target, lag] 被误解释为 source(t-lag) -> target(t)
```

新规则：

```text
graph[target, source, lag] 导出为 source(t-lag) -> target(t)
```

## 同族边策略

```text
所有 20 个变量仍进入 PCMCI+ 条件控制；
但 source_family == target_family 的边不进入输出候选和 V1-V2 对照；
V1 中同族 pair 会标记为 same_family_excluded_by_design。
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_pcmciplus_lpcmci_v2.py
```

## 输出目录

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_v2_c
```
