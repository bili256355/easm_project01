# 当前工程补丁清单：lead_lag_screen_v2_b_audit

## 修改性质

新增独立审计层，不修改 V1，不修改 V2_a 主结果，不重新运行 PCMCI+。

```text
lead_lag_screen/V2_b audit for pcmci_plus_smooth5_v2_a
```

## 单一入口

```text
lead_lag_screen/V2/scripts/run_lead_lag_screen_v2_b_audit.py
```

## 单一输出目录

```text
lead_lag_screen/V2/outputs/pcmci_plus_smooth5_v2_b_audit
```

## 新增文件

```text
lead_lag_screen/V2/README_PCMCI_PLUS_SMOOTH5_V2_B_AUDIT.md
lead_lag_screen/V2/scripts/run_lead_lag_screen_v2_b_audit.py
lead_lag_screen/V2/src/lead_lag_screen_v2_b_audit/__init__.py
lead_lag_screen/V2/src/lead_lag_screen_v2_b_audit/settings.py
lead_lag_screen/V2/src/lead_lag_screen_v2_b_audit/pipeline.py
```

## 审计目标

```text
1. 拆解 PCMCI+ lagged direct-edge 的 graph-selected / raw-p / window-FDR 层；
2. 追踪 V1 Tier1/Tier2/Tier3 候选在 PCMCI+ 各层的去向；
3. 对照 tau=0 contemporaneous diagnostic 与 tau=1..5 lagged edge；
4. 输出窗口、family、pair 级别的收缩来源；
5. 明确解释边界：V2_a 是严格 direct-edge lower bound，不是 pathway 结果。
```

## 不做的事

```text
不重跑 PCMCI+；
不测试同族 conditioning 敏感性；
不改 V2_a 主判定；
不生成 pathway / mediator / chain；
不把 V1-only 关系写成假关系；
不把 PCMCI+ supported 写成物理机制成立。
```

## 运行命令

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_lead_lag_screen_v2_b_audit.py
```
