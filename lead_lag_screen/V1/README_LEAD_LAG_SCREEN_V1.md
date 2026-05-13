\
# lead_lag_screen/V1

## 定位

这是 pathway 重建前的第一层时间资格筛查，不是路径成立证明层。

它只回答：

```text
X -> Y 是否具备基本时间顺序支持？
```

不回答：

```text
是否因果成立；
是否是直接边；
是否是完整中介路径；
是否排除共同驱动；
是否已经形成物理机制解释。
```

## 工程入口

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

## 默认输入

```text
D:\easm_project01\foundation\V1\outputs\baseline_a\indices\index_anomalies.csv
```

## 默认输出

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_a
```

## 方法约定

1. 不读取旧 pathway V1/V2 结果。
2. 使用既有 9 个窗口：S1/T1/S2/T2/S3/T3/S4/T4/S5。
3. 窗口归属按 target-side：Y(t) 必须属于窗口 W。
4. 所有 lag pair 必须 same-year，禁止跨年拼接。
5. lag > 0 表示 X leads Y。
6. lag = 0 只作为 same-day coupling 诊断，不作为超前证据。
7. 统计支持使用 AR(1) surrogate max-stat。
8. 强度不是固定 r 阈值 hard gate，而是 null-relative effect-size 表征。
9. 方向稳健使用 year-block bootstrap 检查正滞后证据是否稳定强于反向和同期。
10. ambiguous 结果保留，不伪装成 yes，也不删除。

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

## 解释边界

`lead_lag_yes` 只能解释为第一层时间资格通过，不能写成 pathway established。
`lead_lag_yes_with_same_day_coupling` 可以进入 strict temporal pool，但必须保留同期耦合风险。
`lead_lag_ambiguous_*` 不进入 strict temporal pool，但进入 expanded temporal risk pool，用于反馈、同期耦合、双向关系和 transition 风险审计。
