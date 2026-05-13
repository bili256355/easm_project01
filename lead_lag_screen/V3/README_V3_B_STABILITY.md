# lead_lag_screen/V3_b：EOF-PC1 正式稳定性判定层

## 定位

`V3_a` 是 EOF-PC1 探索性审计：它可以说明 PC1 层是否存在耦合与正滞后显著性，但不足以正式判断“lag 是否稳定强于同期”。

`V3_b` 是正式稳定性判定层，新增并进入最终 judgement 的内容包括：

1. PC1 模态稳定性；
2. lag-vs-tau0 稳定性；
3. forward-vs-reverse 稳定性；
4. peak-lag 稳定性；
5. 最终 `stability_judgement`。

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3_b_stability.py
```

默认输出：

```text
D:\easm_project01\lead_lag_screen\V3\outputs\eof_pc1_smooth5_v3_b_stability_a
```

可选参数：

```bat
python lead_lag_screen\V3\scripts\run_lead_lag_screen_v3_b_stability.py --relation-bootstrap 1000 --mode-bootstrap 500
```

## 主要输出

```text
eof_pc1_mode_stability.csv
eof_pc1_lag_vs_tau0_stability.csv
eof_pc1_forward_reverse_stability.csv
eof_pc1_peak_lag_stability.csv
eof_pc1_pair_summary_stability_judged.csv
v1_index_vs_v3b_pc1_stability_comparison.csv
t3_meiyu_end_pc1_stability_audit.csv
summary.json
run_meta.json
```

## 判定边界

`stable_lagged_lead` 才能较正式地说 EOF-PC1 层存在稳定正滞后优势。

`significant_lagged_but_tau0_coupled` 只能说明正滞后显著，但不能说 lag 稳定强于同期。

`stable_tau0_dominant_coupling` 说明同期/快速调整主导，不应解释为稳定超前滞后。

`bidirectional_or_reverse_competitive` 说明方向拆分不稳。

`pc1_mode_limited` 说明 PC1 模态本身不稳，不能作为强证据。

## 硬边界

本层不做 pathway、PCMCI、因果发现或中介路径检验。
