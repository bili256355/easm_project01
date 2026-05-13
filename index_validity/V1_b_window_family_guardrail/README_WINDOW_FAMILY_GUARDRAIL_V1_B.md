# index_validity/V1_b_window_family_guardrail

## 定位

这是一个 **窗口 × 对象族的指数场指示性守门审计层**。它只回答：

> 在每个窗口内，每个对象族（P/V/H/Je/Jw）是否仍至少有部分人工指数对本对象的 5 日平滑场本身有指示作用？是否存在“某窗口某对象族全族指数失效”的风险？

它不检验 lead-lag、pathway、因果、场间耦合，也不证明每个指数在每个窗口都完美。

## 关键修正：默认主口径为 smoothed，不是 anomaly

index_validity 的主问题是“指数是否指示自身场对象”。因此默认使用：

```text
smoothed_fields.npz
index_values_smoothed.csv
```

而不是 anomaly。

`anomaly` 只保留为可选辅助审计，用于检查进入 lead-lag 前的 anomaly 口径；它不是本层默认主结果。

## 运行

在完整本地工程中运行主口径：

```bat
cd /d D:\easm_project01
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py
```

默认输出：

```text
D:\easm_project01\index_validity\V1_b_window_family_guardrail\outputs\window_family_guardrail_v1_b_smoothed_a
```

可选 anomaly 辅助口径：

```bat
python index_validity\V1_b_window_family_guardrail\scripts\run_index_validity_window_family_guardrail_v1_b.py --data-mode anomaly
```

默认 anomaly 辅助输出：

```text
D:\easm_project01\index_validity\V1_b_window_family_guardrail\outputs\window_family_guardrail_v1_b_anomaly_aux_a
```

## 输入

主口径默认读取：

```text
foundation/V1/outputs/baseline_smooth5_a/indices/index_values_smoothed.csv
foundation/V1/outputs/baseline_smooth5_a/preprocess/smoothed_fields.npz
```

可选 anomaly 口径读取：

```text
foundation/V1/outputs/baseline_smooth5_a/indices/index_anomalies.csv
foundation/V1/outputs/baseline_smooth5_a/preprocess/anomaly_fields.npz
```

若 `anomaly_fields.npz` 不存在，anomaly 模式才会尝试使用：

```text
smoothed_fields.npz - daily_climatology.npz
```

若本地 preprocess 文件不存在，脚本会明确报错并停止。

## 核心输出

```text
tables/input_file_audit.csv
tables/index_window_representativeness.csv
tables/window_family_guardrail.csv
tables/t3_index_representativeness_audit.csv
tables/t3_window_family_guardrail_audit.csv
tables/figure_manifest.csv
summary/summary.json
summary/run_meta.json
figures/selected_composite_maps/*.png
```

## 主要量化指标

单指数层：

- high-low composite expected contrast
- composite effect size
- weighted field R²
- EOF top-k alignment
- year-bootstrap composite pattern stability

对象族层：

- `family_collapse_risk = low`
- `family_collapse_risk = partial_sensitivity`
- `family_collapse_risk = high`

## 解释边界

- `low` 表示“不支持全族指数失效”，不表示该族每个指数都强。
- `partial_sensitivity` 表示应使用 index-level flags，不能把所有同族指数混成同等可靠。
- `high` 需要人工复核地图后再决定是否停用该 family-window。
- 主结果必须优先看 smoothed 口径；anomaly 口径只能作为辅助。
- 这一步不替代 V1 lead-lag，也不进入 pathway 结论。


## Joint family field coverage enhancement

This patch also writes `tables/window_family_joint_field_coverage.csv`. It asks: given all indices of one object family in one window, how much of that family's own field structure can the index set jointly indicate?

Key fields:

- `joint_field_R2_in_sample`: upper-bound weighted full-field R² from all same-family indices.
- `joint_field_R2_year_cv`: leave-one-year-out weighted full-field R²; preferred field-coverage metric.
- `joint_eof_coverage_top5_year_cv` / `top3`: year-CV coverage of the field's leading EOF-score subspace by the index family.
- `best_single_index_R2`, `mean_single_index_R2`, `joint_gain_over_best_single`: comparison with single-index R².

This remains an index-validity guardrail, not lead-lag/pathway/causal evidence.
