\
# index_validity/V1: smooth5 index validity diagnostics

## 定位

本层只检查 5 日 smoothed index 本身是否像一个可信指数，以及该指数是否物理上指示其代表的底层场结构。

它不做：

```text
lead-lag
自相关风险解释
分窗口稳定性
pathway
下游影响
raw vs anomaly lead-lag
```

## 入口

```bat
cd /d D:\easm_project01
python index_validity\V1\scripts\run_index_validity_smooth5_v1.py
```

## 默认输入

```text
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_values_smoothed.csv
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_daily_climatology.csv
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\indices\index_anomalies.csv
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz
```

## 默认输出

```text
D:\easm_project01\index_validity\V1\outputs\smooth5_v1_a
```

## 输出

### Tables

```text
tables\input_file_audit.csv
tables\yearwise_index_shape_audit.csv
tables\anomaly_reconstruction_basic_check.csv
tables\anomaly_daily_mean_basic_check.csv
tables\physical_composite_sample_info.csv
tables\physical_representativeness_summary.csv
```

### Figures

```text
figures\yearwise\spaghetti_<index>.png
figures\yearwise\year_day_heatmap_<index>.png
figures\yearwise\year_metrics_<index>.png

figures\physical\map_<index>.png
figures\physical\profile_<index>.png
```

`physical_representativeness_summary.csv` 中的 `overall_grade` 默认为 `to_review`，需要人工根据图件判读。
