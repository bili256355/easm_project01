# V1 surrogate_failure_autocorr_audit_a

独立审计层。只读 V1 已有 `lead_lag_pair_summary_stability_judged.csv` 和 V1 指数时间序列，输出 T3/T4 以及全窗口的 surrogate 失败机制审计结果。

## 运行

```bat
python D:\easm_project01\lead_lag_screen\V1\surrogate_failure_autocorr_audit_a\run_v1_surrogate_failure_autocorr_audit_a.py
```

显式指定输入：

```bat
python D:\easm_project01\lead_lag_screen\V1\surrogate_failure_autocorr_audit_a\run_v1_surrogate_failure_autocorr_audit_a.py ^
  --stability-csv D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_stability_judgement_v1_b\tables\lead_lag_pair_summary_stability_judged.csv ^
  --index-csv D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a\indices\index_anomalies.csv
```

## 主要输出

- `tables/index_autocorr_by_window.csv`
- `tables/index_autocorr_summary_by_window.csv`
- `tables/pair_autocorr_context.csv`
- `tables/autocorr_by_v1_status_summary.csv`
- `tables/surrogate_threshold_margin_by_pair.csv`
- `tables/surrogate_failure_attribution_by_pair.csv`
- `tables/surrogate_failure_attribution_counts_by_window.csv`
- `tables/t3_t4_vs_other_window_autocorr_surrogate_contrast.csv`
- `tables/surrogate_failure_autocorr_diagnosis_table.csv`

## 审计边界

这不是 V1 主 surrogate 检验的替代品。脚本使用 AR(1) effective-n 近似生成 max-over-positive-lags null 阈值，用于判断 surrogate 失败更像 observed peak weak，还是 null threshold high。最终成立性仍以 V1 主结果为准。
