# T3 V→P Field-Explanation Hard-Evidence Audit v1_a 修正补丁

## 本次修正定位

这是对上一版 `t3_v_to_p_field_explanation_audit_v1_a` 的替换文件式修正补丁，不新开完整包，不覆盖旧 T3 physical audit。

本次只修正两件事：

1. **删除不必要的大体积全格点 NPZ 输出**
   - 不再输出 `maps/v_to_p_field_explanation_maps.npz`。
   - 全格点 map 只作为内存中间量，用于生成区域汇总、相似性表、诊断表和可选图件。
   - 硬证据输出以 tables / figures / summary 为准。

2. **field-explanation 图件默认改为 cartopy 地图绘制**
   - 默认使用 cartopy `PlateCarree` 投影绘制地图图件。
   - 图中叠加核心预注册区域框：`main_meiyu`、`south_china`、`scs`、`north_northeast`。
   - 如果 cartopy 不可用，代码会回退到普通 lon/lat 图，并在 `figure_manifest.csv` 中记录 `plot_backend`。
   - 如需强制不用 cartopy，可加 `--no-cartopy`。

## 新增/替换入口

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py
```

快速测试：

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py --n-bootstrap 100 --no-figures
```

正式运行：

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py --n-bootstrap 1000
```

如果 cartopy 环境有问题但仍想先产出表格和普通图：

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py --n-bootstrap 1000 --no-cartopy
```

## 输出目录

默认仍输出到：

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_smooth5_a_t3_v_to_p_field_explanation_audit_v1_a
```

## 当前保留的核心输出

```text
tables/field_explained_variance_manifest.csv
tables/region_response_summary.csv
tables/region_response_bootstrap_ci.csv
tables/lag_tau0_region_delta_summary.csv
tables/component_contrast_region_summary.csv
tables/window_map_similarity.csv
tables/window_region_vector_similarity.csv
tables/full_vs_subwindow_dilution_summary.csv
tables/hard_evidence_diagnosis_table.csv
tables/figure_manifest.csv              # 如果生成 figures
figures/*.png                           # 默认 cartopy 地图图件
summary/summary.json
summary/run_meta.json
summary/README_T3_V_TO_P_FIELD_EXPLANATION_AUDIT_V1_A.md
```

## 明确不再输出

```text
maps/v_to_p_field_explanation_maps.npz
maps/map_metadata.json
```

## 科学边界

- 这仍然是硬证据审计层，不是 pathway / causal establishment 层。
- field explanation 图件是视觉辅助，最终判定以区域响应、相似性、component contrast、lag-vs-tau0 和 diagnosis table 为主。
- 不允许把 cartopy 地图上的局地响应直接写成机制成立。
- 不允许把没有持久化 NPZ 理解成“没有计算场图”；场图仍在内存中计算，只是不再保存为大体积中间文件。
