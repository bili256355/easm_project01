# Patch Manifest: T3 V→P Field Explanation Audit v1_a no-NPZ + cartopy hotfix

## Patch type

替换文件式修正补丁。

## Purpose

修正上一版 `t3_v_to_p_field_explanation_audit_v1_a` 中两个工程/输出问题：

1. 删除不必要的 `maps/v_to_p_field_explanation_maps.npz` 全格点中间场持久化输出；
2. 将 field-explanation 图件接入 cartopy，默认使用 cartopy 地图绘制。

## Files replaced / added

```text
README_T3_V_TO_P_FIELD_EXPLANATION_AUDIT_V1_A.md
PATCH_MANIFEST_T3_V_TO_P_FIELD_EXPLANATION_AUDIT_V1_A.md
scripts/run_t3_v_to_p_field_explanation_audit_v1_a.py
src/lead_lag_screen_v1/t3_v_to_p_field_explanation_io.py
src/lead_lag_screen_v1/t3_v_to_p_field_explanation_pipeline.py
src/lead_lag_screen_v1/t3_v_to_p_field_explanation_settings.py
```

The unchanged helper modules are also included for convenience so the patch can be applied to a clean V1 tree or over the previous v1_a patch:

```text
src/lead_lag_screen_v1/t3_v_to_p_field_explanation_math.py
src/lead_lag_screen_v1/t3_v_to_p_field_explanation_regions.py
```

## Output changes

Removed:

```text
maps/v_to_p_field_explanation_maps.npz
maps/map_metadata.json
```

Added/changed:

```text
figures/*_cartopy.png       # when cartopy is available
figure_manifest.csv         # includes plot_backend
summary.json                # includes full_grid_map_arrays_persisted=false
```

## Runtime command

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py --n-bootstrap 1000
```

Emergency fallback without cartopy:

```bat
python D:\easm_project01\lead_lag_screen\V1\scripts\run_t3_v_to_p_field_explanation_audit_v1_a.py --n-bootstrap 1000 --no-cartopy
```

## Verification performed before delivery

- Python syntax compilation for patched files.
- Static search confirms no output write to `v_to_p_field_explanation_maps.npz` remains.
- Static search confirms cartopy backend is wired into the new field-explanation plotting function.
