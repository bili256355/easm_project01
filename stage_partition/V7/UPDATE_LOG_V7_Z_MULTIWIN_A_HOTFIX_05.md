# V7-z-multiwin-a hotfix_05 raw-state helper

修复 hotfix_04 中遗漏的 `_raw_state_matrix_v7z_from_year_cube` 定义。

## 修复内容

- 新增 `_zscore_features_v7z()`。
- 新增 `_raw_state_matrix_v7z_from_year_cube()`。
- observed 与 bootstrap detector 均恢复为：year/day/feature cube → sampled climatology → feature-wise z-score along day → ruptures.Window。
- 不改窗口范围、baseline、pre-post、2D mirror、evidence gate。

## 目的

让 W45 profile 回归测试能够真正运行到 detector 输出阶段。
