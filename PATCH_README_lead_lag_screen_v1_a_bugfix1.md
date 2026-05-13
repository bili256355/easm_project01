# lead_lag_screen_v1_a_bugfix1

## 修复内容

修复 `_bootstrap_direction_for_window()` 中 bootstrap 年份抽样后的索引维度错误。

原错误代码：

```python
boot = panel[year_idx, :, :, :]
```

原因：

```text
panel 是三维数组：(year, day, variable)
year_idx 是二维 bootstrap 年份索引：(B, n_year)
使用 panel[year_idx, :, :] 后，NumPy advanced indexing 会自然得到：
(B, n_year, day, variable)
```

修复为：

```python
boot = panel[year_idx, :, :]
```

## 是否改变方法语义

不改变。  
该修复只接通 bootstrap 抽样执行链，不改变统计判据、输出目录、输入路径或方法定义。

## 替换方式

将本补丁中的文件覆盖到：

```text
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\core.py
```

然后重新运行：

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```
