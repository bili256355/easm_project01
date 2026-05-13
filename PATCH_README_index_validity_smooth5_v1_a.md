\
# index_validity_smooth5_v1_a 补丁说明

## 用途

新增独立层：

```text
D:\easm_project01\index_validity\V1
```

用于检查：

```text
1. 5日 smoothed index 的单年份偏移 / 乱飞情况；
2. index 高低值是否对应其代表的底层物理场；
3. anomaly 只做最低限度的基本重构检查。
```

## 安装

将压缩包中的 `index_validity` 文件夹复制到：

```text
D:\easm_project01
```

## 运行

```bat
cd /d D:\easm_project01
python index_validity\V1\scripts\run_index_validity_smooth5_v1.py
```

## 不涉及内容

```text
不修改 foundation
不修改 lead_lag_screen
不读取 pathway
不评价 lead-lag
不评价下游影响
不做自相关风险解释
```
