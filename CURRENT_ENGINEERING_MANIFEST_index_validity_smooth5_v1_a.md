\
# 当前新增工程清单：index_validity_smooth5_v1_a

## 新增层

```text
index_validity/V1
```

## 单一入口

```text
index_validity/V1/scripts/run_index_validity_smooth5_v1.py
```

## 单一输出目录

```text
index_validity/V1/outputs/smooth5_v1_a
```

## 核心代码文件

```text
index_validity/V1/src/index_validity_v1/settings.py
index_validity/V1/src/index_validity_v1/pipeline.py
index_validity/V1/src/index_validity_v1/data_io.py
index_validity/V1/src/index_validity_v1/index_metadata.py
index_validity/V1/src/index_validity_v1/yearwise_diagnostics.py
index_validity/V1/src/index_validity_v1/physical_composites.py
index_validity/V1/src/index_validity_v1/plotting_yearwise.py
index_validity/V1/src/index_validity_v1/plotting_physical.py
index_validity/V1/src/index_validity_v1/anomaly_basic_check.py
index_validity/V1/src/index_validity_v1/logging_utils.py
```

## 工程边界

```text
只检查指数有效性；
不检查 lead-lag；
不检查 pathway；
不检查下游影响。
```
