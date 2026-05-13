\
# lead_lag_screen_v1_a 补丁说明

## 用途

新增 `D:\easm_project01\lead_lag_screen\V1`，作为 pathway 重建前的正式 lead–lag 时间资格筛查层。

本补丁不覆盖：

```text
foundation
stage_partition
pathway
```

也不读取旧 pathway V1/V2 结果。

## 安装方式

将压缩包中的 `lead_lag_screen` 文件夹复制到：

```text
D:\easm_project01
```

最终结构应为：

```text
D:\easm_project01\lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
D:\easm_project01\lead_lag_screen\V1\src\lead_lag_screen_v1\...
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V1\scripts\run_lead_lag_screen_v1.py
```

## 默认输出

```text
D:\easm_project01\lead_lag_screen\V1\outputs\lead_lag_screen_v1_a
```

## 注意

本层是 temporal eligibility screen，不是 pathway establishment。
