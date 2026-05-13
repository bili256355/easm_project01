# 当前新增/修改工程清单：lead_lag_screen_v2_runtime_b

## 修改层

```text
lead_lag_screen/V2
```

## 修改文件

```text
lead_lag_screen/V2/src/lead_lag_screen_v2/settings.py
lead_lag_screen/V2/src/lead_lag_screen_v2/pipeline.py
lead_lag_screen/V2/README_PCMCI_PLUS_LPCMCI_V2.md
```

## 运行入口不变

```text
lead_lag_screen/V2/scripts/run_pcmciplus_lpcmci_v2.py
```

## 输出目录不变

```text
lead_lag_screen/V2/outputs/pcmci_plus_lpcmci_v2_a
```

## 新增输出

```text
runtime_task_status.csv
runtime_task_timing.csv
runtime_failed_tasks.csv  # 仅任务失败时生成
_task_cache/
_runtime_status/
```

## 语义边界

```text
只改运行调度、checkpoint、resume 与日志；
不改变变量池；
不改变窗口；
不改变 tau；
不改变 PCMCI+ / LPCMCI 设置；
不改变 V1-V2 对照规则。
```
