# lead_lag_screen_v2_runtime_b 性能/可恢复性补丁

## 修改目标

本补丁只改 V2 的运行调度，不改方法语义。

保持不变：

```text
20 个 5日 anomaly index
9 个窗口
PCMCI+
LPCMCI
tau = 0..5
ParCorr
analysis_mode = multiple
每一年作为一个 dataset，不跨年拼接
```

新增：

```text
1. window × method 级别并行；
2. 每个任务单独 checkpoint；
3. 已完成任务自动跳过；
4. 失败任务单独记录；
5. runtime_task_status.csv；
6. runtime_task_timing.csv；
7. summary.json 写入任务成功/失败数量；
8. max_workers / skip_completed_tasks / fail_fast 配置。
```

## 覆盖文件

```text
D:\easm_project01\lead_lag_screen\V2\src\lead_lag_screen_v2\settings.py
D:\easm_project01\lead_lag_screen\V2\src\lead_lag_screen_v2\pipeline.py
D:\easm_project01\lead_lag_screen\V2\README_PCMCI_PLUS_LPCMCI_V2.md
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_pcmciplus_lpcmci_v2.py
```

## 配置

默认：

```text
max_workers = 4
skip_completed_tasks = True
fail_fast = False
```

如果机器内存吃紧，可把 `max_workers` 改为 2。
如果想严格遇错即停，可把 `fail_fast` 改为 True。

## 新增运行状态输出

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_lpcmci_v2_a\runtime_task_status.csv
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_lpcmci_v2_a\runtime_task_timing.csv
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_lpcmci_v2_a\_task_cache\*.csv
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_lpcmci_v2_a\_runtime_status\*.json
```
