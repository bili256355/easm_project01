\
# lead_lag_screen_v2_pcmciplus_lpcmci_a 补丁说明

## 新增内容

新增独立层：

```text
D:\easm_project01\lead_lag_screen\V2
```

统一入口：

```text
D:\easm_project01\lead_lag_screen\V2\scripts\run_pcmciplus_lpcmci_v2.py
```

默认输出：

```text
D:\easm_project01\lead_lag_screen\V2\outputs\pcmci_plus_lpcmci_v2_a
```

## 方法

```text
PCMCI+
LPCMCI
```

设置：

```text
20 个 5日 anomaly index
9 个窗口
tau = 0..5
ParCorr
analysis_mode = multiple
每一年作为一个 dataset，不跨年拼接
```

## 运行

```bat
cd /d D:\easm_project01
python lead_lag_screen\V2\scripts\run_pcmciplus_lpcmci_v2.py
```

## 依赖

需要本地 Python 环境已安装 tigramite。

如未安装，会 fail-fast 报错，不会静默跳过。
