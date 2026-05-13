# 当前工程修改清单：lead_lag_screen_v2_pcmciplus_only_c

## 修改层

```text
lead_lag_screen/V2
```

## 入口脚本

保持不变：

```text
lead_lag_screen/V2/scripts/run_pcmciplus_lpcmci_v2.py
```

说明：

```text
入口名保留兼容，但实际只运行 PCMCI+。
```

## 新输出目录

```text
lead_lag_screen/V2/outputs/pcmci_plus_v2_c
```

## 修改内容

```text
LPCMCI disabled/removed from runtime
PCMCI+ graph direction hotfix
same-family edge exclusion
runtime checkpoint retained
```

## 不改变内容

```text
20 variables remain in conditioning pool
9 windows
tau 0..5
ParCorr
analysis_mode multiple
yearwise multiple datasets
V1 evidence-tier reference
```
