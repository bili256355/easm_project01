# UPDATE_LOG_V7_F

## v7_f - W45 H feature-level progress diagnostic

新增独立入口：

```bat
python D:\easm_project01\stage_partition\V7\scripts\run_w45_H_feature_progress_v7_f.py
```

新增模块：

```text
stage_partition/V7/src/stage_partition_v7/w45_H_feature_progress.py
```

本版本只针对 W45 / anchor=45 的 H(z500) 场，检查 whole-field H early-broad progress 是否由纬向 feature-level timing 异质性造成。

核心边界：

- 不覆盖 V7-e whole-field progress 结果。
- 不修改 V7-e1 / V7-e2 的统计检验。
- 不接 downstream lead-lag/pathway。
- 不做因果解释。
- 只输出 H feature-level progress 诊断、whole-vs-feature 对照和下一步决策建议。


## hotfix_01 - V6 import path repair

- Fixed `ModuleNotFoundError: No module named 'stage_partition_v6'` when running `run_w45_H_feature_progress_v7_f.py`.
- Cause: the V7 script added `stage_partition/V7/src` to `sys.path`, but the new diagnostic reuses V6 profile/state helpers from `stage_partition/V6/src`.
- Repair: add sibling `stage_partition/V6/src` to `sys.path` in both the runner and the diagnostic module before importing `stage_partition_v6`.
- Method/output semantics are unchanged. This is an import-chain hotfix only.
