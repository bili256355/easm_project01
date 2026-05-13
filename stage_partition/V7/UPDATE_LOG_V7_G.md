# UPDATE_LOG_V7_G

## v7_g - W45 H local latitude-band progress diagnostic

新增独立入口：

```bat
python D:\easm_project01\stage_partition\V7\scripts\run_w45_H_latband_progress_v7_g.py
```

新增模块：

```text
stage_partition/V7/src/stage_partition_v7/w45_H_latband_progress.py
```

本版本只针对 W45 / anchor=45 的 H(z500) 场，把 V7-f 中的 single-lat feature progress 改为 sliding local 3-latitude-band progress。

核心目的：

- 检查 V7-f 的 single-lat feature instability 是否可由连续纬度带聚合缓解；
- 判断 W45-H whole-field early-broad progress 是否可能来自纬向内部异质性；
- 形成上游实现诊断：若 lat-band 明显改善，后续 region/feature-level progress 不应继续默认使用 single feature 作为基本分析单元。

边界：

- 不覆盖 V7-e whole-field progress 结果；
- 不修改 V7-e1 / V7-e2 的统计检验；
- 不接 downstream lead-lag/pathway；
- 不做因果解释；
- 不扩展到其他窗口或其他场。
