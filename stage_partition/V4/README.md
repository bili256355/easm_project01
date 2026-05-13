# stage_partition/V4

`stage_partition/V4` 是当前工程包中的**活动分支**，目标是一个独立的 point-layer B-track branch。

---

## 当前硬边界
- A-track 不进入 V4 主线。
- window-layer 逻辑不进入 V4 当前主线。
- pathway-layer 逻辑不进入 V4 当前主线。
- V4 依赖 `foundation/V1/outputs/baseline_a/preprocess/smoothed_fields.npz` 作为上游输入。
- V4 不读取 `mainline_v3_*` 结果表作为运行输入。

---

## 当前 V4 在本工程包中的定位

当前包内的 V4 不是最早骨架版，也不是完全收口版；它已经承载：
- point-layer detector 主链
- raw -> formal point 映射
- point audit universe
- neighbor competition
- yearwise / bootstrap / parameter-path 支持
- point-level B-track summary

同时，当前包内 V4 仍应被视为：
- **可运行、可审理、但尚未完全收口** 的活动分支

---

## 当前最重要的对照关系

若要审理 V4：
- 代码基线以当前包内 `stage_partition/V4/src/stage_partition_v4/` 为准
- 历史结果与逻辑对照，以 `stage_partition/V3` 为重要基线

说明：
- V3 的意义是“对照与审计”，不是当前主线承载体
- V4 的意义是“当前活动代码树”，不是历史 patch 汇编册

---

## 当前包内 bundled outputs 的理解方式

`outputs/mainline_v4_a/` 是包内自带的一份 V4 运行结果，用于：
- 当前状态展示
- 审理与对照
- 结果包结构参考

它不自动等同于：
- 唯一最终科研结果
- 后续每一轮新运行的替代品

如果后续继续推进，应以：
- 当前 V4 代码树
- 新运行生成的新输出
作为新的唯一审理对象。
