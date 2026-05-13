# CURRENT_PACKAGE_STATUS

本文件描述当前整工程包的真实状态。

---

## 1. foundation
- `foundation/V1`：当前正式基础层
- 负责 preprocess 与 indices
- 是后续 stage_partition 层的上游底座

## 2. stage_partition
- `V1`：骨架
- `V2`：较早历史审计链
- `V3`：历史主线对照分支
- `V4`：当前活动 point-layer 分支

### 当前 V4 的状态
当前包内 V4 已具备：
- point-layer detector 主链
- raw -> formal point 映射
- point audit universe
- neighbor competition
- yearwise / bootstrap / parameter-path 支持
- point-level B-track summary

同时，当前 V4 仍应被视为：
- 可运行
- 可审理
- 但尚未完全收口的活动版本

## 3. pathway
- 当前仅 `pathway/V1` 骨架

---

## 4. 如何理解 bundled outputs

### `stage_partition/V3/outputs/*`
- 历史对照结果
- 主要用于审理 V4 与历史主线差异

### `stage_partition/V4/outputs/mainline_v4_a`
- 包内自带的当前 V4 运行结果
- 用于展示当前代码树的一份已有结果
- 不自动等同于未来任意新运行的唯一结果

---

## 5. 当前最容易误读的地方
1. 根目录旧 patch 文档仍然存在，但它们是历史记录，不是当前整包主说明。
2. `stage_partition/V3` 被保留，是因为它是重要对照基线，不等于当前主线仍在 V3。
3. 当前 V4 已经不是最早骨架，但也还不是最终收口版。
