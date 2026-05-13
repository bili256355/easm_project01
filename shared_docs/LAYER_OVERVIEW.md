# LAYER_OVERVIEW

本文件给出当前整工程包中各层的职责与边界。

---

## foundation

负责：
- 输入数据契约检查
- 场级 smoothing
- daily climatology
- anomaly
- 对象指数计算
- 指数日气候态与异常
- 基础日志与 preprocessing 输出

不负责：
- 阶段划分
- 点/窗检测结果解释
- 路径识别

当前正式实现：
- `foundation/V1`

---

## stage_partition

负责：
- 点层状态表达
- 阶段划分相关 detector 与审计
- point/window 审计链
- yearwise / bootstrap / parameter-path 支持层

当前包内版本角色：
- `V1`：骨架
- `V2`：较早审计链与历史输出
- `V3`：历史主线对照分支
- `V4`：当前活动 point-layer 分支

说明：
- 当前包内最值得优先理解的是 `V4`
- 当前包内最重要的历史对照基线是 `V3`

---

## pathway

负责：
- 路径层表达
- 路径竞争/接替/共享审计

当前状态：
- 包内仅 `pathway/V1` 骨架
- 不是当前正式运行主线
