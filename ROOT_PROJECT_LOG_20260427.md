# easm_project01 根目录接管日志

**文件用途**：本文件用于替代 `D:\easm_project01` 根目录中原先散落的非文件夹说明文件，作为当前工程接管后的根目录唯一留存日志。  
**生成日期**：2026-04-27  
**适用工程根目录**：`D:\easm_project01`  
**日志性质**：工程接管日志 / 状态边界记录 / 防混淆说明。  
**重要说明**：本日志不是科研结果表，不是论文结论，也不是运行输出本身；它只记录当前接管时已经确认的工程事实、研究语境边界、已删除内容状态和后续推进前必须遵守的约束。

---

## 1. 当前根目录清理策略

用户计划删除根目录中已有的所有非文件夹内容，只保留本日志文件。  
因此，本日志应承担以下作用：

1. 记录当前工程真实接管状态；
2. 记录哪些内容已确认存在；
3. 记录哪些内容已删除或不能再作为底座；
4. 记录哪些事实不能被误读；
5. 为后续新会话、新补丁、新版本开发提供根目录级入口说明。

本日志不替代各子目录内部的源码、配置、输出或版本说明。后续具体判断仍应以对应版本目录中的代码、配置、`run_meta`、`summary` 和输出表为准。

---

## 2. 当前已确认的基础事实

### 2.1 工程根目录

当前工程根目录为：

```text
D:\easm_project01
```

当前工程采用多版本隔离结构，核心子目录包括：

```text
foundation\V1
stage_partition\V1
stage_partition\V2
stage_partition\V3
stage_partition\V4
stage_partition\V5
stage_partition\V6
stage_partition\V6_1
pathway
shared_docs
logs
transfer_packages
```

其中，`foundation` 与 `stage_partition` 是当前可确认的工程主体；`pathway` 当前不再作为旧结果继承底座。

---

## 3. 基础层与预处理事实

### 3.1 本地基础层输出完整存在

虽然传给接管方的压缩工程包中未包含大体积预处理结果，但用户已明确确认：

```text
本地完整的 foundation/V1 基础层输出存在；
工程能够基于本地完整基础层输出运行。
```

因此，后续不得将上传包内缺少 preprocess 文件误判为工程不可运行或基础层失败。

正确表述应为：

```text
上传包内 preprocess 大文件未包含，是传输限制；
本地 preprocess 完整存在，并可支撑后续 stage_partition 工程运行。
```

---

### 3.2 原始数据经过 9 日滑动处理

用户已明确确认：

```text
原始数据是 9 日滑动的；
因此在开头和末尾出现 NaN 值是预期现象。
```

该事实必须长期保留。后续若在 `smoothed_fields`、profile 构造、valid-day trimming 或 stage detection 中看到首尾 NaN，不得直接判定为数据损坏或预处理失败。

正确理解：

```text
9 日滑动窗口会导致序列两端无法形成完整窗口；
开头和末尾 NaN 属于平滑处理带来的边界缺失；
stage_partition 检测前应通过 shared valid-day trimming 或等价机制处理这些边界缺失；
detector 结果需要再映射回原始 day index。
```

注意：

```text
首尾 NaN 是预期边界效应；
但内部 NaN、异常大面积缺失、变量间不一致缺失仍需单独审计，不能自动归因于 9 日滑动。
```

---

## 4. 当前工程层真实状态

### 4.1 foundation/V1

`foundation/V1` 是当前基础层工程。其职责包括：

```text
读取原始 npy 数据；
生成 smoothed fields；
生成 anomaly fields；
生成 smoothed index values；
生成 index anomalies；
保存 indices 与 preprocess 输出。
```

当前接管事实：

```text
本地基础层输出完整存在；
上传包中 indices 输出可见；
上传包中 preprocess 大文件未包含，但不代表本地缺失。
```

---

### 4.2 stage_partition

`stage_partition` 是当前工程包中最主要、最完整的活动工程区。各版本角色需区分：

| 版本 | 当前接管角色 |
|---|---|
| `V1` | 历史失败/旧路线；因 anomaly 进入阶段划分主状态表达，被判定为对象定义错误 |
| `V2` | 历史过渡/旧 support-retention 审计链；不得混回当前主结果 |
| `V3` | 阶段窗口线的重要历史主线与 V4 合理性参照 |
| `V4` | 当前点层 B-track 干净化主线核心；不引入 A-track |
| `V5` | 局部峰稳定性窄分支 |
| `V6` | 点层 bootstrap screening 分支，含 1000 bootstrap 等支持表达 |
| `V6_1` | 从 V6 点层结果派生的 lightweight window layer |

后续不能仅凭版本号大小判断主线地位。`V6_1` 更新，并不自动意味着它替代 `V4` 成为全部主线。

---

### 4.3 pathway

当前 `pathway` 层必须按以下状态接管：

```text
旧 pathway V1–V2 内容已经被用户删除；
原因是旧路径层内容过于缠绕，且 V1–V2 暂不能用；
当前不再反复修补旧 pathway；
后续若做 pathway，应直接重建。
```

因此：

```text
pathway 旧工程结果不可作为当前可运行底座；
pathway 迁移包中的内容只保留为经验、禁区、失败记录和设计约束；
不能把旧 pathway 结果写成当前可用科研结果。
```

---

## 5. 当前研究语境边界

当前项目总科学问题仍是：

```text
在东亚夏季风 4–9 月季节推进过程中，
降水纬向结构与 Jw / Je / H / V / P 等对象之间的关系骨架
如何配置、竞争、接替与重组；
这些表层关系配置及其背后的上游影响路径，
是否也会随季节推进共同变化，并形成有组织的阶段结构。
```

阶段划分不是最终科学结论本身，而是为关系骨架和 pathway 研究提供时间对象底座。

必须区分：

```text
stage_partition 点/窗口支持结果
≠ pathway 关系骨架结果
≠ 物理机制闭环
≠ 论文最终结论
```

---

## 6. 当前必须保留的 pathway 经验与禁区

虽然旧 pathway 工程已删除，但以下经验必须保留：

### 6.1 不再继承旧 V1–V2 作为工程底座

旧 V1–V2 路径层不能继续修修补补。后续应直接重建新 pathway 线。

### 6.2 pathway 不能只做 terminal-P

新 pathway 必须保留 P 的多角色：

```text
1. P as terminal:
   X -> P
   X -> M -> P

2. P feedback:
   P -> X
   P -> M -> Y

3. P as mediator:
   X -> P -> Y

4. nonP internal:
   Jw / Je / H / V 之间的 direct 或 mediator-like 关系
```

任何只聚焦 terminal-P 的结果必须明确标为子集，不能代表 full-role 主线。

### 6.3 禁止真实年份错配 permutation

不得使用类似：

```text
1981 年季风变量解释 1982 年降水或环流变量
```

这种真实年份错配方式作为 null。  
可保留的经验是 AR(1) synthetic surrogate：生成与原始 year-day 结构等长的合成数据，并在合成数据上重复主计算。

### 6.4 PCMCI / Liang / EOF-PC1 的地位

这些方法可以作为压力测试、对照或外部参照，但不能直接替代主线 pathway 判据，也不能作为所有路径的硬筛选规则。

---

## 7. 当前 stage_partition 防混淆规则

### 7.1 V1 已作废

`stage_partition/V1` 的核心错误是：

```text
把 anomaly 引入阶段划分主状态表达。
```

这不是小 bug，而是对象定义错误。V1 只保留为失败记录。

---

### 7.2 V3 / V4 的关系

`V3` 是窗口线历史主线和合理性参照；`V4` 是点层 B-track 干净化主线。

后续审理 V4 时必须遵守：

```text
先锁定 artifact 身份；
再做 V3 ↔ V4 逐表差分；
再判断差异是否为 BUG；
不能只根据 V4 症状直接讲原因。
```

---

### 7.3 A-track 不进入 V4

V4 第一阶段只承载点层 B-track。  
A-track 保留在 V3 历史线中，不进入 V4 主线。

---

### 7.4 点、窗口、解释层必须分开

必须区分：

```text
detector raw points / score curve
formal primary points
local peaks / weak neighbor peaks
point audit universe
support bands
main windows
retained / dropped
物理解释或候选机制
```

这些不是同一层对象，不得混写。

---

## 8. 当前真实可依赖的工作底座

当前可依赖的工程底座：

```text
foundation/V1 本地完整基础层输出
stage_partition/V3 历史窗口线与参照结果
stage_partition/V4 点层 B-track 主线
stage_partition/V6 点层 bootstrap screening 分支
stage_partition/V6_1 lightweight window 派生分支
```

当前不可依赖为工程底座的内容：

```text
旧 pathway V1–V2
已删除的路径层旧结果
只存在于迁移包中的旧 pathway 输出描述
未进入执行链的设计草案
未核实的物理解释
```

---

## 9. 后续推进前的优先核对项

如果继续 `stage_partition`，优先核对：

```text
1. V4 当前基线是否锁定为 mainline_v4_d 或后续本地最新输出；
2. V4 point audit universe 是否完整保留 weak neighbor peaks；
3. 132/135 competition 是否已合理闭合；
4. yearwise exact / strict / near 的最终定义是否统一；
5. V6 与 V6_1 是否只是点层支持与轻量窗口派生，还是要晋升为主表达层；
6. 是否需要生成一个真正干净、单一基线的 V4 收口版本。
```

如果转入 `pathway` 重建，优先核对：

```text
1. 不继承旧 V1–V2 工程；
2. 从 yes/no 判据开始，不一次性塞入过多审计层；
3. lead-lag correlation 的实现口径先明确；
4. 每年单独计算后汇总，还是窗口内多年拼接，必须先定义；
5. 自相关修正、稳健性检验、强度检验的必要性分层；
6. full-role P 角色必须进入设计，不能只做 terminal-P。
```

---

## 10. 根目录日志维护规则

后续若更新本日志，应遵守：

```text
1. 只记录根目录级事实、版本状态和防混淆边界；
2. 不在根目录日志中粘贴大段结果表；
3. 不把未运行内容写成已完成；
4. 不把研究解释写成工程事实；
5. 不把工程文件存在写成科研结论成立；
6. 每次更新必须注明更新原因、更新日期和影响范围。
```

建议本文件作为根目录唯一保留说明文件，文件名为：

```text
ROOT_PROJECT_LOG_20260427.md
```

---

## 11. 当前接管摘要

当前接管状态可以压缩为：

```text
D:\easm_project01 当前可运行底座是：
foundation/V1 本地完整基础层输出
+ stage_partition 多版本工程。

其中：
V3 是阶段窗口历史主线与参照；
V4 是点层 B-track 当前核心；
V6 / V6_1 是后续点层支持与轻量窗口派生分支；
pathway 旧 V1–V2 已删除，不能继续作为底座。

原始数据经过 9 日滑动，首尾 NaN 是预期边界效应。
上传包缺少 preprocess 是传输限制，不是本地工程失败。
后续 pathway 应直接重建，但必须继承 full-role、AR(1) surrogate、禁止真实年份错配等经验与禁区。
```
