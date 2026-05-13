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

---

# 12. Lead–lag 时间资格层更新记录：5 日滑动与同期耦合处理

**更新日期**：2026-04-27  
**更新性质**：lead–lag 时间资格层结果审理与后续主线裁决。  
**影响范围**：

```text
foundation/V1
lead_lag_screen/V1
future pathway reconstruction
```

本节用于记录当前 lead–lag 时间资格层已经形成的新判断，避免后续把同期耦合、不稳定结果、主 null / audit null 分层和 pathway 结果混写。

---

## 12.1 5 日滑动将作为后续 lead–lag 主线基础

已完成 9 日滑动与 5 日滑动 anomaly index 的 lead–lag 对比。

当前判断：

```text
后续 lead–lag 时间资格层优先基于 5 日滑动 anomaly index 继续推进；
9 日滑动版本保留为 sensitivity / 对照结果。
```

理由：

```text
1. 5 日版本没有破坏主 lead–lag 信号；
2. main lead_lag_yes 数量与 9 日版本接近；
3. 5 日版本显著降低 AR(1) high-persistence / phi clipping 风险；
4. 5 日版本 Tier1 / Tier2 结果比例更高；
5. 5 日版本 Tier3 surrogate-sensitive 结果明显减少；
6. V→P 等核心方向在 5 日版本中仍然稳定存在。
```

边界：

```text
5 日版本更适合 lead–lag 时间资格层；
这不代表 stage_partition 阶段划分也必须改用 5 日；
不同层的平滑尺度可以不同，但必须明确标注。
```

---

## 12.2 anomaly index 仍然高度自相关是客观事实

已确认：

```text
即使使用 anomaly index，一阶自相关仍然可能很高；
anomaly 不等于 whitened series；
9 日滑动会进一步增强 persistence；
5 日滑动会缓解但不会完全消除日际 persistence。
```

因此后续不得写成：

```text
指数已经 anomaly，所以无需自相关修正。
```

正确表述：

```text
anomaly 去除了多年平均日气候态；
但并不消除季节内持续异常、日际 persistence 或滑动平均导致的样本相关。
```

---

## 12.3 当前 lead–lag 层的主判定结构

当前 lead–lag 层不再采用“四个并列 hard gate”：

```text
1. 自相关修正显著性
2. 强度阈值
3. 方向优势阈值
4. 稳健性检验
```

而是收束为：

```text
A. 统计支持层 = 自相关修正显著性 + null-relative strength
B. 方向稳健层 = 正滞后证据相对反向/同期证据的稳定性
C. 强度表征 = 分层、排序、解释，不作为固定 r hard gate
```

重要边界：

```text
不设置固定 r_min 作为强度 hard gate；
不使用固定倍率作为方向 hard gate；
lead_lag_yes 仍然只是时间资格，不是 pathway established。
```

---

## 12.4 Tier1 / Tier2 的当前使用规则

当前 evidence-tier 分层是对 lead–lag 结果的后处理 / 分层 pick，不改变原始 lead_lag_label。

### Tier1

核心含义：

```text
主 lead_lag 判定已经是 lead_lag_yes；
audit null 的窗口内 FDR 也通过；
没有 severe 级高自相关风险。
```

形式：

```text
lead_lag_group = lead_lag_yes
and q_pos_audit_within_window <= 0.10
and pair_phi_risk != severe
```

Tier1 分两类：

```text
Tier1a:
    lead_lag_yes_clear
    audit q pass
    no severe phi risk
    最干净的 clear lead–lag core

Tier1b:
    lead_lag_yes_with_same_day_coupling
    audit q pass
    no severe phi risk
    可作为时间资格候选，但必须保留 same-day coupling 风险
```

### Tier2

核心含义：

```text
主 lead_lag 判定已经是 lead_lag_yes；
audit null 的单项 p 值通过；
但 audit null 的窗口内 FDR 未通过；
没有 severe 级高自相关风险。
```

形式：

```text
lead_lag_group = lead_lag_yes
and p_pos_audit_surrogate <= 0.05
and q_pos_audit_within_window > 0.10
and pair_phi_risk != severe
```

解释：

```text
Tier2 可作为候选，但不能当作最稳核心；
后续必须带 audit-FDR 未过的风险标签。
```

---

## 12.5 S1 审理中的重要更新：同期内容必须显式保留

在 S1 结果整理中发现：

```text
S1 中接近一半方向存在显著 same-day coupling；
S1 的 Tier1 / Tier2 结果中，大多数是 lead_lag_yes_with_same_day_coupling；
因此 S1 的主结构不能被写成纯 lead–lag 传播。
```

正确解释：

```text
S1 更像是同步耦合 + 短滞后调整并存的窗口；
V→P 和 H→P 虽然稳定，但多数带显著 lag=0 成分。
```

后续整理每个窗口时必须区分：

```text
Tier1a: clear lead–lag
Tier1b: lead–lag with same-day coupling
Tier2: audit-moderate candidate
same-day-only / same-day-dominant: 当前不进入主 lead-lag 候选，但保留价值
```

---

## 12.6 关于 lag=0 的最新裁决

当前 lead–lag 层对 lag=0 的处理规则如下：

```text
lag=0 不作为超前证据；
lag=0 不参与 positive_peak_lag 的选择；
positive_peak_lag 只在 +1 到 +5 天范围内寻找；
lag=0 单独进入 same-day coupling 诊断。
```

`正滞后显著` 的含义是：

```text
+1 到 +5 天的正滞后搜索带整体通过 AR(1) surrogate max-stat 检验。
```

不是逐个 lag 单独判断，也不是包括 lag=0。

---

## 12.7 same-day-only 与 same-day-dominant 的当前处理

用户已裁决：

```text
same-day-only 与 same-day-dominant 暂时不作为当前主要 lead–lag 结果整理对象；
但它们不是失去科研价值；
后续可以作为同期耦合、快速响应、反馈闭合、共同阶段重组或 transition risk 的对象单独审理。
```

因此当前窗口主整理优先对象为：

```text
Tier1a
Tier1b
Tier2
```

暂不展开但必须保留的对象：

```text
lead_lag_no_same_day_only
lead_lag_ambiguous_same_day_dominant
lead_lag_ambiguous_coupled_or_feedback_like
lead_lag_ambiguous_bidirectional_close
```

禁止写法：

```text
same-day-only 没有价值；
same-day-dominant 可以删除；
lag0 主导就是无关系；
只保留 clear lead-lag，丢弃同期耦合。
```

正确写法：

```text
same-day-only / same-day-dominant 暂不进入当前 strict lead–lag 主整理；
但作为同期耦合与反馈/共同重组证据保留，后续可单独审计。
```

---

## 12.8 当前后续工作边界

当前可继续推进：

```text
1. 基于 5 日版本，逐窗口整理 Tier1 / Tier2 temporal candidates；
2. 每个窗口必须同时标注 Tier1a / Tier1b / Tier2；
3. 每条关系必须带 positive_peak_lag、peak r、lag0 r、same_day flag；
4. V→P、H→P、H/V internal、P feedback 候选分别整理；
5. 9 日版本作为 sensitivity 对照，而不是主底座。
```

当前不要做：

```text
1. 不要把 lead_lag_yes 写成 pathway established；
2. 不要直接进入完整 pathway 层；
3. 不要把 same-day-only / same-day-dominant 删除；
4. 不要把 S1 解释成纯单向传播窗口；
5. 不要忽略 audit null 与 high-persistence 风险标记。
```

---

## 12.9 当前接管摘要更新

当前项目底座更新为：

```text
foundation/V1:
    baseline_a = 原 9 日滑动基础层，继续保留
    baseline_smooth5_a = 新 5 日滑动基础层，作为 lead–lag 主线输入候选

lead_lag_screen/V1:
    lead_lag_screen_v1_a = 9 日版本，可作对照
    lead_lag_screen_v1_smooth5_a = 5 日版本，后续 lead–lag 主线优先使用

pathway:
    旧 V1/V2 不继承；
    后续 pathway 重建读取 lead_lag_screen 的 Tier1/Tier2 时间资格结果；
    same-day / ambiguous 结果进入 expanded risk pool，不进入当前 strict 主整理。
```

一句话：

```text
后续主要在 5 日滑动 anomaly index 的 lead–lag 结果上推进；
当前优先整理 Tier1/Tier2；
same-day-only 和 same-day-dominant 暂不展开，但作为有价值的同期/反馈/重组风险对象保留。
```

---

# 13. 5日 smoothed index 有效性诊断结论

**更新日期**：2026-04-27  
**更新性质**：index_validity/V1 结果审理与指数层结论固化。  
**对应结果层**：

```text
index_validity/V1/outputs/smooth5_v1_a
foundation/V1/outputs/baseline_smooth5_a
```

本节用于记录 5 日 smoothed index 的指数有效性诊断结论，防止后续把“指数计算错误”“指数定义是否合适”“指数对下游结果的影响”混为一谈。

---

## 13.1 当前诊断层的范围

本轮 `index_validity_smooth5_v1_a` 只回答两个问题：

```text
1. 5日 smoothed index 在单年份上是否出现异常偏移或乱飞；
2. index 高低值是否确实对应其代表的底层物理场结构。
```

本轮不回答：

```text
1. lead-lag 是否成立；
2. pathway 是否成立；
3. 自相关是否影响下游检验；
4. 分窗口稳定性如何；
5. raw/anomaly lead-lag 是否一致。
```

因此，本节结论不得被直接升级为 pathway 结果，也不得被用来替代 lead-lag 统计审计。

---

## 13.2 工程运行状态

`index_validity/V1` 已完整运行成功。

已确认输出包括：

```text
summary/summary.json
summary/run_meta.json
tables/input_file_audit.csv
tables/yearwise_index_shape_audit.csv
tables/anomaly_reconstruction_basic_check.csv
tables/anomaly_daily_mean_basic_check.csv
tables/physical_composite_sample_info.csv
tables/physical_representativeness_summary.csv
figures/yearwise/*.png
figures/physical/*.png
```

`summary.json` 显示：

```text
status = success
n_indices = 20
n_years = 45
n_days = 183
n_yearwise_figures = 60
n_physical_figures = 36
cartopy_status = available
anomaly_basic_status = pass
```

结论：

```text
工程链条已跑通；
图表输出完整；
cartopy 可用；
不是半运行或输出缺层结果。
```

---

## 13.3 anomaly 基础检查结论

anomaly 只做最低限度检查，结果通过。

已确认：

```text
index_anomalies = index_values_smoothed - index_daily_climatology
```

在数值精度范围内成立。

同时：

```text
按 day 对所有年份求 anomaly 均值后，结果接近 0。
```

结论：

```text
anomaly 派生关系没有暴露问题；
不需要把 anomaly 作为主要怀疑对象；
后续如果出现问题，应优先检查指数定义与物理代表性，而不是 anomaly 生成步骤。
```

---

## 13.4 V_NS_diff 的实现结论

已查工程实现。

文件位置：

```text
foundation/V1/src/foundation_v1/object_index_builder.py
```

当前实现逻辑为：

```text
north = Σ V(15–30N)
south = Σ V(10–15N)
V_NS_diff = north - south
```

因此：

```text
V_NS_diff = north(15–30N) - south(10–15N)
```

当前物理 composite 中 `V_NS_diff` 高值对应北侧 v850 更强、南侧相对弱，与工程实现一致。

结论：

```text
V_NS_diff 在计算上没有发现问题；
若后续认为该变量不合适，问题属于指数定义/命名/物理解释方向，而不是代码实现错误。
```

使用时必须明确写作：

```text
V_NS_diff 表示 north-minus-south，而不是 south-minus-north。
```

---

## 13.5 5日 smoothed index 是否存在计算层错误

当前结论：

```text
没有证据表明 5日 smoothed index 存在整体计算错误；
没有证据表明指数整体乱飞或全局性崩坏；
主要物理 composite 与指数定义总体一致。
```

尤其：

```text
P 类指数物理指示性清楚；
V_strength 与 V_pos_centroid_lat 物理指示性基本清楚；
Je/Jw axis_lat 指示急流轴位置的结构清楚；
H 类有可解释空间结构。
```

需要注意：

```text
H_west_extent_lon、H_zonal_width 等 extent/width 类指数，
仅凭 high/low/diff map 还不如 P profile 或 jet axis profile 那样直观；
后续若重点使用，建议增加边界/contour/阈值区域叠加图。
```

---

## 13.6 单年份偏移是否能够解释

当前结论：

```text
单年份偏差总体可以解释；
它更像年际异常、年际型态差异或物理对象位置/边界不稳定；
不是指数计算错误的直接证据。
```

需要明确：

```text
yearwise_index_shape_audit.csv 中的 flag 是相对排名标记；
每个指数按年份分布取前 10% 标记；
20 个指数 × 约 5 个年份 ≈ 100 个 flag 是规则自然结果；
不能把 flag 数量直接解释为“100 个异常错误”。
```

`offset` 偏高的含义：

```text
某一年相对多年日均参考曲线整体偏高/偏低；
这通常可解释为该年物理对象整体偏强、偏弱、偏北、偏南、偏宽或偏窄。
```

`roughness / curvature` 偏高的含义：

```text
某一年日序列跳动或折返较强；
这不一定是错误；
位置、宽度、extent、axis、centroid 类指数天然更容易受物理对象分裂、弱化、跳区、阈值边界变化影响。
```

因此：

```text
单年偏差可作为人工查看年份的索引；
不能直接作为指数错误判据。
```

---

## 13.7 当前最可能的风险类型

如果某些指数后续出现问题，优先归入以下类型：

```text
1. 指数定义是否适合；
2. 指数是否过于敏感；
3. 位置/宽度/边界类指数是否容易跳区；
4. 某些指数是否需要更稳健的物理表达；
5. 命名与符号方向是否需要更清楚。
```

而不是优先归入：

```text
1. 5日 smoothed index 计算错误；
2. anomaly 生成错误；
3. 工程链条未接通；
4. 整体指数乱飞。
```

---

## 13.8 当前指数使用建议

初步分层如下。

### 可较放心继续使用

```text
P 类大部分指数：
    P_main_band_share
    P_south_band_share_18_24
    P_main_minus_south
    P_spread_lat
    P_north_band_share_35_45
    P_north_minus_main_35_45
    P_total_centroid_lat_10_50

V_strength
V_pos_centroid_lat
Je_axis_lat
Jw_axis_lat
```

### 可使用，但解释时需注意图件与定义敏感性

```text
H_strength
H_centroid_lat
Je_strength
Jw_strength
Je_meridional_width
Jw_meridional_width
```

### 计算无问题，但需明确命名/定义方向

```text
V_NS_diff = north - south
```

### 若后续重点使用，建议增强图件确认

```text
H_west_extent_lon
H_zonal_width
```

原因：

```text
这两个属于边界/范围类指数；
仅 high/low/diff map 可看到空间结构，但不如直接叠加阈值边界、west extent 线或 width 标记直观。
```

---

## 13.9 最终裁决

当前可固化的结论：

```text
5日 smoothed index 在计算层没有发现问题；
anomaly 派生关系没有问题；
V_NS_diff 的实现是 north-minus-south，与 composite 表现一致；
单年偏差总体可解释，不构成指数算错证据；
若后续有疑点，应主要归为指数定义是否合适、是否敏感、是否需要更稳健表达，而不是工程计算错误。
```

一句话：

```text
当前 5日指数层可以作为后续 lead-lag 主线的输入底座；
但对位置、宽度、边界类指数，应在解释时保留“定义敏感性”标记。
```

---

## 13.10 后续禁止混淆

后续不得写成：

```text
yearwise flag 多，所以指数坏了；
V_NS_diff 图像与 south-minus-north 不一致，所以代码错了；
anomaly 可能出问题，所以 lead-lag 不可信；
H_west_extent_lon 图不够直观，所以指数一定错误。
```

正确写法：

```text
yearwise flag 是相对排名型人工查看线索；
V_NS_diff 当前定义就是 north-minus-south；
anomaly 基础检查通过；
H_west_extent_lon / H_zonal_width 需要更强的边界叠加图来确认代表性。
```

---

# 14. 迁移包查漏补缺审计增补记录

**更新日期**：2026-04-27  
**更新性质**：对本会话迁移包的查漏补缺审计结果固化。  
**目的**：防止新会话在接管时把工作层结果误写成最终结论，把派生结构误写成直接输出，把未实现内容误写成已完成工作。

本节不是重新总结项目，而是补充那些在迁移包中可能被压缩、弱化或容易失真的关键状态信息。

---

## 14.1 5日 lead–lag 主线形成过程必须保留

5 日 lead–lag 主线不是任意更换平滑窗口，而是由 9 日版本的 high-persistence 审计推动形成。

演化链条如下：

```text
1. 9日版本 lead–lag 已完整跑通；
2. 结果显示 AR(1) phi 大量触顶 0.95；
3. 用户质疑 surrogate 运行速度过快、是否真正做了 null；
4. 审计后确认 surrogate 确实生成 AR(1) null，但主 null 是 pooled window-variable AR(1)，不是 year-specific AR(1)；
5. 补充 yearwise-scale audit null；
6. 补充 raw_phi_before_clip / phi clipping 审计；
7. 9日版本确认存在真实 high-persistence 风险；
8. 改跑 5日 smoothed anomaly index；
9. 5日版本 main lead_lag_yes 数量没有崩，Tier1/Tier2 更稳，Tier3 明显减少，phi clipping 大幅下降；
10. 因此裁决：5日作为后续 lead–lag temporal eligibility 主底座，9日保留为 sensitivity 对照。
```

正确理解：

```text
5日不是因为“9日错误”而替代9日；
而是因为5日在当前 lead–lag 时间资格层中统计风险更低、审计稳定性更好。
```

禁止误读：

```text
9日全部无效；
5日是唯一绝对正确平滑；
平滑选择只是用户偏好。
```

---

## 14.2 surrogate null 的能力边界必须保留

当前 lead–lag 使用的主 null 是：

```text
pooled_window_variable_ar1
```

含义：

```text
每个 window × variable 跨年份 pooled 估计一组 AR(1) persistence 参数；
每一年独立生成 synthetic series；
保留原始 NaN mask；
不跨年拼接；
不做 year-specific phi。
```

当前 audit null 是：

```text
pooled_phi_yearwise_scale_ar1
```

含义：

```text
phi 仍使用 pooled phi；
每个 year × variable 使用自己的 mean/std scale；
只用于检查 yearwise amplitude 差异对 null 的影响；
不改变主 lead_lag_label。
```

必须保留的限制：

```text
AR(1) surrogate null 只能控制单变量 persistence 与 lag 搜索造成的随机相关峰；
它不能排除共同驱动；
不能排除多变量协同；
不能排除共同季节内背景；
不能排除真实物理耦合；
不能提供因果证明。
```

因此：

```text
lead_lag_yes = 时间资格通过
lead_lag_yes ≠ 因果
lead_lag_yes ≠ pathway established
```

---

## 14.3 positive_peak_lag 定义必须醒目标注

`positive_peak_lag` 的定义是：

```text
在 +1 到 +5 天正滞后搜索带内，绝对相关最大的 lag。
```

它不是：

```text
-10 到 +10 全 diagnostic lag 范围的最大峰；
lag0；
唯一显著 lag；
精确传播时间。
```

lag0 的处理规则：

```text
lag=0 不参与 positive_peak_lag；
lag=0 单独进入 same-day coupling 诊断。
```

对于宽峰结果，正确写法是：

```text
正滞后支持带在 +5 天附近较强；
或 +4/+5 附近达峰；
不能写成严格 5 天传播。
```

这一点尤其适用于 S1 中曾被单独检查的两个 unexpected nonP internal / upstream-like 结果：

```text
V_strength → Jw_strength
V_pos_centroid_lat → Je_axis_lat
```

---

## 14.4 S1 unexpected nonP internal 结果必须保留但不能过度解释

S1 中用户特别关注过两个“猎奇但不像假的”结果：

```text
V_strength → Jw_strength
    positive_peak_lag = +5
    positive_peak_r ≈ 0.227
    Tier1a

V_pos_centroid_lat → Je_axis_lat
    positive_peak_lag = +5
    positive_peak_r ≈ 0.241
    Tier1a
```

已检查完整 lag 曲线：

```text
不是 +5 单点孤立假峰；
正滞后侧从 +2/+3 开始增强；
在 +4/+5 或 +5/+6 附近形成宽峰；
lag0 不强；
负滞后侧弱。
```

当前定性：

```text
统计上存在较干净的正滞后时间资格信号；
但只能作为 unexpected nonP internal / upstream-like candidate；
不能讲成 V 驱动 Jw 或 V 驱动 Je；
不能讲成物理传导路径。
```

后续若整理，应单独归入：

```text
unexpected nonP internal candidates
```

---

## 14.5 same-day-only / same-day-dominant 不能被删除

用户已明确裁决：

```text
same-day-only 与 same-day-dominant 暂不作为当前主要 lead–lag 结果整理对象；
但它们不是失去科研价值。
```

必须保留为：

```text
同期耦合证据
快速响应证据
反馈闭合线索
共同阶段重组线索
transition risk 对象
bidirectional / coupled structure 候选
```

禁止写法：

```text
same-day-only 没有价值；
same-day-dominant 可以删；
lag0 主导就是无关系；
只保留 clear lead-lag。
```

正确写法：

```text
same-day-only / same-day-dominant 当前不进入 Tier1/Tier2 主整理；
但作为同期耦合、反馈和重组风险对象保留，后续可单独审计。
```

---

## 14.6 index_validity 的证据层级必须区分

`index_validity/V1` 的代码直接输出包括：

```text
yearwise_index_shape_audit.csv
anomaly_reconstruction_basic_check.csv
anomaly_daily_mean_basic_check.csv
physical_composite_sample_info.csv
physical_representativeness_summary.csv
yearwise figures
physical composite figures
summary.json
run_meta.json
```

其中：

```text
physical_representativeness_summary.csv
中的 direction_consistency / structure_clarity / contrast_strength / overall_grade
仍为 to_review。
```

因此，本会话中关于：

```text
P 类指数物理指示性清楚；
V_strength / V_pos_centroid_lat 基本合理；
Je/Jw axis_lat 指示性清楚；
V_NS_diff 计算无问题；
单年偏差总体可解释。
```

这些属于：

```text
人工审图后的诊断判断
```

而不是：

```text
代码自动输出的 final grade
```

后续若需要正式表格，必须人工填写或另写 grading 层，不能把当前口头审理误写成代码直接结果。

---

## 14.7 V_NS_diff 的裁决必须保留

已查工程实现：

```text
foundation/V1/src/foundation_v1/object_index_builder.py
```

当前实现为：

```text
north = ΣV(15–30N)
south = ΣV(10–15N)
V_NS_diff = north - south
```

即：

```text
V_NS_diff = north-minus-south
```

当前 composite 表现：

```text
V_NS_diff 高值对应北侧 v850 更强、南侧相对弱。
```

这与工程实现一致。

裁决：

```text
V_NS_diff 在计算层没有问题；
若后续认为该变量不合适，问题属于指数定义、命名或物理解释方向；
不是代码错误。
```

后续所有解释必须写清：

```text
V_NS_diff = north - south
```

不得默认理解为 south - north。

---

## 14.8 index_validity 的输入与 cartopy 状态补充

当前 `index_validity/V1` 读取的是：

```text
foundation/V1/outputs/baseline_smooth5_a/preprocess/smoothed_fields.npz
```

而不是单独读取：

```text
precip_smoothed.npy
u200_smoothed.npy
z500_smoothed.npy
v850_smoothed.npy
```

该 bundle 需要包含：

```text
precip_smoothed
u200_smoothed
z500_smoothed
v850_smoothed
lat
lon
years
```

当前运行状态：

```text
cartopy_status = available
```

代码行为：

```text
若 cartopy 不可用，当前实现会 fallback 到普通 matplotlib map，并在 summary 中记录状态；
不是静默跳过。
```

如后续需要严格 cartopy，可另改为 fail-fast，但这尚未实现。

---

## 14.9 尚未实现内容不能写成已有结果

以下内容尚未实现或尚未定稿：

```text
1. same-day-only / same-day-dominant 专门审计层；
2. H_west_extent_lon / H_zonal_width 的阈值边界、west extent line、zonal width 叠加图；
3. 完整 pathway reconstruction；
4. physical_representativeness_summary 的人工 grading 写回；
5. 全部窗口的逐条 Tier1/Tier2 结果整理；
6. unexpected nonP internal candidates 的系统审计；
7. 后续 pathway 输入池最终规则。
```

禁止写成：

```text
已完成 same-day 审计；
已完成 pathway 重建；
H extent/width 已最终确认；
physical_representativeness 已自动 grading；
Tier1/Tier2 已经是路径边。
```

---

## 14.10 后续逐窗口整理的最低格式

后续整理 T1、S2、T2、S3、T3、S4、T4、S5 时，不能只给 family-level 汇总。

用户已要求 S1 逐条列出 Tier1/Tier2，并带上滞后。因此后续每个窗口至少保留字段：

```text
window
source
target
family direction
positive_peak_lag
positive_peak_r
lag0_r
evidence_tier
same_day flag
lead_lag_label
简短解释边界
```

尤其要区分：

```text
Tier1a = clear lead-lag
Tier1b = lead-lag with same-day coupling
Tier2 = audit-moderate candidate
```

不得把 Tier1a 和 Tier1b 混成一个“稳定 lead-lag”。

---

## 14.11 T3 与其他敏感窗口需要保留风险标记

5 日版本整体比 9 日更稳，但这不代表每个窗口都同样稳。

已知风险：

```text
T3 在 5日版本中曾显示 main yes 与 audit-stable 之间存在明显不稳风险；
后续逐窗口整理 T3 时必须谨慎；
不能用全季 V→P 稳定性覆盖 T3 的局部不稳定。
```

若继续使用该判断，需要回到结果表确认具体数字，不能凭记忆扩展。

---

## 14.12 H_west_extent_lon / H_zonal_width 的图件限制

当前 index_validity 中：

```text
H_west_extent_lon
H_zonal_width
```

已有 high / low / diff map。

但这类指数本质是：

```text
边界 / extent / width 类指数
```

仅看 high-low-diff map 不如直接叠加：

```text
阈值区域边界
west extent line
zonal width reference
```

因此当前裁决是：

```text
计算层没有发现错误；
但若后续将其作为关键路径变量或关键物理解释对象，应先补增强图件；
否则只能说“有可解释空间结构”，不能说“代表性已最终确认”。
```

---

## 14.13 用户约束补充：不得为了推进而简化

用户明确要求：

```text
不能为了实现目标而简化、忽视内容；
不能先做工程上足够的版本，后面再做论文要求；
不能因为困难就回避；
不能重复提出已裁决问题；
提出阻挡项时必须给明确建议。
```

已裁决内容包括：

```text
1. 后续 lead–lag 主线用 5日版本；
2. V_NS_diff 计算层无问题，定义为 north-south；
3. same-day-only / same-day-dominant 暂不主整理但保留价值；
4. old pathway V1/V2 不继承；
5. index_validity 不做下游影响诊断。
```

新会话不得反复把这些当成未裁决问题提出。

---

## 14.14 最小接管提醒

若新会话继续推进，必须先说明其依据的是：

```text
5日 lead_lag_screen_v1_smooth5_a 已运行结果；
5日 index_validity_smooth5_v1_a 已运行结果；
Tier1/Tier2 仅作为 temporal eligibility candidates；
same-day-only/dominant 保留但暂不主整理；
V_NS_diff = north - south；
old pathway V1/V2 不继承。
```

任何后续解释都必须避免：

```text
把 lead-lag 写成路径；
把 Tier 写成因果；
把人工审图写成代码自动 final grade；
把未实现图件或审计写成已有结果。
```

