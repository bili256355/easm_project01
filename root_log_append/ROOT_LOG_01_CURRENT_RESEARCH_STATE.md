# ROOT_LOG_01_CURRENT_RESEARCH_STATE

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_01_CURRENT_RESEARCH_STATE.md`

## 0. 本日志的性质

这是当前项目的**科研状态总控日志**。

它记录当前研究主线、方法层结果、解释边界、未解决问题和最容易误读的地方。它不是输出表，也不是物理机制结论。

---

## 1. 当前核心研究主题

当前研究仍围绕：

> 东亚夏季风 4–9 月季节推进过程中，降水纬向/南北结构与 Jw / Je / H / V / P 等对象之间的关系骨架如何配置、竞争、接替与重组；这种表层关系配置及其背后的上游影响路径是否随季节推进形成有组织的阶段结构。

核心对象：

- **P**：降水结构；
- **V**：850 hPa 风 / 低层环流；
- **H**：500 hPa 高度场 / 副高相关对象；
- **Je**：东侧 200 hPa 急流对象；
- **Jw**：西侧 200 hPa 急流对象。

当前研究重点不是“寻找唯一峰值日”，而是：

> 在多峰族结构下，识别 strict joint main windows、non-strict candidates、object-native peaks、energy-dominant families、lineage-assigned families、secondary-supported peaks、weak/strong precursors、broad transitions、near-tie groups，并判断它们如何组织成对象间时序骨架。

---

## 2. 当前主线位置

当前 V10 分支是：

> **peak / order / multipeak 方法层诊断分支**。

它不是整个 EASM pathway 项目的全部，也不能替代早期 lead-lag、path diagnosis、PCMCI/EOF/CCA 对照、T3/T4 空间解释等继承线。

当前应保留的更大项目背景包括：

- stage_partition V2 / V7 的阶段窗口继承；
- lead-lag V1 anomaly index 路线；
- AR(1) surrogate、bootstrap 稳健性、显著性驱动强度判断；
- T3/T4 V→P 关系弱化与 surrogate null 过滤；
- path diagnosis v1_6_20–v1_6_27 历史线；
- P 的多角色：终点、反馈源、mediator、nonP internal 关系。

---

## 3. 当前相对可依赖的方法层结果

### 3.1 Joint main-window lineage

当前主骨架：

- joint candidates：18、45、81、96、113、132、135、160；
- strict accepted main windows：45、81、113、160；
- non-strict candidates：18、96、132、135。

解释边界：

- strict accepted windows 可以作为当前主骨架；
- non-strict candidates 必须保留为真实候选，不是污染；
- strict accepted 不等于干净单峰转换；
- non-strict 不等于噪声或应删除对象。

### 3.2 Object-native peak structure

当前确认对象自身不是单峰结构，而是多候选峰族。

已接收的重要例子：

- H：19、35、57、77、95、115、129、155；
- Jw：41、83、104、130、152；
- P/V/Je 也具有多峰候选结构。

解释边界：

- object-native peak 不等于 joint strict window peak；
- object-native energy-dominant peak 不等于 lineage-assigned peak；
- window-context assigned peak 不得写成对象唯一主峰。

### 3.3 Sensitivity 结果

当前方法层判断：

- peak day 的主要敏感源：input smoothing 与 detector_width；
- match radius 主要影响 support class；
- band / merge 主要影响 derived window；
- detector penalty 通常不改 peak day；
- local peak distance 影响较小。

解释边界：

- 当前方法链不是“到处都敏感”；
- 敏感性集中在输入平滑与检测时间尺度；
- smooth5 vs smooth9 是输入资料差异，不是普通算法小参数。

### 3.4 Object-to-object order

当前判断：

- V10.4 才是真正的对象间 order sensitivity；
- tau=2 下没有 true order reversal；
- 主要不确定来自 near-tie / grouping，而不是对象间方向整体颠倒。

解释边界：

- 对象间 timing skeleton 可以作为方法层骨架；
- 不能把 near-tie 内部硬排序；
- 不能把 object order 写成因果链。

---

## 4. V10.5 对解释层的最新修正

V10.5 没有推翻 V10.1–V10.4 的方法层结果，也不足以形成高度怀疑。

但 V10.5 足以推翻以下简化解释：

- 单峰式窗口解释；
- 干净转换解释；
- 强先行默认解释；
- profile-energy top1 更真解释；
- assigned peak 是对象唯一主峰解释。

当前更安全的总表述：

> V10.4 给出的是 window-context timing skeleton；V10.5 显示该 skeleton 嵌在多峰族强度景观中。当前应把原主线从“单峰式窗口次序”升级为“多峰族结构下的 window-context timing skeleton”。

---

## 5. V10.5 当前具体认识

### 5.1 assigned peak 与 energy-dominant peak 可以分离

已接收的典型 competition cases：

- W045-H：energy top1 day19/20，assigned day35；
- W113-P：energy top1 day88/97，assigned day113；
- W160-Jw：energy top1 day130/135，assigned day152。

解释边界：

- energy top1 不是更高真值；
- top1 mismatch 不等于 assigned peak 不支持；
- top-k support 更适合表达 candidate-family competition。

### 5.2 non-strict peak 未晋升有方法层原因

当前理解：

- non-strict candidates 在 joint detector 中多数是 lower-score + lower-bootstrap；
- 它们不是“joint detector 高分但只差 bootstrap”的简单情况；
- day96 是中高分但未晋升；
- day18、132、135 在 joint_all score 中不高。

解释边界：

- 单对象 profile-energy 强，不能直接推翻 joint strict accepted 结果；
- non-strict candidates 应保留为候选峰族，而不是升格或删除。

### 5.3 profile-energy 子方法自身也会强但不稳

当前理解：

- profile-energy top1 可能强，但年重抽样稳定性不足；
- lineage-assigned peak 可能不是 top1，但 top-k support 更稳定；
- strength 与 stability 必须分开看。

---

## 6. 当前不能说死的解释

禁止写成已确认结论：

- H 导致 Jw；
- Jw 导致 P；
- H 强烈领先 W045；
- Jw day152 是 W160 最强 Jw 峰；
- W045 是干净单峰转换；
- V10.4 assigned peak 是对象唯一主峰；
- profile-energy top1 更强，所以原结果错；
- object order 等于因果链；
- strict window 等于物理机制完成。

---

## 7. 当前图像层判断的状态

当前有图像层核查后的候选判断：

- V 峰形较清楚；
- Je 更像宽峰或长转变背景；
- H 介于 V 与 Je 之间；
- W113-Jw 更像强先行；
- W045-H / W160-Jw 更像 secondary 或 weak assigned peak。

状态：

> 这些仍是图像层观察 + 初步核查，不是峰形态量化结论。后续必须通过 peak morphology audit，用峰宽、谷深、半高宽、峰间分离度等指标落表。

---

## 8. 当前未解决科学问题

- 多峰族分别对应什么物理过程？
- weak precursor 是否具有物理意义？
- energy-dominant non-strict family 与 strict window 是否构成前置—主转换关系？
- broad transition 是否代表持续调整而不是单次转换？
- near-tie group 中对象是否可进一步物理区分？
- 低维物理指标为何暂不支持 profile-energy 峰？
- V10 peak/order 结果如何与 lead-lag/path diagnosis 主线重接？

---

## 9. 当前结论等级

### 方法层较稳

- V10.1–V10.4 方法层骨架；
- strict accepted 45/81/113/160；
- non-strict 18/96/132/135；
- 多对象峰族结构；
- V10.4 tau=2 下无 true order reversal；
- V10.5 暴露多峰族和 strength–stability 分离。

### 解释层需限制

- weak precursor；
- broad transition；
- clean/broad peak morphology；
- energy-dominant vs lineage-assigned 的物理意义。

### 尚不能作为结论

- 因果链；
- 物理机制；
- 跨资料验证；
- yearwise object-order support；
- cartopy 空间场解释；
- 正式对象物理指标验证。
