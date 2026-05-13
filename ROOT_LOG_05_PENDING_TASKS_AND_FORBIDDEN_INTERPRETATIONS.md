# ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS.md`

## 0. 本日志的性质

这是当前项目的**未完成任务、待核对项与禁止误读日志**。

后续推进前必须先看本日志，防止把未实现内容写成已有结果，或把方法层输出写成物理结论。

---

## 1. 当前明确未实现的任务

### 1.1 peak_family_atlas_v10_5_f

状态：已讨论但未实现。

目标：

- 把所有对象候选峰族、rank、support、lineage、角色整理成表；
- 区分 strict accepted、non-strict candidate、object-native family、energy-dominant family、lineage-assigned family、secondary-supported family。

不能写成：

- 已有峰族图谱；
- 已完成所有候选峰身份整理。

---

### 1.2 peak morphology audit

状态：已讨论但未实现。

目标：

- 量化峰宽；
- 峰间谷深；
- 半高宽；
- 峰间分离度；
- 清晰峰 / 宽峰 / 多峰簇；
- 支撑 “V clean / Je broad / H intermediate” 这类图像层观察。

不能写成：

- V 峰形清楚已经统计确认；
- Je 宽峰已经量化；
- H intermediate 已经落表。

当前允许表述：

> 图像层初步观察显示 V 峰形较清楚、Je 更宽、H 居中；该判断尚未通过 peak morphology audit 量化。

---

### 1.3 cartopy spatial validation

状态：已讨论但未实现。

目标：

- 画原始空间场差异图；
- 做 field/profile evolution maps；
- 对 W045、W113、W160 等窗口做空间场验证；
- 尤其关注 S3–T3–S4、T3/T4 V→P 弱化的空间解释。

不能写成：

- 已有空间场物理证明；
- V10.5 已经提供 cartopy 空间验证；
- V→P 支持区域转移已被空间图证明。

要求：

- 地图默认使用 cartopy；
- 不能只输出 npz 而不画可解释图；
- 必须相对适当基线比较，不能无条件叙事。

---

### 1.4 yearwise object-order support

状态：已讨论但未实现。

目标：

- 检查 object-to-object order 是否具有年际稳定性；
- 不是只看 pooled / aggregated order；
- 需要区分 near-tie 与 true order reversal。

不能写成：

- V10.4 order 已经具有完整 yearwise support；
- V10.5_d profile-energy family stability 等于 object-order yearwise validation。

---

### 1.5 object physical metrics validation

状态：已讨论但未实现。

目标：

- 用更物理化对象指标替代粗 low-dimensional profile metrics；
- 例如 H_strength、H_centroid_lat、H_west_extent_lon、Jw_axis_lat、Jw_strength、Je_axis_lat、V_strength、V_NS_diff、P band share、P centroid、P spread 等。

不能写成：

- 当前 low-dimensional metrics 已经足够；
- 当前 V10.5 已完成正式物理指标验证。

---

### 1.6 V10 与 lead-lag/pathway 重接

状态：已讨论但未实现。

目标：

- 对齐 V10 smooth-field/profile peak 与 lead-lag anomaly index；
- 重新连接 V10 多峰族 / object order 与 path diagnosis 变量、窗口、路径；
- 明确 V10 分支在整个 pathway 项目中的位置。

不能写成：

- V10 已替代 lead-lag；
- V10 已替代 path diagnosis；
- V10 peak/order 已经解释 pathway。

---

## 2. 当前待核对项

1. V10.5_d / V10.5_e 具体输出表是否与迁移包描述完全一致；
2. V10 最新解释层修正是否已经写入旧 root log；
3. V7/V10 accepted windows 的继承关系是否需要进一步文件级封口；
4. `lead_lag_screen/V4` 的科研角色；
5. `index_validity` 与当前 V10 / lead-lag 主线的关系；
6. root 旧 logs / manifests / patch readmes 中是否还有未吸收的关键失败记录；
7. README 中个别 Windows 路径显示乱码/断行是否只是 GitHub 渲染问题；
8. 本地数据文件与 GitHub 可见结构是否一致。

---

## 3. 禁止误读清单

### 3.1 禁止把 V10.5 写成推翻 V10.1–V10.4

错误表述：

> V10.5 证明 V10.4 错了。

正确表述：

> V10.5 没有推翻 V10.1–V10.4 的方法层骨架；它修正了原先过度简化的“单峰、干净、强先行”解释。

---

### 3.2 禁止把 profile-energy top1 当作更高真值

错误表述：

> profile-energy top1 更强，所以应该替代 assigned peak。

正确表述：

> profile-energy top1 是 energy-dominant family；它可能不同于 lineage-assigned family。top1 mismatch 不等于 V10.4 assigned peak 不支持。

---

### 3.3 禁止把 non-strict candidate 当作噪声

错误表述：

> non-strict 没过，所以可以忽略。

正确表述：

> non-strict candidates 是真实候选峰族，但未晋升为 strict accepted main windows。它们应保留并解释其 joint score、bootstrap、profile-energy、window context。

---

### 3.4 禁止把 strict window 写成干净单峰转换

错误表述：

> strict accepted window 是干净单峰转换。

正确表述：

> strict accepted 是 joint 主骨架，但窗口内部可能存在 early fluctuation、宽峰、弱先行、non-strict competitor、多峰簇。

---

### 3.5 禁止把 object order 写成因果链

错误表述：

> H 先于 Jw，所以 H 导致 Jw。

正确表述：

> V10.4 的 object order 是 timing/order sensitivity audit，不是因果推断。order 只能作为方法层 timing skeleton。

---

### 3.6 禁止把 V10.4 assigned peak 写成对象唯一主峰

错误表述：

> V10.4 assigned peak 是该对象在全季的主峰。

正确表述：

> V10.4 assigned peak 是 window-context lineage assignment。对象全季可能还有 object-native energy-dominant peaks 或 secondary-supported peaks。

---

### 3.7 禁止把 V10.3 order inversion audit 写成对象间次序审计

错误表述：

> V10.3 已经证明对象间次序稳定。

正确表述：

> V10.3 确认同一 scope 内候选序列没有 inversion；V10.4 才是对象间 order sensitivity 审计。

---

### 3.8 禁止把图像层观察写成统计事实

错误表述：

> V clean、Je broad、H intermediate 已经统计确认。

正确表述：

> 这些目前是图像层观察和初步核查，尚未通过 peak morphology audit 量化。

---

### 3.9 禁止把 low-dimensional metrics 支持弱解释为 profile-energy 无效

错误表述：

> 低维指标弱，所以 profile-energy 不可信。

正确表述：

> 当前低维指标偏粗，不能充分承载 profile-level 形态变化；这是指标抽象层负结果，不直接否定 profile-energy。

---

### 3.10 禁止把工程存在写成科研成立

错误表述：

> 目录/文件存在，所以结果成立。

正确表述：

> 工程存在只能说明有实现承载；是否进入执行链、是否运行、结果是否可用、解释是否成立，需要分别核对。

---

## 4. 后续推进优先级建议

### 第一优先级：peak_family_atlas

原因：

- 当前所有解释都需要一个正式峰族图谱；
- 它能把 strict、non-strict、object-native、energy-dominant、lineage-assigned、secondary-supported 统一落表；
- 是避免误读 V10.5 的最小增强。

### 第二优先级：peak morphology audit

原因：

- 当前 “V clean / Je broad / H intermediate” 仍是图像层判断；
- 需要用峰宽、谷深、半高宽、分离度量化；
- 它能把多峰族解释从观察推进到方法层证据。

### 第三优先级：V10 与 lead-lag/pathway 重接设计

原因：

- V10 不能替代整个 pathway 项目；
- 需要对齐 smooth-field/profile peak 与 anomaly index lead-lag；
- 需要重新连接关系路径骨架。

### 第四优先级：cartopy spatial validation

原因：

- 物理解释最终需要空间场证据；
- 尤其 T3/T4 V→P 弱化需要空间机制核查。

---

## 5. 后续新会话必须遵守的最低协议

1. 先判断用户说法是否成立，再分析原因，再执行；
2. 先核查图、表、代码、输出，再解释；
3. 不得把派生结构当直接输出；
4. 不得把未落实实现当已有结果；
5. 不得省略失败记录、负结果、不可信结果；
6. 不得把工程存在当科研成立；
7. 不得把 V10 当前分支当作整个 EASM pathway 项目；
8. 若出现会话叙述与工程现实冲突，标记为“需核对”，不能自行抹平。
