# ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN.md`

## 0. 本日志的性质

这是当前项目的**版本状态与执行链日志**。

核心目的：防止把“讨论过”误写成“已实现”，把“文件存在”误写成“已进入执行链”，把“已运行”误写成“科研结论成立”。

状态分类固定为：

- 已讨论但未实现；
- 已实现但未确认进入执行链；
- 已进入执行链但覆盖范围不明；
- 已运行但结果不完整/不可信；
- 已运行且结果可用；
- 已废弃；
- 历史参考，不作为当前依据；
- 工程存在，科研角色待核对。

---

## 1. 总体版本状态表

| 分支 | 版本/模块 | 工程承载 | 研究角色 | 当前状态 | 可依赖程度 | 不能误读 |
|---|---|---|---|---|---|---|
| foundation | V1 | `foundation/V1` | 预处理与 smoothed fields 输入底座 | 工程存在，未完成本地输入核查 | 可作为输入底座线索 | 不能确认本地数据完整存在 |
| index_validity | smooth5_v1_a 等 | `index_validity` + root manifests | 指数有效性/平滑输入相关支线 | 工程存在，科研角色待核对 | 暂作历史/支线参考 | 不能并入 V10 主线结论 |
| lead_lag_screen | V1 | `lead_lag_screen/V1` | lead-lag anomaly index 基线 | 历史继承底座 | 研究偏好有效，输出未本轮复核 | 不能替代 V10 peak/order |
| lead_lag_screen | V1_1 | `lead_lag_screen/V1_1` | T3/T4、窗口长度/指数扩展等测试 | 历史参考，不作为当前依据 | 保留负结果价值 | 不能说 T3 已恢复 |
| lead_lag_screen | V2 | `lead_lag_screen/V2` | PCMCI+ / runtime / audit / sensitivity | 历史对照与支线 | 需谨慎参考 | PCMCI+ 不能替代主发现 |
| lead_lag_screen | V3 | `lead_lag_screen/V3` | EOF/PC1 等对照 | 历史对照 | 保留计算成本/低分辨率经验 | 不能当主线发现 |
| lead_lag_screen | V4 | `lead_lag_screen/V4` | 未在本轮接清 | 工程存在，科研角色待核对 | 暂不依赖 | 不能自动当最新主线 |
| stage_partition | V1 | `stage_partition/V1` | 旧阶段分割 | 已废弃 | 不作为依据 | anomaly 混入 state 定义错误 |
| stage_partition | V2 | `stage_partition/V2` | raw smoothed field profile 阶段分割主线底座 | 历史主线底座，需接回 | 方法原则仍有效 | 本轮未逐文件核查 |
| stage_partition | V7 | `stage_partition/V7` | accepted-window field-level timing diagnostic | 已实现，有工程承载；运行结果需本地核查 | 可作继承底座 | 不做因果，不重新裁决窗口 |
| stage_partition | V10 main | `stage_partition/V10` | V9 peak/subpeak 语义复现 | 已运行且结果可用 | 方法层底座 | 不重新发现窗口，不做物理解释 |
| stage_partition | V10.1 | `stage_partition/V10/v10.1` | joint main-window reproduce | 已运行且结果可用 | 方法层主骨架 | 不做 object-native，不做 pair-order |
| stage_partition | V10.2 | `stage_partition/V10/v10.2` | object-native peak discovery | 已运行且结果可用 | 多峰族底座 | object-native peak 不等于 strict joint peak |
| stage_partition | V10.3 | `stage_partition/V10/v10.3` | peak discovery sensitivity | 已运行且结果可用 | 方法敏感性底座 | 不是对象间 order 审计 |
| stage_partition | V10.4 | `stage_partition/V10/v10.4` | object-to-object order sensitivity | 已运行且结果可用 | 对象间 order 方法层底座 | order 不等于因果 |
| stage_partition | V10.5 | `stage_partition/V10/v10.5` | profile-energy / index / strength-stability validation | 已运行且结果可用，但解释需限制 | 方法层验证 | 不是物理解释层，不是因果证据 |
| pending | peak_family_atlas | 未见清晰工程承载 | 峰族图谱整理 | 已讨论但未实现 | 不能依赖 | 不能写成已有结果 |
| pending | peak morphology audit | 未见清晰工程承载 | 峰宽、谷深、峰清晰度量化 | 已讨论但未实现 | 不能依赖 | V clean / Je broad 不能写成统计事实 |
| pending | cartopy spatial validation | 未见清晰工程承载 | 空间场验证 | 已讨论但未实现 | 不能依赖 | 不能写成物理空间证据 |
| pending | yearwise object-order support | 未见完整承载 | 年际稳定性审计 | 已讨论但未实现 | 不能依赖 | V10.5_d 不等于完整 object-order yearwise support |
| pending | object physical metrics validation | 未见正式承载 | 正式对象物理指标验证 | 已讨论但未实现 | 不能依赖 | low-dimensional metrics 不等于正式物理指标 |

---

## 2. V7 执行链状态

### 工程承载

- `stage_partition/V7/scripts/run_field_transition_timing_v7_a.py`
- `stage_partition/V7/src/stage_partition_v7`
- `stage_partition/V7/outputs`
- `stage_partition/V7/logs`
- `stage_partition/V7/README.md`
- `stage_partition/V7/STRUCTURE_MANIFEST.txt`

### 已知执行链语义

V7-a：

1. 读 V6/V6_1 logs 和 output tables 作为 audit evidence；
2. 验证四个 accepted bootstrap-supported points；
3. 读取 V6_1 derived window table，只保留 45、81、113、160；
4. 从 `smoothed_fields.npz` 重建 stage-partition profiles；
5. 构建 P/V/H/Je/Jw field-only state matrices；
6. 应用同 V6 的 `ruptures.Window` score-profile semantics；
7. 提取每个 field 在每个 accepted transition window 内的 peak score day；
8. 写表、audit logs、可选 score-profile plots。

### 状态判断

- 工程承载：清晰；
- 运行结果：存在 outputs/logs，但本轮未逐表核验；
- 科研角色：accepted-window field-level timing diagnostic；
- 使用边界：不做因果、不重新裁决窗口、不纳入全部 candidate windows。

---

## 3. V10 主工程线执行链状态

### 工程承载

- `stage_partition/V10/scripts/run_peak_subpeak_reproduce_v10_a.py`
- `stage_partition/V10/src/stage_partition_v10/peak_subpeak_reproduce_v10_a.py`
- `stage_partition/V10/outputs/peak_subpeak_reproduce_v10_a`
- `stage_partition/V10/logs/peak_subpeak_reproduce_v10_a`

### 状态判断

- 已实现；
- 已运行；
- 方法层结果可用；
- 作为 V9/V10 peak/subpeak 语义复现底座。

### 边界

- 不重新发现 accepted windows；
- 不做 object-native full-season peak discovery；
- 不做物理解释。

---

## 4. V10.1 执行链状态

### 工程承载

- `stage_partition/V10/v10.1/scripts`
- `stage_partition/V10/v10.1/src`
- `stage_partition/V10/v10.1/outputs/joint_main_window_reproduce_v10_1`
- `stage_partition/V10/v10.1/logs`

### 状态判断

- 已实现；
- 已运行；
- 结果可用；
- 复现 joint candidate → bootstrap → band → derived windows → strict accepted lineage。

### 关键结果

- joint candidates：18、45、81、96、113、132、135、160；
- strict accepted：45、81、113、160；
- non-strict：18、96、132、135。

### 边界

- 不做 object-native peak discovery；
- 不做 pair-order；
- 不重新裁决 non-strict。

---

## 5. V10.2 执行链状态

### 工程承载

- `stage_partition/V10/v10.2/scripts`
- `stage_partition/V10/v10.2/src`
- `stage_partition/V10/v10.2/outputs/object_native_peak_discovery_v10_2`
- `stage_partition/V10/v10.2/logs`

### 状态判断

- 已实现；
- 已运行；
- 结果可用；
- 建立 P/V/H/Je/Jw object-native candidate peak catalog。

### 重要解释边界

- H day19 是 object-native weak candidate / joint day18 non-strict lineage，不是污染；
- H day35 是 W045-conditioned / lineage-assigned peak；
- object-native peak 不等于 joint strict accepted window。

---

## 6. V10.3 执行链状态

### 工程承载

- `stage_partition/V10/v10.3/scripts`
- `stage_partition/V10/v10.3/src`
- `stage_partition/V10/v10.3/outputs/peak_discovery_sensitivity_v10_3`
- `stage_partition/V10/v10.3/logs`

### 状态判断

- 已实现；
- 已运行；
- 结果可用；
- 完成 peak discovery sensitivity 与 closure audits。

### 关键判断

- peak day 主要对 input smoothing 与 detector_width 敏感；
- match radius 主要影响 support class；
- band / merge 主要影响 derived window；
- penalty 与 local peak distance 对 peak day 影响较小。

### 边界

- V10.3 的 candidate order inversion audit 不是对象间 order sensitivity；
- 不得写成“V10.3 已证明对象间次序稳定”。

---

## 7. V10.4 执行链状态

### 工程承载

- `stage_partition/V10/v10.4/scripts`
- `stage_partition/V10/v10.4/src`
- `stage_partition/V10/v10.4/outputs/object_order_sensitivity_v10_4`
- `stage_partition/V10/v10.4/logs`

### 状态判断

- 已实现；
- 已运行；
- 结果可用；
- 是真正的 object-to-object order sensitivity 审计。

### 关键判断

- tau=2 下没有 true order reversal；
- 主要不确定来自 near-tie / grouping；
- 不是对象间方向整体颠倒。

### 边界

- order 不等于 causality；
- near-tie 内部不能硬排序；
- 不裁决 non-strict 是否应进入主窗口。

---

## 8. V10.5 执行链状态

### 工程承载

- `stage_partition/V10/v10.5/scripts/run_field_index_validation_v10_5_a.py`
- `stage_partition/V10/v10.5/src`
- `stage_partition/V10/v10.5/outputs`
- `stage_partition/V10/v10.5/logs`
- `stage_partition/V10/v10.5/README.md`
- `stage_partition/V10/v10.5/README_STRENGTH_CURVE_EXPORT_V10_5_E.md`

### 状态判断

- V10.5_a 已实现并运行；
- V10.5_b/c/d 已作为 hotfix/扩展输出纳入；
- V10.5_e strength curve export 已实现并运行；
- 结果可用，但解释必须限制。

### 关键输出层

- validation summary；
- profile-energy curves；
- top-k profile-energy peaks；
- assigned peak energy rank；
- candidate family switch inventory；
- selection-reason audit；
- strength–stability matrix；
- key competition cases；
- full-season strength curves。

### 关键解释

- V10.5 没有推翻 V10.1–V10.4 方法层主线；
- V10.5 推翻的是“单峰、干净、强先行”的过度简化解释；
- profile-energy top1 mismatch 不等于 assigned peak 不支持；
- strength 与 bootstrap stability 必须分开。

### 边界

- 不重跑 peak discovery；
- 不重新定义 accepted windows；
- 不提供 physical mechanism；
- 不提供 causal proof；
- 不等于 cartopy spatial validation；
- 不等于 formal object physical metrics validation。

---

## 9. 当前未实现项状态表

| 任务 | 状态 | 需要承载 | 当前不能写成 |
|---|---|---|---|
| peak_family_atlas_v10_5_f | 已讨论但未实现 | 峰族、rank、support、lineage、角色表 | 已完成图谱 |
| peak morphology audit | 已讨论但未实现 | 峰宽、谷深、半高宽、峰间分离度 | V clean / Je broad 已量化 |
| cartopy spatial validation | 已讨论但未实现 | 空间场差异图、profile evolution maps | 物理空间机制已证明 |
| yearwise object-order support | 已讨论但未实现 | 年际稳定性、order recurrence | object order 年际稳健已证明 |
| object physical metrics validation | 已讨论但未实现 | H_strength、Jw_axis_lat、P band share 等正式物理指标 | 低维粗指标已足够 |
| V10 与 lead-lag/pathway 重接 | 已讨论但未实现 | anomaly index 与 smooth-field peak 口径对齐 | V10 已替代 pathway 主线 |

---

## 10. 后续更新规则

以后每次新增版本或补丁，不再写子目录 `UPDATE_LOG_*`，而是在本日志中追加：

- 分支；
- 版本；
- 工程承载；
- 入口；
- 输出；
- 状态；
- 可依赖程度；
- 不能误读。
