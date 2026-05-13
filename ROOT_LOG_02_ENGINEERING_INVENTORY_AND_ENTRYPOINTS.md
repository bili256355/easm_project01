# ROOT_LOG_02_ENGINEERING_INVENTORY_AND_ENTRYPOINTS

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_02_ENGINEERING_INVENTORY_AND_ENTRYPOINTS.md`

## 0. 本日志的性质

这是当前工程的**根目录工程索引与入口日志**。

它只记录工程承载、目录、入口、输出、已知边界。它不把“文件存在”自动解释为“科研结论成立”。

本日志基于 GitHub 可见结构与会话迁移包整理；未完成本地全量代码运行验收。

---

## 1. 顶层工程结构

GitHub 可见根目录包括：

- `.idea`
- `foundation/V1`
- `index_validity`
- `lead_lag_screen`
- `logs`
- `shared_docs`
- `stage_partition`
- `.gitattributes`
- `.gitignore`
- 多个 `CURRENT_ENGINEERING_MANIFEST_*`
- 多个 `PATCH_FILE_LIST_*`
- 多个 `PATCH_MANIFEST_*`
- 多个 `PATCH_README_*`
- 多个 `ROOT_PROJECT_LOG_20260427*`

工程语言：Python 100%。

接管判断：

> 工程不是单一 V10 分支。V10 是当前 peak/order/multipeak 方法层分支，但根目录还包含 foundation、index_validity、lead_lag_screen、logs、shared_docs、stage_partition 多版本等承载。

---

## 2. `foundation/V1`

可见结构：

- `notebooks`
- `scripts`
- `src/foundation_v1`
- `DEPENDENCY_MANIFEST.json`
- `VERSION_POLICY.md`

研究/工程角色：

- 预处理与 foundation 输入底座；
- 与 smoothed fields、baseline smooth9、smooth5 输入相关；
- 当前 V10.1–V10.5 使用 foundation smoothed fields。

当前状态：

- 工程承载存在；
- 未在本轮完成本地输入数据文件核查；
- 不能仅凭目录确认 `smoothed_fields.npz` 在本地完整存在或与输出完全一致。

---

## 3. `index_validity`

可见状态：

- 根目录存在 `index_validity`；
- 根目录也有 `CURRENT_ENGINEERING_MANIFEST_index_validity_smooth5_v1_a.md` 与对应 patch 文件。

当前状态：

- 工程承载存在；
- 科研角色未在本轮 V10 迁移包中完全接清；
- 暂标记为：**工程存在，科研角色待核对**。

---

## 4. `lead_lag_screen`

可见子版本：

- `V1`
- `V1_1`
- `V2`
- `V3`
- `V4`

根目录还存在大量 lead-lag 相关 manifest / patch：

- `CURRENT_ENGINEERING_MANIFEST_lead_lag_screen_v1_a.md`
- `CURRENT_ENGINEERING_MANIFEST_lead_lag_screen_v2_b_audit.md`
- `CURRENT_ENGINEERING_MANIFEST_lead_lag_screen_v2_c_sensitivity.md`
- `CURRENT_ENGINEERING_MANIFEST_lead_lag_screen_v2_pcmciplus_*`
- `PATCH_README_lead_lag_screen_*`
- `PATCH_FILE_LIST_lead_lag_screen_*`

研究角色：

- 早期 lead-lag anomaly index 继承线；
- AR(1) surrogate、bootstrap、显著性、稳定性检验的重要底座；
- PCMCI+ / LPCMCI / EOF / CCA 等对照路线的承载；
- T3/T4 V→P 关系弱化和 surrogate null 过滤的历史背景承载。

当前状态：

- 工程承载存在；
- 本轮未逐个输出文件核查；
- 当前不能把 lead_lag_screen 的任一版本自动提升为最新主线；
- 后续若回到 pathway / lead-lag 主问题，必须重新接回 anomaly index 口径与 null/stability 检验。

---

## 5. `stage_partition`

可见子版本：

- `V1`
- `V2`
- `V3`
- `V4`
- `V5`
- `V6`
- `V6_1`
- `V7`
- `V8`
- `V9`
- `V9_1`
- `V9_2`
- `V9_3`
- `V10`

工程判断：

> stage_partition 是当前工程中最主要的版本化诊断树之一。不同版本不能自动视为线性继承的最新结果；必须结合日志、README、outputs 与会话迁移包判断其科研地位。

---

## 6. `stage_partition/V7`

可见结构：

- `logs`
- `outputs`
- `scripts`
- `src/stage_partition_v7`
- `README.md`
- `STRUCTURE_MANIFEST.txt`
- 大量 `UPDATE_LOG_V7_*`

V7 README 接收状态：

- V7 是建立在 accepted stage-partition time-object layer 之上的独立诊断分支；
- 第一任务是：对四个 accepted transition windows，诊断各 field-level state 何时达到 strongest detector-profile transition signal；
- accepted windows：45、81、113、160；
- candidate/sub-threshold peaks：18、96、132、135 不用于 V7-a 诊断。

V7-a 做什么：

- 读 V6/V6_1 logs 和 output tables 作为 audit evidence；
- 验证四个 accepted bootstrap-supported points；
- 读取 V6_1 derived window table，仅保留 anchored at 45、81、113、160 的窗口；
- 从 `smoothed_fields.npz` 重建 profiles；
- 构建 P/V/H/Je/Jw field-only state matrices；
- 使用与 V6 相同的 `ruptures.Window` score-profile semantics；
- 提取每个 field 在每个 accepted transition window 内的 peak score day；
- 写表、audit logs 和可选 score-profile plots。

V7-a 不做什么：

- 不重新裁决 accepted windows；
- 不纳入所有 candidate windows；
- 不使用 downstream lead-lag/pathway 结果；
- 不推断因果方向；
- 不诊断 spatial earliest/latest regions；
- 不把 timing order consistency 当作科学 claim。

主入口：

```bash
python D:\easm_project01\stage_partition\V7\scriptsun_field_transition_timing_v7_a.py
```

当前状态：

- 工程承载清晰；
- 角色是 accepted-window field-level timing diagnostic；
- 不能替代 V10 multipeak/order 分支；
- 大量 V7 更新日志应被根目录日志吸收为版本历史，不建议继续分散保留。

---

## 7. `stage_partition/V10`

可见结构：

- `logs/peak_subpeak_reproduce_v10_a`
- `outputs/peak_subpeak_reproduce_v10_a`
- `scripts`
- `src/stage_partition_v10`
- `v10.1`
- `v10.2`
- `v10.3`
- `v10.4`
- `v10.5`
- `README_V10_PEAK_SUBPEAK_REPRODUCE_A.md`
- `README_V10_PEAK_SUBPEAK_REPRODUCE_A_HOTFIX02_MAIN.md`
- `UPDATE_LOG_V10_PEAK_SUBPEAK_REPRODUCE_A.md`
- `UPDATE_LOG_V10_PEAK_SUBPEAK_REPRODUCE_A_HOTFIX02_MAIN.md`
- `V10_METHOD_LAYER_BASELINE_SUMMARY_LOG_20260512.md`

当前工程角色：

> V10 是当前 peak/subpeak reproduce、joint lineage、object-native peak discovery、sensitivity、object order sensitivity、profile-energy validation 的主承载分支。

---

## 8. `stage_partition/V10` 主工程线

主入口：

```bash
python D:\easm_project01\stage_partition\V10\scriptsun_peak_subpeak_reproduce_v10_a.py
```

主要承载：

- `V10/scripts/run_peak_subpeak_reproduce_v10_a.py`
- `V10/src/stage_partition_v10/peak_subpeak_reproduce_v10_a.py`
- `V10/outputs/peak_subpeak_reproduce_v10_a`
- `V10/logs/peak_subpeak_reproduce_v10_a`

研究角色：

- 复现 V9 object peak/subpeak 抓取语义；
- 提供 window-conditioned object peak/subpeak 复现底座；
- 不负责重新发现 accepted windows；
- 不负责 object-native full-season discovery；
- 不负责物理解释。

---

## 9. `stage_partition/V10/v10.1`

可见结构：

- `logs`
- `outputs/joint_main_window_reproduce_v10_1`
- `scripts`
- `src`
- `README.md`
- `UPDATE_LOG.md`

研究角色：

- joint main-window reproduce；
- 复现 joint objects → full-season candidate peaks → bootstrap support → candidate point bands → derived windows → strict accepted-window lineage。

边界：

- 不做 object-native peak discovery；
- 不做 pair-order；
- 不做物理解释；
- 不重新裁决 W135 或其他 non-strict windows。

---

## 10. `stage_partition/V10/v10.2`

可见结构：

- `logs`
- `outputs/object_native_peak_discovery_v10_2`
- `scripts`
- `src`
- `README.md`
- `UPDATE_LOG.md`

研究角色：

- P/V/H/Je/Jw object-native full-season peak discovery；
- 使用与 V10.1 已验证的 free-season discovery 语义；
- 将 object-native candidates 映射到 V10.1 joint lineage 与 V10 window-conditioned peaks。

边界：

- 不做 sensitivity；
- 不做物理解释。

---

## 11. `stage_partition/V10/v10.3`

可见结构：

- `logs`
- `outputs/peak_discovery_sensitivity_v10_3`
- `scripts`
- `src`
- `README.md`
- `UPDATE_LOG.md`

研究角色：

- peak discovery sensitivity；
- 覆盖 joint_all 与 P/V/H/Je/Jw object-native free-season peak discovery；
- 识别 input smoothing、detector_width、match radius、band、merge、penalty 等因素对 peak/support/window 派生的影响。

边界：

- 不做物理解释；
- 不做 pair-order；
- 不重新裁决 strict accepted windows。

---

## 12. `stage_partition/V10/v10.4`

可见结构：

- `logs`
- `outputs/object_order_sensitivity_v10_4`
- `scripts`
- `src`
- `README.md`
- `UPDATE_LOG.md`

研究角色：

- object-to-object order sensitivity；
- 读取 V10.1 joint lineage、V10.2 object-native peak catalog/mapping、V10.3 sensitivity results；
- 不重跑 peak discovery、bootstrap、detector scoring、band construction、window merging；
- 只做 candidate assignment 和 object-to-object order comparison。

边界：

- 方法层 timing/order stability audit；
- 不建立因果；
- 不建立物理机制；
- 不裁决 non-strict candidate 是否应进入主结果。

---

## 13. `stage_partition/V10/v10.5`

可见结构：

- `logs`
- `outputs`
- `scripts`
- `src`
- `README.md`
- `README_STRENGTH_CURVE_EXPORT_V10_5_E.md`
- `UPDATE_LOG.md`
- `UPDATE_LOG_STRENGTH_CURVE_EXPORT_V10_5_E.md`

当前入口：

```bash
python D:\easm_project01\stage_partition\V1010.5\scriptsun_field_index_validation_v10_5_a.py
```

可选输入覆盖：

```bash
set V10_5_SMOOTHED_FIELDS=D:\path	o\smoothed_fields.npz
```

Scope：

- Target lineages/windows：W045、W113、W160；
- Objects：P、V、H、Je、Jw；
- 使用 foundation smoothed fields 与 V10.4 baseline object assignments；
- 不重跑 peak discovery；
- 不重新定义 accepted windows；
- 不提供 physical mechanism 或 causal proof。

V10.5_a 输出：

- `validation_summary_v10_5_a.csv`
- `profile_validation/profile_energy_peak_by_window_object_k_v10_5_a.csv`
- `profile_validation/profile_energy_curves_by_window_object_v10_5_a.csv`
- `profile_validation/profile_energy_primary_k_summary_v10_5_a.csv`
- `index_validation/object_metric_timing_by_window_v10_5_a.csv`
- `index_validation/object_index_support_summary_v10_5_a.csv`
- `order_validation/object_order_validation_by_window_v10_5_a.csv`

V10.5_b 输出：

- `profile_validation/profile_energy_topk_peaks_by_window_object_v10_5_b.csv`
- `profile_validation/v10_4_assigned_peak_energy_rank_v10_5_b.csv`
- `profile_validation/candidate_family_switch_inventory_v10_5_b.csv`
- `validation_summary_candidate_family_v10_5_b.csv`

V10.5_c 输出：

- `profile_validation/candidate_family_selection_reason_audit_v10_5_c.csv`
- `profile_validation/candidate_family_role_long_v10_5_c.csv`
- `validation_summary_candidate_family_selection_reason_v10_5_c.csv`

V10.5_d 输出：

- `profile_validation/main_method_peak_strength_vs_bootstrap_v10_5_d.csv`
- `profile_validation/profile_energy_family_bootstrap_stability_v10_5_d.csv`
- `profile_validation/profile_energy_family_pair_comparison_v10_5_d.csv`
- `profile_validation/candidate_family_strength_stability_matrix_v10_5_d.csv`
- `profile_validation/key_competition_cases_v10_5_d.csv`
- `FIELD_INDEX_VALIDATION_V10_5_D_STRENGTH_STABILITY_SUMMARY.md`

解释边界：

- V10.5 是 method-layer validation package；
- 可用于识别 profile-energy support、secondary support、candidate-family competition；
- 不是 physical explanation layer；
- 不应作为 direct causal evidence。

---

## 14. 当前工程索引结论

- V7 和 V10 的工程承载都较清晰；
- V10.1–V10.5 具有较明确的分工；
- lead_lag_screen 仍是重要继承分支，但本轮未完成全量重接；
- foundation/V1 是输入底座，但本轮未完成本地数据存在性核查；
- index_validity 工程存在，但科研角色暂未在本轮主线中接清；
- 大量旧日志应由根目录日志吸收，不建议继续分散保留。
