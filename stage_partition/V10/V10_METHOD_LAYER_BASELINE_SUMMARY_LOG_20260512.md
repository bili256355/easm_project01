# V10 方法层基底总结日志

文件名：`V10_METHOD_LAYER_BASELINE_SUMMARY_LOG_20260512.md`  
日期：2026-05-12  
适用工程层级：`D:\easm_project01\stage_partition\V10`  
日志性质：方法层基底总结 / 接管封口记录  
不是：论文结论、物理解释、对象间因果/先后结论、accepted windows 重新裁决

---

## 0. 当前总状态

截至本日志，V10 线已经完成四类方法层基底工作：

1. **V10 主工程线**：独立语义复现 V9 的 window-conditioned object peak/subpeak 抓取层。
2. **V10.1**：独立语义复现联合对象从自由 peak 点到 derived windows，再到 strict accepted lineage 的主窗口发现链条。
3. **V10.2**：对 P / V / H / Je / Jw 分别建立 object-native full-season peak catalog，并映射到 V10.1 joint lineage 与 V10 window-conditioned object peak。
4. **V10.3**：基于 V10.1 / V10.2，完成 joint_all + 五对象自由发现 peak 流程的单因素敏感性测试，并完成封口审计。

当前可作为后续工作的**方法层基底**。  
当前不可直接作为**物理解释层结果**。

---

## 1. 已废除内容与废除原因

以下旧内容不得再作为科研结论或方法证据链继续使用：

- `V9 peak_selection_sensitivity_v9_a` 的旧混合式 sensitivity 结论；
- `W045_Jw_H_peak_sensitivity_cause_audit_v1`；
- `W045_Jw_H_peak_sensitivity_cause_audit_v1_1`；
- 旧审计中关于 H 领先不可信、H early cluster 不合法、Jw axis-shift candidate、Jw-H order inflated、v1/v1_1 的 primary cause / interpretation level / filtered order、把 day17–19 / day18 解释成污染或外侧异常的所有判定。

废除原因：

1. 旧 sensitivity 不是严格的 V9 复现型敏感性测试，而是混合了 V9 original candidates 与 fallback local-prepost contrast detector。
2. v1/v1_1 建立在旧 sensitivity 输出上，继承了候选生成语义不一致问题。
3. v1_1 使用 fixed system/core window 过滤前导对象，导致 H 的 V9 baseline peak day35 被错误降级。
4. 旧流程没有先建立 joint free-detection lineage 和 object-native candidate catalog，导致已知弱候选被误判为污染。
5. 旧 pair-order / admissibility 判定属于后处理规则，不是原方法直接输出。

保留价值：这些旧内容只作为失败记录和设计禁区保留；不能进入后续结果解释。

---

## 2. V10 主工程线：V9 object peak/subpeak 抓取复现

位置：

```text
V10/scripts/run_peak_subpeak_reproduce_v10_a.py
V10/src/stage_partition_v10/peak_subpeak_reproduce_v10_a.py
V10/outputs/peak_subpeak_reproduce_v10_a
```

已确认四层 regression audit 全部通过：

```text
main_window_selection              pass
object_profile_window_registry     pass
raw_profile_detector_scores        pass
bootstrap_selected_peak_days       pass
```

确认内容：

- V10 主工程版本已经完全复现 V9 的 peak/subpeak 抓取输出层；
- 主峰选择、候选 subpeak registry、raw detector score、bootstrap selected peak days 均与 V9 对齐；
- 早期 V10-a1 中出现的 bootstrap 差异已定位为 overlap 语义复刻错误：V9 使用 `overlap_fraction`，V10-a1 错用 `overlap_days`；
- hotfix02 后该错误已修正，差异归零；
- 旧临时 audit bundle 可删除，当前可信入口是 V10 主工程线。

边界：V10 主工程线复现的是**已知 accepted windows 内的 window-conditioned object peak/subpeak 抓取**；它不负责重新发现 accepted windows、object-native full-season peak discovery 或物理解释。

---

## 3. V10.1：联合对象主窗口发现链条复现

位置：

```text
V10/v10.1/scripts/run_joint_main_window_reproduce_v10_1.py
V10/v10.1/src/joint_main_window_reproduce_v10_1.py
V10/v10.1/outputs/joint_main_window_reproduce_v10_1
```

复现链条：

```text
联合对象 state matrix
    ↓
ruptures.Window detector
    ↓
detector local peaks
    ↓
candidate registry
    ↓
bootstrap recurrence support
    ↓
candidate point bands
    ↓
derived windows
    ↓
window membership
    ↓
window uncertainty / return-day distribution
    ↓
strict accepted lineage mapping
```

已确认 regression audit 全部通过：

```text
ruptures_primary_points                  pass
detector_local_peaks_all                 pass
baseline_detected_peaks_registry         pass
candidate_points_bootstrap_summary       pass
candidate_points_bootstrap_match_records pass
candidate_point_bands                    pass
derived_windows_registry                 pass
window_point_membership                  pass
window_uncertainty_summary               pass
window_return_day_distribution           pass
```

V10.1 复现出的 joint-all candidate peak days：

```text
18, 45, 81, 96, 113, 132, 135, 160
```

V10.1 复现出的 derived windows：

```text
W001: 16–20,    main peak 18
W002: 40–48,    main peak 45
W003: 75–86,    main peak 81
W004: 94–99,    main peak 96
W005: 107–117,  main peak 113
W006: 128–138,  main peak 132, members 132 + 135
W007: 155–164,  main peak 160
```

strict accepted lineage：

```text
45, 81, 113, 160
```

non-strict / derived candidate lineage：

```text
18, 96, 132, 135
```

关键纠错：

- day18 不是污染；它是 joint free-detection chain 中的 derived non-strict candidate window；
- day96 / day132 / day135 也属于已知 joint candidate lineage；
- W006 中 day132 和 day135 是近邻候选组，day132 为 baseline derived window main peak，day135 为成员候选；
- strict accepted set 是 derived candidate windows 的主线子集，不是全体 candidate lineage。

边界：V10.1 不重新裁决 accepted windows，不做单对象峰发现、敏感性测试或物理解释。

---

## 4. V10.2：对象自身 full-season peak discovery

位置：

```text
V10/v10.2/scripts/run_object_native_peak_discovery_v10_2.py
V10/v10.2/src/object_native_peak_discovery_v10_2.py
V10/v10.2/outputs/object_native_peak_discovery_v10_2
```

任务：

```text
P / V / H / Je / Jw 各自 full-season object-native peak discovery
    ↓
bootstrap recurrence support
    ↓
object-derived windows
    ↓
mapping to V10.1 joint lineage
    ↓
mapping to V10 window-conditioned object peak
```

对象候选峰谱系：

```text
P:  18, 34, 45, 65, 82, 95, 113, 126, 138, 158
V:  16, 30, 45, 69, 80, 95, 114, 136, 161
H:  19, 35, 57, 77, 95, 115, 129, 155
Je: 26, 33, 46, 81, 94, 111, 136, 162
Jw: 41, 83, 104, 130, 152
```

关键 H 纠错：

```text
H day19:
    H object-native weak candidate
    maps to joint day18 non-strict lineage
    not pollution
    not W045-H main peak

H day35:
    H object-native candidate
    also V10 W045-conditioned H main peak
```

其他对象也存在 non-strict lineage 附近对象候选，例如 P/V 接近 day18，P/V/H/Je 接近 day96，P/V/H/Je/Jw 接近 day132/135。

边界：V10.2 只是对象自身候选峰谱系；不说明某个峰有物理意义，不说明对象间先后关系，不重新纳入 non-strict windows，不做敏感性测试。

---

## 5. V10.3：peak discovery sensitivity

位置：

```text
V10/v10.3/scripts/run_peak_discovery_sensitivity_v10_3.py
V10/v10.3/src/peak_discovery_sensitivity_v10_3.py
V10/v10.3/outputs/peak_discovery_sensitivity_v10_3
```

覆盖 scope：

```text
joint_all
P_only
V_only
H_only
Je_only
Jw_only
```

测试组：

```text
detector_width
detector_penalty
local_peak_distance
bootstrap_match_radius
candidate_band
window_merge
smooth_input
smooth5_internal_detector_width
smooth5_internal_penalty
smooth5_internal_peak_distance
smooth5_internal_match_radius
smooth5_internal_band
smooth5_internal_merge
```

### 5.1 当前支持的方法层判断

```text
candidate peak 点主要对两类因素敏感：
    1. input smoothing
    2. detector_width

其他参数主要作用于派生层：
    match_radius        → bootstrap support class
    candidate_band      → candidate band / derived window boundary
    window_merge        → derived window merge / main peak attribution
    detector_penalty    → 基本不改变 peak day
    local_peak_distance → 影响较小
```

### 5.2 smooth9 内部

在固定 smooth9 input 下：

- detector_width 是主要 peak-day 敏感源；
- detector_penalty 基本不改变 peak day；
- match_radius 不改变 peak day，主要改变 support class；
- band / merge 不改变 peak day，主要影响 derived window。

### 5.3 smooth5 输入层

smooth5 文件已确认真实加载：

```text
baseline_smooth9:
D:\easm_project01\foundation\V1\outputs\baseline_a\preprocess\smoothed_fields.npz

smooth5:
D:\easm_project01\foundation\V1\outputs\baseline_smooth5_a\preprocess\smoothed_fields.npz
```

smooth5 输入相对 smooth9：

- 会引入候选峰小幅日偏移；
- 会引入少数新增候选；
- 会改变部分 support class；
- 不是峰谱系整体崩塌。

关键 H 结果：

```text
H smooth9 baseline:
19, 35, 57, 77, 95, 115, 129, 155

H smooth5:
18, 35, 55, 60, 77, 95, 113, 129, 145, 155
```

因此：

- H day35 在 smooth5 下仍为 day35；
- H day19 对应到 smooth5 day18；
- day18/19 是稳定存在的 object-native candidate family；
- smooth5 下 H 若干候选 support class 变强。

### 5.4 smooth5 内部

smooth5 内部继续测试后确认：在 smooth5 input 内部，candidate peak day 仍主要对 detector_width 敏感。

smooth5 内部统计：

```text
detector_width:
    candidate shift / missing = 82
    windows changed = 91

local_peak_distance:
    candidate shift / missing = 4

detector_penalty:
    candidate shift / missing = 0

bootstrap_match_radius:
    candidate shift / missing = 0
    support class changed = 32

candidate_band:
    candidate shift / missing = 0
    windows changed = 55

window_merge:
    candidate shift / missing = 0
    windows changed = 18
```

### 5.5 HOTFIX06 封口审计

新增封口审计表：

```text
cross_scope/candidate_order_inversion_summary_v10_3.csv
cross_scope/detector_width_bootstrap_support_summary_v10_3.csv
cross_scope/candidate_shift_type_summary_v10_3.csv
lineage_mapping/new_candidate_inventory_v10_3.csv
```

已确认：

#### a. order inversion

```text
n_inversions total = 0
NO_ORDER_INVERSION = 240
INSUFFICIENT_MATCHED_CANDIDATES = 13
```

含义：在有足够 matched candidates 的配置中，没有发现已匹配候选峰之间的前后顺序颠倒。注意这不是对象间次序结论。

#### b. detector_width bootstrap

已对以下最小组补跑 bootstrap：

```text
DET_WIDTH_16
DET_WIDTH_24
SMOOTH5_DET_WIDTH_16
SMOOTH5_DET_WIDTH_24
```

结果：

```text
detector_width:
    candidates compared = 96
    support class changed = 39

smooth5_detector_width:
    candidates compared = 120
    support class changed = 51
```

含义：detector_width 不只影响 peak day，也会影响部分 bootstrap support class；smooth5 内部这种影响更明显。

#### c. shift 类型拆分

关键结论：

```text
missing_baseline_candidate = 0
```

也就是说，baseline 候选没有真正消失。主要变化是 same day、±2 天以内小幅漂移、少量 3–5 天或 5–8 天移动、少量大位移 / new candidate。

#### d. new candidate inventory

新增候选总数：

```text
21
```

来源：

```text
DET_WIDTH_16 = 7
SMOOTH_INPUT_5D = 7
SMOOTH5_DET_WIDTH_16 = 7
```

lineage 类型：

```text
JOINT_STRICT_ACCEPTED_LINEAGE = 12
JOINT_NON_STRICT_OR_KNOWN_LINEAGE = 6
NEW_CANDIDATE_NO_KNOWN_LINEAGE = 3
```

含义：大多数新增候选仍靠近已知 joint lineage；只有少数无已知 lineage；新增候选主要出现在较短 detector_width 或 smooth5 输入中，符合短尺度结构更容易显露的预期。

---

## 6. 当前最稳妥的方法层结论

当前 V10.3 可以作为后续工作基底，支持以下方法层结论：

1. peak discovery 并非对所有参数广泛敏感。
2. candidate peak day 的主要敏感源是 input smoothing 与 detector_width。
3. 在固定输入下，无论 smooth9 还是 smooth5，detector_width 都是主要 peak-day 敏感源。
4. match radius 主要影响 bootstrap support class，而不是 peak day。
5. candidate band 与 window merge 主要影响 peak→derived window 的派生层，而不是 peak day。
6. detector penalty 基本不改变 peak day。
7. local peak distance 影响较小。
8. 具体 peak day 存在几日尺度浮动，因此后续不应把单日日期写成无误差日。
9. 已匹配候选峰之间没有出现前后顺序颠倒。
10. baseline 候选没有真正 missing；敏感性更多表现为小幅漂移、局部新增、局部合并/拆分、support class 变化。

建议总表述：

```text
V10.3 的敏感性审计显示，peak discovery 的主要不确定性集中在输入平滑尺度和 detector_width；其他参数主要作用于 support class 或 derived-window 派生层。候选峰的具体日期会有几日浮动，并有少数新增候选，但已匹配候选峰之间没有出现前后顺序颠倒。detector_width 作为核心检测尺度，不仅影响 peak day，也会影响部分 bootstrap support class，尤其在 smooth5 输入下更明显。
```

---

## 7. 当前不能支持的内容

当前仍不能写成：

```text
H 物理领先
Jw-H 谁先谁后
P/V/H/Je/Jw 对象间相对次序稳定
某个 candidate 有物理机制
non-strict candidate 应进入主窗口
strict accepted windows 应重新裁决
detector_width 敏感意味着方法失败
smooth5 比 smooth9 更正确
smooth9 比 smooth5 更正确
```

特别注意：V10.3 的 order inversion 是 **scope 内 candidate sequence** 的 order inversion；它不是五对象之间的 order sensitivity。若后续讨论对象间次序，需要另做 candidate-lineage-aware object order sensitivity。

---

## 8. 当前仍需保留的边界

1. V10.1 / V10.2 / V10.3 是**方法层底座**，不是物理解释结果。
2. day18 / day19 是 known weak / non-strict lineage，不是污染，但也不是主结果。
3. day35 是 H object-native candidate，并且是 V10 W045-conditioned H main peak。
4. 132 / 135 是近邻候选组；peak 存在较稳定，但 derived window main peak 归属有派生规则敏感性。
5. detector_width 是 detection temporal scale 参数，不是物理窗口宽度。
6. input smoothing 是独立敏感源；其变化正常，因为输入数据本身改变。
7. 后续若进入对象间次序或物理解释，必须另起分析层，不得直接把 V10.3 方法层结果当成物理结论。

---

## 9. 后续建议

可以基于当前方法层底座继续推进，但推荐顺序为：

1. 先整理 V10.1–V10.3 的方法层结果表；
2. 再决定是否进入：
   - object-order sensitivity；
   - selected candidates 的 profile / spatial physical audit；
   - specific windows 的对象间相对时序；
   - non-strict lineage 的物理背景审计；
3. 不建议继续无限扩展参数网格。当前方法层敏感性已经基本收敛到：
   - input smoothing；
   - detector_width；
   - support class / derived window 派生层。

---

## 10. 最终封口状态

当前 V10 方法层底座可标记为：

```text
V10 main peak/subpeak reproduction: PASS
V10.1 joint main-window discovery lineage reproduction: PASS
V10.2 object-native peak discovery catalog: PASS
V10.3 peak discovery sensitivity method audit: PASS_FOR_METHOD_LAYER_BASELINE
```

科研状态：

```text
可作为后续分析底座；
不可直接作为物理结论；
不可替代后续对象间次序或空间/物理解释审计。
```
