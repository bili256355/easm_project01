# lead_lag_screen/V2 PCMCI+ 对照组封口日志

**版本对象**：`lead_lag_screen/V2`  
**封口对象**：PCMCI+ smooth5 对照组，包括 `V2_a`、`V2_b_audit`、`V2_c_s3_t3_vp_c2_all_a`  
**封口日期**：2026-04-28  
**封口性质**：方法审计封口 / 负结果保留 / 后续主线降级说明  
**是否作为 pathway 主结果**：否  
**是否作为 lead–lag 主筛选工具**：否  
**唯一可保留使用场景**：代表变量池下的 PCMCI+ 附录级 / 压力测试级对照

---

## 1. 本轮 V2 的原始目的

`lead_lag_screen/V2` 的目的不是替代 `V1`，也不是直接建立 pathway，而是在 5 日指数底座上引入 PCMCI+，作为 `V1` 超前滞后相关筛选的严格对照组。

原始设定为：

```text
输入：foundation/V1/outputs/baseline_smooth5_a/indices/index_anomalies.csv
方法：PCMCI+ with ParCorr
窗口：沿用 V1 的 9 个窗口
滞后：主输出 tau=1..5，tau=0 单独作为同期诊断
变量池：沿用 V1 的 20 个指数
同族关系：不报告同族 source-target 边
conditioning：V2_a 中允许同族变量进入 controls
输出性质：conditional direct-edge control，不是 pathway result
```

---

## 2. V2_a 运行事实

`V2_a = pcmci_plus_smooth5_v2_a` 已完整运行。

关键运行事实：

```text
n_windows = 9
n_years = 45
n_variables = 20
n_reported_directed_pairs = 308
n_lagged_candidate_tests = 13860
n_lagged_graph_selected = 48
n_lagged_pcmci_plus_supported = 10
n_tau0_candidate_tests = 2772
n_tau0_supported = 88
```

工程状态：

```text
已进入执行链
已完整跑完
无 fallback
无失败窗口
输出表齐全
```

但结果层面出现严重问题：PCMCI+ lagged supported 只剩 10 条，且核心的 `V→P` 在多个关键窗口被异常压没。

---

## 3. V2_b 审计事实

`V2_b_audit = pcmci_plus_smooth5_v2_b_audit` 用于审计 V2_a 的压缩来源。

关键审计结果：

```text
n_lagged_tests = 13860
n_lagged_graph_selected = 48
n_lagged_graph_raw_p05 = 48
n_lagged_supported_window_fdr = 10
n_v1_lead_lag_yes = 248
n_v1_yes_pcmci_lagged_supported = 5
n_tau0_supported_pairs = 88
```

V1 yes 在 PCMCI+ 下的命运：

```text
V1_yes_and_PCMCI_lagged_supported = 5
V1_yes_graph_raw_p05_lost_by_fdr = 8
V1_yes_only_tau0_supported_in_PCMCI = 21
V1_yes_only_tau0_graph_selected_in_PCMCI = 11
V1_yes_no_PCMCI_lagged_or_tau0_signal = 203
```

审计结论：

```text
V2_a 的压缩不是单纯 FDR 过严造成的。
更大的压缩已经发生在 PCMCI+ graph-selection / conditioning 阶段。
```

因此，V2_a 不能解释为“严格但可靠地筛出了核心科学边”，而应解释为“全指数高维条件直接边设定下发生了系统性漏检”。

---

## 4. V→P sanity check 失败

V→P 是本项目中不能被 PCMCI+ 全面压没的基础关系。合理结果至少应表现为：

```text
lagged 中有一定系统性信号；
或者 tau0 中有成体系的同期/快速调整信号。
```

但 V2_a 中 V→P 在多个窗口表现异常：

```text
S3: lagged supported = 0, tau0 supported = 0
T3: lagged supported = 0, tau0 supported = 0
T4: lagged supported = 0, tau0 supported = 0
```

其中 S3/T3 对应梅雨期及其转换附近，V→P 完全消失不符合本研究的基本物理认知和 V1 证据。该现象被判定为 V2_a 的关键失效标志。

---

## 5. V2_c 病因定位事实

`V2_c = pcmci_plus_smooth5_v2_c_s3_t3_vp_c2_all_a` 不是为了生成新主结果，而是为了定位 V2_a 的病因。

### 5.1 C1：S3/T3 定向 no-same-family-controls

C1 只针对 `S3/T3` 的 `V→P`，去掉 source family 与 target family 内其他同族派生指标的 controls。

C1 汇总：

```text
variant = c1_targeted_no_same_family_controls
family_edge = V→P
n_lagged_tests = 210
n_lagged_graph_selected = 8
n_lagged_supported_window_fdr = 4
n_tau0_tests = 42
n_tau0_graph_selected = 19
n_tau0_supported = 16
```

窗口分解：

```text
S3 V→P:
  lagged graph selected = 8
  lagged supported window-FDR = 4
  tau0 graph selected = 12
  tau0 supported = 11

T3 V→P:
  lagged graph selected = 0
  lagged supported = 0
  tau0 graph selected = 7
  tau0 supported = 5
```

C1 结论：

```text
S3 原本 V→P 完全消失，是同族 / 同源派生指标 conditioning 过度控制造成的。
T3 的 lagged V→P 没有恢复，但 tau0 明显恢复，说明 T3 至少存在同期型 V→P 信号。
```

这证明 V2_a 把梅雨期 V→P 压没，不是科学事实，而是方法设定造成的失真。

### 5.2 C2：family-representative PCMCI+

C2 使用代表变量池全窗口运行，代表变量为：

```text
P_main_minus_south
P_spread_lat
P_total_centroid_lat_10_50
V_strength
V_NS_diff
H_strength
H_centroid_lat
Je_strength
Je_axis_lat
Jw_strength
Jw_axis_lat
```

C2 中 V→P 汇总：

```text
variant = c2_family_representative_standard
family_edge = V→P
n_lagged_tests = 270
n_lagged_graph_selected = 11
n_lagged_supported_window_fdr = 2
n_lagged_supported_family_fdr = 3
n_tau0_tests = 54
n_tau0_graph_selected = 27
n_tau0_supported = 20
n_v1_lead_lag_yes = 108
n_v1_yes_recovered_by_lagged_window_fdr = 2
n_v1_yes_recovered_by_tau0 = 18
```

C2 结论：

```text
代表变量池下，V→P 至少在多数窗口以 tau0 形式恢复，S2/S3 有一定 lagged 恢复。
这说明 20 个强共线派生指数一起进入 PCMCI+，会导致 graph-selection collapse / 过度残差化。
```

---

## 6. 根本病因判定

V2 的根本问题不是 tigramite 不可用，也不是 PCMCI+ 工程没有跑通，而是当前 all-index PCMCI+ 的检验对象与本研究对象发生了错位。

本研究真正关心的是：

```text
场对象之间的关系：V 场 ↔ P 场，H 场 ↔ P 场，Jw/Je/H/V/P 之间的窗口化关系骨架。
```

但 all-index PCMCI+ 实际检验的是：

```text
某一个 V 指数在剥离其他 V 指数之后，
对某一个 P 指数在剥离其他 P 指数之后的残差信息，
是否仍然具有独立 conditional direct effect。
```

这导致：

```text
同一物理场的多个派生指数被当作普通协变量互相控制；
场对象内部的共同结构信号被剥离；
V→P 这类基础场间关系被压缩成极窄的 index-specific residual effect；
最终出现核心物理关系被系统性漏检。
```

因此，当前 V2_a 的 all-index PCMCI+ 结果应判为：

```text
工程上有效；
方法审计上有价值；
科学对照层不可信；
不能作为主筛选或主解释结果。
```

---

## 7. PCMCI+ 在本项目中的唯一可保留使用方式

经 V2_a、V2_b、V2_c 审计后，PCMCI+ 在本项目中只有一种相对可用场景：

```text
先选出每个物理对象的代表变量，
在低维 family-representative 变量池上运行 PCMCI+，
并仅作为附录级 / 压力测试级 conditional direct-edge 对照。
```

不可再使用的场景：

```text
全 20 指数 all-index PCMCI+ 作为主结果；
把同族派生指标全部放进 conditioning pool 后解释 negative result；
用 V2_a headline 10 条边代表季风关系骨架；
用 PCMCI+ 结果否定 V1 lead–lag temporal eligibility；
把 PCMCI+ direct edge 直接写成 pathway established。
```

可保留的有限场景：

```text
family-representative PCMCI+；
作为附录级 pressure test；
用于说明在低维代表变量条件下，部分 direct-edge 是否仍可出现；
不作为主筛选工具；
不替代 V1；
不承担 pathway 建立任务。
```

---

## 8. 运行成本与解释成本封口

即使已经知道 no-same-family-controls 或 family-representative 可以部分修复，PCMCI+ 仍不适合作为本项目主工具。

原因：

```text
1. 运行成本高：
   如果要公平使用 PCMCI+，需要多套变量池、多套 conditioning 规则、多窗口、多 tau、多层 FDR 与 tau0/lagged 分层。

2. 解释成本高：
   每个 negative result 都必须判断是物理关系不存在、被同族 controls 吸收、被中介吸收、tau0 主导、窗口太短，还是强共线导致 graph-selection collapse。

3. 问题同构性不足：
   PCMCI+ 识别 conditional direct edge，而本研究关心 field-to-field 关系骨架、窗口接替、路径、反馈、中介与多角色 P。
```

因此：

```text
PCMCI+ 不进入主工具链；
不继续投入大规模修补；
不作为 pathway 输入池的主筛；
保留为方法审计记录和代表变量版本的低维压力测试。
```

---

## 9. 后续主线决策

`lead_lag_screen/V2` 在此封口后，后续主线回到：

```text
lead_lag_screen/V1 smooth5 temporal eligibility
+ AR(1) surrogate max-stat
+ audit null
+ lagged / tau0 / same-day 分层
+ family/window 结果整理
+ 后续 pathway 层重新设计
```

PCMCI+ 只保留为：

```text
V2_a：all-index PCMCI+ 失败/高门槛 lower-bound 记录
V2_b：说明 V2_a 压缩来源的审计记录
V2_c：说明同族 controls 与代表变量池可部分修复 V→P 的病因定位记录
```

---

## 10. 封口结论

最终封口结论如下：

```text
lead_lag_screen/V2 的 PCMCI+ 对照组已经完成工程运行、审计和病因定位。

all-index PCMCI+ 在当前 20 个同源派生指数体系中发生系统性失真：
同族派生指标高度耦合，conditioning 会过度残差化场对象内部共同结构信号，导致核心 V→P 关系在 S3/T3 等关键窗口被异常压没。

敏感性实验表明，去除同族 controls 或改用代表变量池后，V→P 信号恢复，说明 V2_a 的 negative result 不是可靠科学结论，而是方法设定失效。

由于修复 PCMCI+ 需要高运行成本、高解释成本，并且其 conditional direct-edge 语义与本研究的 field-to-field pathway 问题不完全同构，PCMCI+ 不适合作为主工具。

后续唯一可接受的 PCMCI+ 使用方式，是在选出代表变量后，作为低维 family-representative 的附录级 / 压力测试级对照。除此之外，V2 不再继续扩展为主线。
```

