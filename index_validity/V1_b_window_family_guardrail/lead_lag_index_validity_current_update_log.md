# lead_lag_screen / index_validity 当前状态更新日志

## 0. 日志用途

本日志用于记录近期围绕 **5 日 smooth5 lead–lag 主线可信度** 所做的一组方法审计、反证测试和指数有效性守门测试的当前结论。

本日志不是普通结果摘要，而是用于后续接管时防止误判：

- 不把 `lead_lag_yes` 写成物理机制成立；
- 不把指数阶段敏感性误写成指数错误；
- 不把 PCMCI+ / CCA 的负面或宽泛结果误用为主线判据；
- 不把 T3 / 梅雨结束附近的指数层弱化误写成场—场关系真实崩溃。

---

## 1. 当前主线状态

当前主工作仍然应回到：

```text
lead_lag_screen/V1
5日 smooth5 index lead–lag temporal eligibility screen
```

其科学地位是：

```text
时间资格层 / temporal eligibility screen
```

而不是：

```text
物理机制证明
因果路径证明
中介链成立证明
```

经过近期 PCMCI+、EOF-PC1、CCA、index_validity 守门审计后，当前判断是：

> 5 日 V1 lead–lag 作为 temporal eligibility 层的可信度进一步提高。  
> 虽然人工指数存在阶段适用性和局部敏感性，但目前没有证据支持“某个窗口某个对象族的指数全体失效”。  
> 因此，V1 的主体结果不能简单归因于指数错误或场间关系消失。

---

## 2. PCMCI+ V2 封口状态

### 2.1 V2_a all-index PCMCI+

V2_a 已经跑通，但结果被判为不适合作为科学对照主结果。

主要问题：

```text
all-index PCMCI+ 在 20 个同源派生指数体系中发生强烈过度残差化；
V→P 被异常压没；
S3/T3 等关键窗口中 V→P 在 lagged 和 tau0 中均表现异常；
这与基本物理认知、V1、V3、V4 均不一致。
```

### 2.2 V2_b / V2_c 诊断

V2_b 审计显示：

```text
压缩不是单纯由 FDR 造成；
PCMCI+ graph selection 本身已经极度收缩；
大量 V1 lead_lag_yes 未进入 PCMCI+ lagged/tau0 signal。
```

V2_c 敏感性显示：

```text
去掉同族 controls 后，S3 V→P 明显恢复；
T3 V→P 至少 tau0 恢复；
family-representative 低维变量池下，V→P 在多数窗口恢复为 tau0 或部分 lagged。
```

### 2.3 V2 封口结论

PCMCI+ 不再作为主工具。

可保留的唯一有限使用方式是：

```text
family-representative 低维变量池下，
作为附录级 / 压力测试级 direct-edge lower-bound 对照。
```

all-index PCMCI+ 的 negative result 不能用于否定 V1 的 V→P、H→P 等关系。

---

## 3. EOF-PC1 V3 状态

V3 使用：

```text
5日 smooth5 anomaly fields
窗口内 EOF-PC1
每个对象只取 PC1
```

V3 的价值是审计：

```text
V1 中 T3 / 梅雨结束附近的指数层弱化，
是否可能来自人工指数阶段适用性下降，
而不是场—场关系真实崩溃。
```

主要结论：

```text
T3 中 V_PC1→P_PC1 仍然很强；
H_PC1→P_PC1 在 T3 也明显出现；
但大量关系表现为 same-day dominant / direction ambiguous。
```

V3 不支持：

```text
T3 场—场关系真实崩溃
```

更合理的表述是：

```text
T3 是强近同步耦合与方向拆分困难窗口；
V1 中 T3 的 Tier1 崩掉，不能直接解释为真实关系消失。
```

限制：

```text
P_PC1 解释率整体偏低；
PC1 不能代表全部降水结构；
V3 不是 pathway 证明。
```

---

## 4. CCA V4 封口状态

V4 使用：

```text
5日 smooth5 anomaly fields
窗口级 EOF 降维
核心场对 lagged CCA
```

运行结果显示：

```text
V→P、H→P、H→V、Jw→Je、Je→H 等核心场对几乎全部表现为 strong coupling；
多数关系为 tau0 / lag1 close；
少数 Jw→Je 更接近 lagged dominant。
```

V4 的价值：

```text
证明核心场对在 T3 / 梅雨结束附近没有场间耦合崩溃；
支持“V1 指数层弱化不等于场关系消失”。
```

但 V4 已封口为：

```text
场间耦合存在性审计
```

不能作为：

```text
路径成立判据
方向成立判据
因果链判据
```

原因：

```text
CCA 分辨率太低；
容易把核心场对都判为强耦合；
当前版本也没有像 V1 一样显式使用 AR(1) surrogate max-stat 排除自相关。
```

---

## 5. index_validity/V1_b 窗口—对象族守门审计

### 5.1 主审计口径

经修正后，index_validity 主审计口径应为：

```text
smoothed field + smoothed index values
```

而不是 anomaly。

原因：

```text
index_validity 关心的是指数对其自身场对象结构的指示性；
应首先检查指数是否指示平滑场本身；
anomaly 只能作为附加口径，不应作为主审计口径。
```

### 5.2 单指数窗口指示性结果

`index_window_representativeness.csv` 显示：

```text
strong = 155
moderate = 24
weak_but_usable = 1
high_risk = 0
not_supported = 0
```

`window_family_guardrail.csv` 显示：

```text
45 个 window × family 全部为 family_collapse_risk = low
partial_sensitivity = 0
high = 0
```

因此：

> 没有任何窗口、任何对象族出现“全族指数失效”。

### 5.3 T3 结果

T3 尤其关键。结果显示：

```text
T3 × P：7 个 P 指数全部 strong
T3 × V：3 个 V 指数全部 strong
T3 × H：3 strong + 1 moderate
T3 × Je：2 strong + 1 moderate
T3 × Jw：2 strong + 1 moderate
```

这说明：

> T3 / 梅雨结束附近没有 P/V/H/Je/Jw 任一对象族全体指数失效。

---

## 6. 单指数 field R² 的窗口变化

单指数 field R² 显示：

```text
P 族整体 R² 偏低，但 T3 不是低点；
V 族 R² 全季最高，T3 有下降但不崩溃；
H 族平均 R² 在 S3/T3/S4 较低，但主要受 width/extent 类指数影响；
Je/Jw 族整体稳定，Jw 中后期反而增强。
```

关键判断：

> 从单指数 R² 看，指数代表性存在阶段变化和类型差异，但没有 window × family 层面的整体崩溃。  
> T3 尤其不是 R² 失效点。

---

## 7. 同族指数联合覆盖度结果

新增 `window_family_joint_field_coverage.csv` 用来回答：

```text
同一对象族多个指数合起来，能覆盖窗口内该场结构的多少？
```

### 7.1 总体分布

```text
very_high_joint_coverage = 4
high_joint_coverage = 34
moderate_joint_coverage = 7
low / high risk = 0
```

`collapse_risk_update` 全部为：

```text
no_family_collapse_update_supported = 45 / 45
```

### 7.2 T3 联合覆盖度

T3 的 joint coverage 全部为 high：

```text
P  : joint_field_R2_year_cv = 0.235, EOF top5 coverage = 0.558
V  : joint_field_R2_year_cv = 0.413, EOF top5 coverage = 0.534
H  : joint_field_R2_year_cv = 0.468, EOF top5 coverage = 0.527
Je : joint_field_R2_year_cv = 0.368, EOF top5 coverage = 0.445
Jw : joint_field_R2_year_cv = 0.434, EOF top5 coverage = 0.496
```

这说明：

> T3 中各对象族不是靠单个指数勉强支撑，而是多个指数联合后能覆盖相当一部分场结构。  
> T3 不存在混合指示程度崩溃。

### 7.3 P 族特别说明

P 的单指数 R² 不高，但联合覆盖明显增强：

```text
P 族在 S2 / T2 / S3 / T3 的 EOF top5 coverage 较高；
T3 P top5 coverage = 0.558；
S5 P 是相对最低点，但仍为 moderate_joint_coverage。
```

这说明：

```text
P 的各个指数分别覆盖降水结构的不同分量；
单指数不惊艳，但组合起来不能说错，也不能说失效。
```

---

## 8. 当前对指数体系的最新判断

当前不能说：

```text
指数完美
每个指数都很强
指数能完整替代原始场
```

但可以说：

```text
指数体系没有出现 window × family 全族失效；
单指数虽有阶段敏感性，但联合覆盖度支持对象族整体仍有场指示能力；
T3 / 梅雨结束附近没有指数族崩溃；
P 的单指数解释度不惊艳，但多指数联合覆盖度可接受；
V/H/Je/Jw 的联合覆盖度整体更稳。
```

一句话总结：

> 指数选取并不完美，也不是“惊艳”的全场解释器；  
> 但它们作为对象族的多指标表达是可用的，不能说是错的。  
> 当前证据支持“局部阶段敏感性”，不支持“全族指数失效”。

---

## 9. 对 V1 lead–lag 主线的影响

近期审计后，对 V1 的判断应更新为：

```text
V1 smooth5 lead–lag 作为 temporal eligibility 层的可信度增强；
V1 结果不能简单归因于 PCMCI+ 反证、指数全族失效、或场间关系消失；
但 V1 仍不能直接证明物理机制或因果 pathway。
```

更精确的表述：

> V1 的主体结构，尤其 V→P 全季主干、H→P 次级关系、P feedback 与 nonP internal 候选，已经通过多轮方法审计获得更高可信度。  
> 但这些仍是时间资格和相关结构，不是路径成立终点。

---

## 10. 当前推荐保留的主线结构

后续建议采用：

```text
主线：
lead_lag_screen/V1 smooth5 index lead–lag

底座守门：
index_validity/V1_b smoothed window-family guardrail
index_validity/V1_b joint family coverage

辅助审计：
lead_lag_screen/V3 EOF-PC1
lead_lag_screen/V4 CCA

封口 / 失败对照：
lead_lag_screen/V2 all-index PCMCI+
```

其中：

```text
V2 不作为主工具；
V3/V4 不作为 pathway 证明；
index_validity 只守门指数对自身场的指示性；
V1 仍是后续 pathway 候选池的主要 temporal eligibility 来源。
```

---

## 11. 当前不能贸然推进的部分

仍然不能直接声称：

```text
lead_lag_yes = causal path
Tier1/Tier2 = mechanism established
V→P 一定是直接驱动路径
H→P 一定是直接路径
P feedback 一定是反馈机制成立
```

下一步如果进入 pathway，需要新增：

```text
direct / indirect 分解
mediator 检验
反馈与同日耦合分层
窗口内路径关系重建
```

---

## 12. 当前阶段总判断

当前阶段可以封存为：

> 经 PCMCI+ 失败对照、EOF-PC1 指数适用性审计、CCA 场间耦合审计，以及 index_validity 的单指数与联合覆盖度守门测试后，5 日 V1 lead–lag 结果作为 temporal eligibility 层的可信度显著增强。  
> 指数体系不是完美的，也并不对所有场结构有很高单指数解释度，但没有出现任何窗口对象族全体失效。  
> 单个指数可能阶段敏感，尤其宽度、范围和边界类指数；但对象族联合覆盖度整体可接受，T3 也没有崩溃。  
> 因此，V1 结果可以继续作为后续 pathway 重建的候选时间资格底座，但不能单独作为物理机制或因果路径成立的证据。
