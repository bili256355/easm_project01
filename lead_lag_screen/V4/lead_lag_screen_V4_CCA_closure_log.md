# lead_lag_screen/V4 CCA 封口日志

## 0. 文档性质

本文档是 `lead_lag_screen/V4` 的**封口日志**，不是普通结果摘要，也不是新的科学结论稿。

本日志用于明确：

1. V4-CCA 的工程运行状态；
2. V4-CCA 实际回答了什么问题；
3. V4-CCA 不能回答什么问题；
4. V4-CCA 为什么不能作为路径成立判据；
5. V4-CCA 在后续研究中的保留地位；
6. 后续主线应如何回到 V1 lead–lag 与 V3 EOF-PC1 审计结果。

---

## 1. V4 的原始目的

### 1.1 背景

在 V1 smooth5 指数 lead–lag 结果中，T3 / 梅雨结束附近出现了明显弱化特征：

- `Tier1 = 0`；
- V→P 只剩少量 Tier2；
- H→P 在指数层不明显；
- same-day / ambiguous 结构增强。

这带来一个关键疑问：

> T3 / 梅雨结束附近的弱化，是真实的场—场关系崩溃，还是人工指数在该阶段适用性下降导致的表征偏差？

V3 EOF-PC1 已经表明：T3 在 PC1 主模态层仍然存在较强 V→P、H→P、H→V 等耦合，但 PC1 只是单场最大方差模态，不一定是两个场之间最相关的模态。

因此引入 V4-CCA 的初衷是：

> 不再只看人工指数或单场 PC1，而是直接寻找两个场之间的最优耦合模态，审计 T3 / 梅雨结束附近核心场对是否仍存在场间耦合。

---

## 2. V4 的工程实现定位

### 2.1 版本定位

```text
lead_lag_screen/V4
field_cca_smooth5_v4_a
```

V4 的定位是：

```text
5日 smooth5 anomaly field 的窗口级 lagged CCA 场间耦合审计层
```

它不是：

```text
pathway 层；
因果推断层；
路径成立判据；
V1 lead–lag 的替代版本；
PCMCI+ 的替代版本。
```

### 2.2 输入数据

V4 使用：

```text
foundation/V1/outputs/baseline_smooth5_a/preprocess/anomaly_fields.npz
```

即 5 日平滑 anomaly 场。

### 2.3 核心对象

V4 只对核心场对运行 CCA：

```text
V → P
H → P
H → V
Jw → Je
Je → H
```

### 2.4 方法概要

V4 对每个窗口、每个核心 pair 执行：

1. 对 source field 与 target field 分别做 EOF 降维；
2. 主结果使用 `k=5` EOF scores；
3. 敏感性使用 `k=3` EOF scores；
4. 对 lag = 0..5 分别做 CCA；
5. 输出 cross-validated canonical correlation；
6. 输出 permutation / bootstrap 稳定性结果；
7. 将时间结构分为：
   - `tau0_dominant`
   - `lagged_dominant`
   - `lagged_tau0_close`
   - `weak_or_unstable`

---

## 3. V4 的运行事实

### 3.1 工程运行状态

V4 结果包显示：

```text
status = success
fallback_used = false
input = baseline_smooth5_a / anomaly_fields.npz
windows = 9
core pair directions = 5
k_eof = 5 主结果，k_eof = 3 敏感性
CCA lag = 0..5
```

工程层面可以确认：

> V4 已进入执行链并成功运行，输出完整，不是 fallback 结果。

### 3.2 主结果规模

主结果 `k=5`：

```text
9 个窗口 × 5 个核心方向 = 45 条 CCA pair summary
```

45 条主结果全部被判为：

```text
CCA_strong_coupling
```

这说明 V4 在当前设定下对核心场对的耦合非常敏感。

---

## 4. V4 的主要结果事实

### 4.1 时间结构总体分布

`k=5` 主结果中：

```text
lagged_tau0_close = 29
tau0_dominant = 12
lagged_dominant = 4
```

因此 V4 的总体结果不是“所有关系都是纯同期”，也不是“多数关系是清楚滞后”，而是：

> 核心场对普遍存在强耦合，但 tau0 与短滞后 lag1/lag2/lag3 往往非常接近，方向与时滞难以干净拆分。

### 4.2 V→P 的窗口表现

V→P 在所有窗口均表现为强耦合。

典型结果：

```text
S1: tau0 = 0.886, lag1 = 0.861
T1: tau0 = 0.929, lag1 = 0.851
S2: tau0 = 0.910, lag1 = 0.865
T2: tau0 = 0.880, lag1 = 0.890
S3: tau0 = 0.879, lag1 = 0.828
T3: tau0 = 0.827, lag1 = 0.786
S4: tau0 = 0.841, lag1 = 0.791
T4: tau0 = 0.863, lag1 = 0.801
S5: tau0 = 0.902, lag1 = 0.809
```

关键判断：

> V4-CCA 不支持任何窗口中 V–P 场间耦合崩溃的说法。T3 中 V–P 仍然很强，只是 tau0 与 lag1 接近，且 tau0 略占优。

### 4.3 T3 / 梅雨结束窗口

T3 是本轮 V4 最重要的审计窗口。

`k=5` 下：

```text
V→P:  tau0 = 0.827, lag1 = 0.786, close
H→P:  tau0 = 0.793, lag1 = 0.786, close
H→V:  tau0 = 0.751, lag1 = 0.734, close
Jw→Je: tau0 = 0.572, lag1 = 0.586, close
Je→H: tau0 = 0.886, lag1 = 0.862, close
```

T3 的核心特征是：

> 所有核心场对都存在较强耦合，但几乎全部表现为 tau0 与短滞后 close，而不是清楚的单向滞后传播。

因此，V4 对 T3 的正确信息是：

```text
T3 不是场—场耦合消失窗口；
T3 是强近同步耦合 / 快速调整 / 方向拆分困难窗口。
```

### 4.4 Jw→Je 的特殊性

V4 中真正较像 `lagged_dominant` 的方向主要是 Jw→Je：

```text
S1: lagged_dominant
T1: lagged_dominant
S2: lagged_dominant
T2: lagged_dominant
S3–S5: 多转为 lagged_tau0_close
```

这与 V1 和 V2 中 Jw→Je 相对“干净”的现象一致。

但是这仍不能写成路径成立，只能说：

> 在 CCA 耦合模态层面，Jw→Je 是核心 pair 中最具有短滞后结构的方向。

---

## 5. V4 的有效贡献

V4 的有效贡献是有限但明确的。

### 5.1 能支持的判断

V4 可以支持以下判断：

1. 核心场对之间普遍存在强耦合模态；
2. T3 / 梅雨结束附近并非场—场关系完全崩溃；
3. V→P 在所有窗口都存在强场间耦合；
4. H→P 在 T3 也存在强场间耦合；
5. T3 的主要特征是强近同步耦合，而不是无关系；
6. V1 指数层 T3 弱化不能直接解释为真实场关系消失；
7. V3 PC1 审计与 V4 CCA 审计共同支持“指数适用性 / 方向拆分 / 同期主导”解释。

### 5.2 最适合保留的表述

V4 最适合保留为：

```text
场间耦合存在性审计；
指数弱化是否等于场关系消失的反证层；
T3 / 梅雨结束附近强近同步耦合的辅助证据。
```

---

## 6. V4 不能支持的判断

V4 不能支持以下判断：

1. V→P 因果路径成立；
2. H→P 因果路径成立；
3. Jw→Je 是完整路径主轴；
4. 某条 pathway 已经建立；
5. 某个变量是 mediator；
6. 某个窗口中某方向具有可分辨的单向传播；
7. CCA 支持的高相关等于物理路径成立；
8. V4 可以替代 V1 lead–lag；
9. V4 可以作为后续 pathway 候选池的主筛选工具。

---

## 7. 自相关与 null 的限制

### 7.1 当前 V4 做了什么控制

V4 使用了以下设计：

```text
5日 anomaly fields
cross-validation by year
year-block permutation of target field
preserve within-year day sequence
bootstrap by year block
max over lag = 0..5
```

这些设计可以避免部分明显问题：

```text
不跨年拼接；
不把同一年同时用于训练和测试；
不随机破坏年内 day sequence；
不把普通训练内 CCA correlation 直接当最终值。
```

### 7.2 当前 V4 没有做什么控制

V4 没有像 V1 一样显式进行：

```text
AR(1) surrogate max-stat
effective autocorrelation adjustment
phase-randomized surrogate
circular shift null
block-shift null
严格 lag0-vs-lagged 差异检验
```

因此必须明确：

> V4 没有充分排除时间自相关、共同季节演变或强同步结构对 CCA 高相关的贡献。

这意味着：

> V4-CCA 不能作为路径成立或方向成立的统计检验。

### 7.3 为什么这对 CCA 特别重要

CCA 是“为最大相关而生”的方法。

在当前问题中，P/V/H/Je/Jw 等场对象共享明显的季节内演变、低频背景和同步调整结构。即使使用 anomaly 与 year-block permutation，CCA 仍然很容易找到强耦合模态。

因此，V4 的高相关不能直接解释为：

```text
路径成立；
因果方向成立；
某对象独立驱动另一个对象。
```

只能解释为：

```text
两个场之间存在可被 CCA 提取的强耦合模态。
```

---

## 8. 为什么 V4 不适合作为路径工具

V4 不适合作为路径工具，原因有四个。

### 8.1 分辨率太低

V4 中 45 条核心主结果全部为 strong coupling。  
这意味着它几乎不能区分“哪些核心 pair 不成立”。

对于路径筛选来说，一个工具如果几乎把所有核心场对都判为强耦合，其筛选分辨率不足。

### 8.2 同期 / 短滞后高度接近

多数结果是：

```text
lagged_tau0_close
```

而不是清楚的：

```text
lagged_dominant
```

这说明 V4 很难拆出明确时间方向。

### 8.3 CCA 不直接给因果方向

CCA 本质上寻找两个场之间最大相关的线性组合，不是因果方向检验。

即使做 `X(t-lag) vs Y(t)`，结果也仍然是耦合模态强弱，不等于路径方向成立。

### 8.4 自相关控制不足

当前 V4 的 null 没有达到 V1 AR(1) surrogate max-stat 的时间结构控制水平，因此不能用来替代 V1 的时间资格判定。

---

## 9. V4 的最终封口定性

V4 的最终定性为：

```text
已运行；
结果完整；
证明核心场对之间普遍存在强耦合模态；
支持 T3 / 梅雨结束不是场间关系崩溃；
但分辨率太低，不能用于路径成立判据；
当前版本未充分排除时间自相关影响；
不作为后续 pathway 主筛选工具。
```

更简洁地说：

> V4-CCA 是有效的“场间耦合存在性审计”，不是有效的“路径成立检验”。

---

## 10. 与 V1 / V3 的关系

### 10.1 V1

V1 仍是当前主线：

```text
5日人工指数 lead–lag temporal eligibility screen
```

V1 的作用是：

```text
给出窗口 × family × 指数层的时间资格与 same-day coupling 分层。
```

V1 不能被 V4 替代。

### 10.2 V3

V3 是：

```text
EOF-PC1 主模态审计层
```

V3 说明：

```text
T3 在 PC1 层仍有强 V→P / H→P / H→V 等耦合；
T3 更像 same-day dominant / 方向拆分困难，而不是无关系。
```

### 10.3 V4

V4 是：

```text
CCA 场间最优耦合模态审计层
```

V4 进一步说明：

```text
即使不局限于 PC1，核心场对在 T3 仍有很强耦合模态；
但由于 CCA 分辨率过低，它不能证明路径。
```

最终三者关系：

```text
V1 = 主时间资格层
V3 = 指数适用性 / PC1 主模态审计
V4 = 场间耦合存在性审计
```

---

## 11. 后续使用规则

### 11.1 可以使用 V4 的方式

后续可以引用 V4 来支持：

```text
T3 并非场间耦合消失；
V1 T3 弱化不应被解释为真实崩溃；
梅雨结束附近是强近同步耦合 / 快速调整窗口；
核心场对之间存在普遍耦合模态。
```

### 11.2 禁止使用 V4 的方式

后续禁止将 V4 写成：

```text
路径成立；
因果链成立；
V→P 直接驱动成立；
H→P 直接驱动成立；
Jw→Je 是主 pathway；
V4 比 V1 更高级、更真实；
CCA 已排除自相关后证明路径。
```

### 11.3 后续是否继续修 V4

不建议继续把 V4 修成主路径工具。

若继续增加 AR(1)、phase randomization、circular shift、block surrogate、lag0-vs-lagged 差异检验等，V4 仍然主要回答“场间耦合模态是否存在”，而不会自然变成 pathway 方法。

因此：

```text
V4 在当前阶段封口；
不继续扩展为主工具；
只作为审计层保留。
```

---

## 12. 最终封口结论

最终封口结论如下：

> `lead_lag_screen/V4` 的 field-CCA 审计已成功运行，并显示核心场对在所有窗口普遍存在强耦合模态。尤其 T3 / 梅雨结束附近，V–P、H–P、H–V、Je–H 等关系仍然很强，说明该窗口不能解释为场—场关系崩溃。V4 支持的是“强近同步耦合 / 快速调整 / 方向拆分困难”的解释，而不是“关系消失”。  
>
> 但 CCA 对场间共同模态极其敏感，当前版本没有像 V1 那样显式排除 AR(1) 自相关影响，且 45 条核心结果全部为 strong coupling，筛选分辨率不足。因此 V4 不能作为路径成立、方向成立或因果链成立的证据。V4 只保留为场间耦合存在性审计和指数适用性反证材料；后续主线仍回到 V1 smooth5 lead–lag，辅以 V3 EOF-PC1 和 V4 CCA 的表达适用性审计结论。
