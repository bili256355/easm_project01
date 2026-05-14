# ROOT_LOG_04_FAILED_NEGATIVE_UNRELIABLE_RECORDS

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_04_FAILED_NEGATIVE_UNRELIABLE_RECORDS.md`

## 0. 本日志的性质

这是失败记录、负结果、不可信结果与被降级路线的专门日志。

本项目的原则是：

> 负结果、失败结果、不可信结果不是无价值内容。只要原因、层次和风险被分解清楚，它们本身就是科研接管的重要内容。

本日志不能删除；后续新会话接管时必须优先阅读。

---

## 1. stage_partition V1 废弃

状态：已废弃。

失败层：对象定义 / stage state 构造。

原因：

- V1 将 anomaly 混入 stage-defining state；
- 这被判定为对象定义错误；
- 后续阶段分割主线转向 V2 raw smoothed field profiles。

当前处理：

- V1 不作为当前 stage partition 主线依据；
- 保留为失败记录；
- 后续不得重新把 anomaly state 当作阶段定义底座。

价值：

- 明确了 stage-defining state 应来自 raw smoothed field profiles，而不是 anomaly 混合状态。

---

## 2. 旧 V9 sensitivity / v1 / v1_1 判定废除

状态：已废弃，不可作为当前依据。

失败层：方法理解 + 工程语义复现 + 结果解释。

原因：

- 没有严格复刻 V9/V10 peak 语义；
- 把 H day18/19 误当污染；
- 用 fixed window 排除 H day35 等前缘候选；
- 对对象间 order 的旧判断不可保留。

当前处理：

- 旧 sensitivity / v1 / v1_1 的“次序变化”判断不可继续使用；
- 当前以 V10、V10.1、V10.2 语义复现和对象峰谱系为底座；
- 对象间 order 以后以 V10.4 为正式审计。

价值：

- 暴露了旧方法没有对齐 peak/subpeak 语义的问题；
- 促成 V10、V10.1、V10.2 的重建。

---

## 3. V10.5_a top1 验证初判失败

状态：已修正。

失败层：验证设计 / 结果解释。

问题：

- V10.5_a 只看 profile-energy top1；
- 当 top1 抓到另一个已知候选峰族时，把 V10.4 assigned peak 误判为 `NOT_SUPPORTED`；
- 典型冲突包括 W045-H、W113-P、W160-Jw。

修正：

- V10.5_b：top-k candidate-family validation；
- V10.5_c：selection-reason audit；
- V10.5_d：strength–stability audit；
- V10.5_e：full-season strength curve export。

当前处理：

- V10.5_a 原始 `NOT_SUPPORTED` 不能单独引用；
- top1 mismatch 不等于 assigned peak 不支持；
- 应解释为 candidate-family competition。

价值：

- 暴露了 energy-dominant family 与 lineage-assigned family 可能分离；
- 促成多峰族框架。

---

## 4. low-dimensional profile metrics 支持弱

状态：负结果，必须保留。

失败/负结果层：指标抽象。

现象：

- profile-energy 有较多支持；
- low-dimensional profile metrics 支持不强。

当前解释：

- 当前低维指标偏粗，尚不能稳定承载 profile-level 形态变化；
- 不能因此否定 profile-energy；
- 也不能把当前低维指标当成强物理佐证。

后续要求：

- 需要发展更正式的 object physical metrics；
- 例如 H_strength、H_centroid_lat、Jw_axis_lat、Jw_strength、P band share、P centroid、P spread 等对象物理量。

价值：

- 提醒后续不能用粗指标草率替代物理解释。

---

## 5. 用户批评“未查验就解释”

状态：交互失败记录，必须保留。

失败层：审计流程 / 解释习惯。

问题：

- 曾过快顺着用户观察解释曲线；
- 未先核对图像和表格；
- 部分结论如“先行都是弱先行”“V 造成 W081 不干净”过度泛化。

修正：

- 后续必须先查验图、表、代码、输出；
- 必须先判断用户说法是否成立，再分析为什么，再执行；
- 图像层判断必须标记为图像层，不能自动写成量化结果。

价值：

- 强化后续接管协议；
- 防止顺从式解释造成科研失真。

---

## 6. PCMCI+ 路线的限制与负结果

状态：辅助/对照，不能作为主发现。

问题：

- PCMCI+ 找到的部分边与直接 lead-lag 直觉不一致；
- 曾出现 odd Je→Jw 等关系；
- 强同族/同对象内部耦合可能让 residualized direct links 变弱；
- PCMCI+ 更适合作为 secondary validation blurb，而不是主发现路线。

当前处理：

- PCMCI+ 不替代 lead-lag 主线；
- 不替代 V10 peak/order 主线；
- 若未来使用，需要明确变量代表、同族传递限制、直接边解释边界。

价值：

- 防止后续迷信条件独立图方法；
- 保留其作为辅助验证的可能性。

---

## 7. LPCMCI 计算成本过高

状态：降级/放弃。

问题：

- 计算成本过高；
- 难以作为当前主线可运行方法；
- 不适合在当前工程中默认推进。

当前处理：

- 不作为主线；
- 后续除非有明确计算资源与降维策略，否则不优先恢复。

价值：

- 防止后续再次把高成本低收益路线当优先方向。

---

## 8. EOF/PC1 / CCA 对照路线的限制

状态：辅助/对照，不作为主发现。

问题：

- EOF/PC1 提供 coarse mode，分辨率较低；
- CCA / EOF-PC1 容易出现“很多关系都显著”的现象；
- 对路径推断帮助有限；
- bootstrap 计算可能极慢。

当前处理：

- 只能作为粗模态对照；
- 不能替代高分辨率对象变量和窗口级分析；
- 不能作为 pathway 主发现。

价值：

- 防止后续用低维粗模态掩盖对象级变化。

---

## 9. Liang information flow 试验错误

状态：负结果 / 方法风险。

问题：

- 试验中曾出现 degenerate covariance 错误；
- 当前没有形成稳定可用主线。

当前处理：

- 暂不作为当前主线；
- 若未来恢复，必须先解决退化协方差和适用条件。

---

## 10. T3/T4 V→P 弱化与 surrogate null 过滤

状态：重要负结果，继承保留。

现象：

- T3 / T4 中 V→P lead-lag 关系显著弱化；
- 大量关系主要被 surrogate null 过滤；
- V1_1 改 window length / 加 indices 后未恢复 T3 信号。

当前处理：

- 不能删除；
- 不能解释为“没有价值”；
- 应作为 EASM 季节推进中关系重组的重要负结果保留。

后续要求：

- 若回到 lead-lag/pathway 主线，必须重查 T3/T4 过滤原因；
- 需要结合空间场解释：S3–T3–S4 相对变化、V 对 P 支持区域是否转移、SCS/southern vs central/northern support。

---

## 11. path diagnosis 历史问题

状态：历史主线，部分结果可参考，部分机制未成熟。

已知问题：

- v1_6_20a 有 structured skeleton，但 no coupled/ordered；
- v1_6_25 打开 coupled layer，但 short/long chain 处理偏宽；
- v1_6_26 只对 short transitions 形成 ordered propagation；
- v1_6_27 增加 prioritization metrics，但 long-chain vs short-chain independence 尚未成熟；
- competitor selection ties、strongest-candidate promotion side-track、mirror-chain 等均需谨慎。

当前处理：

- path diagnosis 不能被 V10 peak/order 分支替代；
- 也不能未经重接直接拿早期 path 结果解释 V10 多峰族；
- 未来需要正式建立 V10 peak/order 与 pathway 变量/窗口/路径之间的映射。

---

## 12. “最强候选”路线风险

状态：降级 / 警惕。

问题：

- 只选 strongest candidate 容易掩盖真实的多峰族结构；
- 容易把 energy-dominant peak 误当主窗口 peak；
- 容易删除 secondary-supported、weak precursor、non-strict but meaningful candidates。

当前处理：

- 不再默认以“最强候选”作为主裁决；
- 必须同时看 lineage、support、stability、window context、candidate family role。

---

## 13. 当前失败记录总原则

后续新会话必须遵守：

1. 失败记录不能删；
2. 负结果不能因为“不好看”而压缩；
3. 不可信结果要说明为什么不可信；
4. 被修正结果要保留修正链；
5. 已废弃路线不能重新偷渡进主线；
6. 任何新解释必须先核对输出、图、表、代码。
