# V7-a field transition timing audit log

## Scope

This run answers only: around accepted transition windows, when does each field show its strongest detector-score transition signal?

It does not infer causality, downstream pathways, or spatial earliest regions.

## Accepted windows

Accepted peak days: [45, 81, 113, 160]
Excluded candidate days: [18, 96, 132, 135]
Bootstrap match threshold: 0.95

## Log evidence excerpts

### V6 UPDATE_LOG excerpt

```text

- 18
- 45
- 81
- 96
- 113
- 132
- 135
- 160

其中 `is_formal_primary` 仅作为注记，不再作为是否进入 bootstrap 主筛的前置门槛。
- **bootstrap 次数：1000**
- **主匹配口径：±5 天**
- **主显著性判断：以 bootstrap 为主**
- **yearwise：仅辅助，不进入主显著性决策**

### 3. 默认主线输出固定回 `mainline_v6_a`

### 4. 点层主结果当前暂定
按 **1000 次 bootstrap + 5 天口径**，当前通过主显著性的点为：

- **45**
- **81**
- **113**
- **160**

当前未作为主路径锚点推进的点：
- 18
- 96
- 132
- 135

其中：

- `132 / 135` 暂不作为当前主路径层重点对象
- 它们保留记录，后续需要时再回看

## 当前主线结论
`V6` 当前已可作为点层主筛结果使用。  
后续路径层默认优先围绕四个已过显著性的点推进：

- 45
- 81
- 113
- 160
```

### V6_1 UPDATE_LOG excerpt

```text
`V6_1` 不承担：

- 第二套点层主显著性裁决
- competition
- parameter-path
初版 `V6_1` 中曾出现：

- `CP005 (113) -> 66–119`

并错误地把 `81` 和 `113` 并进同一窗口。  
已确认原因有两层：

- 修复 merge 的组右边界 bug
- 对已过点层主筛的主点增加保护，不再轻易被误并窗
- 近邻点仍允许共享窗口，因此 `132/135` 仍可自然合窗

### 4. 当前窗口结果已明显合理化
当前已确认：

- `45` 单独成窗
- `81` 单独成窗
- `113` 单独成窗
- `160` 单独成窗
- `132 / 135` 共享一个窗口
- 不确定性层仍然只是窗口属性，没有长成第二套窗口对象

也就是说，`81` 与 `113` 的错误合并已经被修掉。

### 5. 当前窗口主线使用建议
后续路径层当前优先使用的窗口对象为：

- **45 对应窗口**
- **81 对应窗口**
- **113 对应窗口**
- **160 对应窗口**

`132 / 135` 的共享窗当前先保留记录，暂不作为主路径层重点对象。  
其主峰是 `132` 还是 `135`，当前不影响主线推进，后续需要时再回看。

### 6. 五阶段使用原则
当前后续路径层可采用：

- **4 个已过显著性的转换窗口**
- 主线先围绕四个过显著性的转换窗口推进
- 五阶段是由这些窗口切分出来的阶段对象
- 不再让 `132 / 135` 这类未过主显著性的近邻对象干扰主线

## 当前主线结论
`V6_1` 当前已经基本达到“可用于后续路径层时间对象层”的程度。  
当前可先定住的主结构是：

```

## Cross-check

The accepted points are cross-checked against V6 `candidate_points_bootstrap_summary.csv` and V6_1 `derived_windows_registry.csv`.

Logs are treated as evidence to audit, not as automatic truth.