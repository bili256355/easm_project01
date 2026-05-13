# UPDATE_LOG_V6.md

## 版本定位
`V6` 定位为：**点层初筛主线**。  
职责是对 baseline detector 检出的全部候选点做 bootstrap 主筛，并给出 yearwise 辅助信息。  
当前不承担：

- window 判定
- competition
- parameter-path
- final judgement

## 关键更新

### 1. 候选点集合改为 baseline detector 全部 local peaks
不再只围绕 formal primary 进行主筛。  
当前 baseline candidate registry 共 8 个点：

- 18
- 45
- 81
- 96
- 113
- 132
- 135
- 160

其中 `is_formal_primary` 仅作为注记，不再作为是否进入 bootstrap 主筛的前置门槛。

### 2. bootstrap 口径正式收口
当前主线合同已定为：

- **bootstrap 次数：1000**
- **主匹配口径：±5 天**
- **主显著性判断：以 bootstrap 为主**
- **yearwise：仅辅助，不进入主显著性决策**

### 3. 默认主线输出固定回 `mainline_v6_a`
后续 `V6` 主链默认输出使用：

- `mainline_v6_a`

此前因补丁过程出现过与 `6-b` 实验线输出/配置混线的问题，已修正。  
当前 `6-a` 与 `6-b` 在配置和输出上应视为已隔离。

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

### 5. 6-b 实验层保留，但不进入当前主线决策
`V6-b selection-frequency` 已经作为**并行实验层**建立。  
其作用是提供：

- 时间轴上的 bootstrap 选中频率曲线
- 区间级存在性参考

但当前主线不以它替代 `V6-a` 点级 bootstrap 主表。  
当前推进路径层时，仍以 `V6-a` 为准。

## 当前主线结论
`V6` 当前已可作为点层主筛结果使用。  
后续路径层默认优先围绕四个已过显著性的点推进：

- 45
- 81
- 113
- 160
