# foundation/V1 VERSION_POLICY

## 版本角色
这是基础层第一正式版本，只负责：
- preprocess
- indices

## 冻结边界
- 不写入任何阶段划分输出
- 不写入任何路径层输出
- 不引入窗口检测或分段逻辑

## 依赖
- 原始输入目录：`D:\工作目录\data\my_data`

## 对外输出
- `preprocess/`
- `indices/`

这些输出将作为后续 `stage_partition/*` 和 `pathway/*` 的候选输入底座。
