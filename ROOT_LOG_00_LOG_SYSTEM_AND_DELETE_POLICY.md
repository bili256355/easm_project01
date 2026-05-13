# ROOT_LOG_00_LOG_SYSTEM_AND_DELETE_POLICY

生成日期：2026-05-13
建议放置位置：`D:\easm_project01\ROOT_LOG_00_LOG_SYSTEM_AND_DELETE_POLICY.md`

## 0. 本日志的性质

这是一份**根目录日志制度与旧日志删除策略**。它不是科研结果日志，也不是完整工程验收报告。

本轮整理的基本事实边界：

1. 本日志体系基于：
   - 会话迁移包；
   - GitHub 仓库公开页面可见结构；
   - 当前会话中已明确的用户工程治理约束。
2. 当前整理者不能直接修改你的 GitHub 仓库，也没有完成本地全量运行级验收。
3. 因此，本日志体系的定位是：
   - **根目录接管底稿**；
   - **旧日志删除前的保留框架**；
   - **后续新会话接管时的首要入口**。
4. 它不是以下内容：
   - 不是对所有旧日志真伪的最终裁决；
   - 不是所有输出结果的逐表验收；
   - 不是物理机制结论；
   - 不是替代源码、输出表、run_meta 的唯一证据。

---

## 1. 未来日志统一规则

从本轮开始，建议采用以下硬规则：

> **所有人工维护型日志、接管日志、研究状态日志、工程状态日志、版本封口日志，默认全部写入工程根目录 `D:\easm_project01`。**

不再建议在以下位置继续新增人工维护型日志：

- `stage_partition/V*/UPDATE_LOG_*.md`
- `stage_partition/V*/logs/*.md`
- `lead_lag_screen/V*/UPDATE_LOG_*.md`
- `foundation/V*/logs/*.md`
- `outputs/` 下的解释性日志

允许继续存在但不作为主接管日志的文件：

- `README.md`：子版本使用说明；
- `STRUCTURE_MANIFEST.txt`：结构说明；
- `VERSION_POLICY.md`：版本制度；
- `outputs/*.csv`、`outputs/*.json`、`outputs/*.md`：结果或结果说明；
- `run_meta*`：运行元信息；
- `scripts/`、`src/`：代码承载。

---

## 2. 本轮根目录日志体系

本轮建议保留 6 份主日志 + 1 份旧日志索引：

1. `ROOT_LOG_00_LOG_SYSTEM_AND_DELETE_POLICY.md`
   - 日志制度；
   - 旧日志删除边界；
   - 后续维护规则。

2. `ROOT_LOG_01_CURRENT_RESEARCH_STATE.md`
   - 当前科研状态总控；
   - 当前主线、边界、解释层收缩；
   - V10 分支与整个 EASM 项目的关系。

3. `ROOT_LOG_02_ENGINEERING_INVENTORY_AND_ENTRYPOINTS.md`
   - 工程目录索引；
   - 主要模块、入口、输出目录；
   - 工程承载与科研角色的边界。

4. `ROOT_LOG_03_VERSION_STATUS_AND_EXECUTION_CHAIN.md`
   - 各分支版本状态；
   - 入口、输出、执行链、可用性；
   - “讨论 / 实现 / 执行链 / 运行结果”分层。

5. `ROOT_LOG_04_FAILED_NEGATIVE_UNRELIABLE_RECORDS.md`
   - 失败记录；
   - 负结果；
   - 不可信结果；
   - 已废弃或降级路线。

6. `ROOT_LOG_05_PENDING_TASKS_AND_FORBIDDEN_INTERPRETATIONS.md`
   - 未完成任务；
   - 待核对项；
   - 禁止误读清单；
   - 后续推进前必须检查的内容。

7. `ROOT_LOG_INDEX_OLD_LOGS_CONSOLIDATION_TABLE.csv`
   - 旧日志整合索引；
   - 删除前检查表；
   - 不建议作为科研叙述正文阅读。

---

## 3. 旧日志删除策略

### 3.1 可以进入删除候选的文件类型

以下文件可以在本轮根目录日志确认后，进入删除候选：

- `UPDATE_LOG*.md`
- `ROOT_PROJECT_LOG*.md`
- `CURRENT_ENGINEERING_MANIFEST*.md`
- `PATCH_README*.md`
- `PATCH_MANIFEST*.md`
- `PATCH_FILE_LIST*.txt`
- 子目录 `logs/` 下的人工解释日志

### 3.2 不建议直接删除的文件类型

以下文件不建议作为“日志”直接删除：

- `README.md`
- `STRUCTURE_MANIFEST.txt`
- `VERSION_POLICY.md`
- `DEPENDENCY_MANIFEST.json`
- `outputs/*.csv`
- `outputs/*.json`
- `outputs/*.npz`
- `run_meta*`
- 源码与入口脚本

原因：这些文件属于工程结构、依赖、数据输出、运行元信息或代码承载，不等同于人工日志。删除它们会削弱后续核查能力。

---

## 4. 根目录日志后续维护规则

以后每轮新增工程或科研判断时，只更新对应根目录日志：

- 新科研结论或解释边界：更新 `ROOT_LOG_01`；
- 新入口、输出、目录变化：更新 `ROOT_LOG_02`；
- 版本状态、执行链、运行状态变化：更新 `ROOT_LOG_03`；
- 失败、负结果、不可信结果：更新 `ROOT_LOG_04`；
- 未完成项、禁止误读、待核对项：更新 `ROOT_LOG_05`。

不要再在子版本目录中额外开新的 `UPDATE_LOG_*`。

---

## 5. 删除旧日志前的最低检查条件

删除旧日志前，至少满足：

1. `ROOT_LOG_INDEX_OLD_LOGS_CONSOLIDATION_TABLE.csv` 已覆盖你准备删除的旧日志类型；
2. 关键历史信息已被吸收到 `ROOT_LOG_01`–`ROOT_LOG_05`；
3. V10.5 对原解释层的修正已进入根目录日志；
4. 失败记录与负结果已进入 `ROOT_LOG_04`；
5. peak_family_atlas、peak morphology audit、cartopy spatial validation、yearwise validation 等未实现项已进入 `ROOT_LOG_05`；
6. 你本人确认不再依赖旧日志作为唯一证据。

---

## 6. 重要限制声明

本日志体系保留的是**接管所需的最小高价值信息**，不是旧日志逐字归档。

若未来发生争议，应优先回查：

1. 源代码；
2. 输出表；
3. run_meta；
4. 本根目录日志；
5. 已删除旧日志的备份，如果你在删除前自行保留。
