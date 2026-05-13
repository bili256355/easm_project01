easm_project01 / stage_partition / V2
audit_sync_v1_a replacement-file patch

本补丁只新增“审计同步入口 + 审计同步 pipeline”，不修改 detector 主链。

补丁目的：
1. 固化唯一主窗口对象表：stage_partition_main_windows.csv
2. 拆开点/band 证据层：window_evidence_mapping.csv
3. 生成逐窗 support 审计：window_support_audit.csv
4. 生成逐窗 retention 审计：window_retention_audit.csv
5. 给出可信度分层：audit_trust_tiers.csv

应用方式：
- 将本补丁包内文件覆盖/新增到 D:\easm_project01 对应位置
- 然后运行：
  python D:\easm_project01\stage_partition\V2\scripts\run_stage_partition_v2_audit_sync.py

默认路径：
- 工程根目录：D:\easm_project01
- 源结果目录：D:\easm_project01\stage_partition\V2\outputs\baseline_a
- 新输出目录：D:\easm_project01\stage_partition\V2\outputs\audit_sync_v1_a
- 新日志目录：D:\easm_project01\stage_partition\V2\logs\audit_sync_v1_a

说明：
- 本补丁不重新跑 foundation
- 本补丁不重新跑 detector
- 本补丁只审计和重收口 V2 现有结果层
