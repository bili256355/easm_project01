Apply this patch into D:\easm_project01 by replacing the files under stage_partition\V2\src\stage_partition_v2.

This patch does NOT change the detector logic.
It only converts empty-slice warning paths into explicit audited missing-data handling for:
- profiles.py
- state_vector.py
- support_audit.py
and wires the new audit outputs in pipeline.py.
