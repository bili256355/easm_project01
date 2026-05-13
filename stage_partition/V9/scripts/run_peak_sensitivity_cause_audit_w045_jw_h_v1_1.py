# -*- coding: utf-8 -*-
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).resolve()
V9_ROOT = SCRIPT_PATH.parents[1]
SRC_ROOT = V9_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v9.peak_sensitivity_cause_audit_w045_jw_h_v1_1 import (  # noqa: E402
    run_peak_sensitivity_cause_audit_w045_jw_h_v1_1,
)

if __name__ == "__main__":
    run_peak_sensitivity_cause_audit_w045_jw_h_v1_1(V9_ROOT)
