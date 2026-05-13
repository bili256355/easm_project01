from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V7_ROOT = THIS_FILE.parents[1]
V7_SRC = V7_ROOT / "src"
if str(V7_SRC) not in sys.path:
    sys.path.insert(0, str(V7_SRC))

from stage_partition_v7.progress_order_significance import run_progress_order_significance_audit_v7_e1


if __name__ == "__main__":
    run_progress_order_significance_audit_v7_e1()
