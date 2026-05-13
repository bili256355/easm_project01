from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V4_ROOT = THIS_FILE.parents[1]
SRC_DIR = V4_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lead_lag_screen_v4 import LeadLagScreenV4Settings, run_lead_lag_screen_v4


if __name__ == "__main__":
    summary = run_lead_lag_screen_v4(LeadLagScreenV4Settings())
    print(summary)
