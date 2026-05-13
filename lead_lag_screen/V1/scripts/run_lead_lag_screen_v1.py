\
from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parents[1]
SRC = V1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1 import LeadLagScreenSettings, run_lead_lag_screen_v1


if __name__ == "__main__":
    settings = LeadLagScreenSettings()
    run_lead_lag_screen_v1(settings)
