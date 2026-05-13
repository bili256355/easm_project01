from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_SRC = _THIS.parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lead_lag_screen_v3 import LeadLagScreenV3Settings, run_lead_lag_screen_v3


if __name__ == "__main__":
    run_lead_lag_screen_v3(LeadLagScreenV3Settings())
