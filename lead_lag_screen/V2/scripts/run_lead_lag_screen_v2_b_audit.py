from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V2_ROOT = THIS_FILE.parents[1]
SRC_DIR = V2_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lead_lag_screen_v2_b_audit import LeadLagScreenV2BAuditSettings, run_v2_b_audit


def main() -> None:
    settings = LeadLagScreenV2BAuditSettings()
    summary = run_v2_b_audit(settings)
    print(summary)


if __name__ == "__main__":
    main()
