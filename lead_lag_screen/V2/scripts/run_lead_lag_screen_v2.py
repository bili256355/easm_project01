from __future__ import annotations

import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V2_ROOT = THIS_FILE.parents[1]
SRC_DIR = V2_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lead_lag_screen_v2 import LeadLagScreenV2Settings, run_pcmci_plus_smooth5_v2
from lead_lag_screen_v2.logging_utils import setup_logger


def main() -> None:
    settings = LeadLagScreenV2Settings()
    logger = setup_logger(settings.log_dir)
    summary = run_pcmci_plus_smooth5_v2(settings, logger=logger)
    print(summary)


if __name__ == "__main__":
    main()
