from __future__ import annotations

import sys
from pathlib import Path

VERSION_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = VERSION_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from foundation_v1.pipeline_runner import run_foundation_v1


if __name__ == "__main__":
    run_foundation_v1()
