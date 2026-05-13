from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BUNDLE_ROOT = HERE.parents[1]
SRC_ROOT = BUNDLE_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from joint_main_window_reproduce_v10_1 import run_joint_main_window_reproduce_v10_1

if __name__ == '__main__':
    run_joint_main_window_reproduce_v10_1(BUNDLE_ROOT)
