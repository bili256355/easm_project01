from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v6.pipeline_selection_frequency import run_stage_partition_v6_b_selection_frequency


if __name__ == '__main__':
    run_stage_partition_v6_b_selection_frequency()
