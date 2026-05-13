from pathlib import Path
import sys

THIS = Path(__file__).resolve()
V7_ROOT = THIS.parents[1]
SRC = V7_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v7.W45_2d_prepost_metric_mirror_v7_z_2d_a import (  # noqa: E402
    run_W45_2d_prepost_metric_mirror_v7_z_2d_a,
)

if __name__ == "__main__":
    run_W45_2d_prepost_metric_mirror_v7_z_2d_a(V7_ROOT)
