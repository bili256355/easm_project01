from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
V7_ROOT = THIS_FILE.parents[1]
SRC_ROOT = V7_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stage_partition_v7.W45_multi_object_prepost_clean_mainline_v7_z_clean import (  # noqa: E402
    run_W45_multi_object_prepost_clean_mainline_v7_z_clean,
)

if __name__ == "__main__":
    run_W45_multi_object_prepost_clean_mainline_v7_z_clean(V7_ROOT)
