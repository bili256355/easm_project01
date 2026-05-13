from pathlib import Path
import sys

V93_ROOT = Path(__file__).resolve().parents[1]
SRC = V93_ROOT / "src"
sys.path.insert(0, str(SRC))

from stage_partition_v9_3.direct_year_profile_supervised_svd_v9_3_a import (
    run_direct_year_profile_supervised_svd_v9_3_a,
)

if __name__ == "__main__":
    run_direct_year_profile_supervised_svd_v9_3_a(V93_ROOT)
