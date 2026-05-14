from __future__ import annotations

from pathlib import Path
import sys


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../stage_partition/V10/v10.7/scripts/run_*.py -> project root is parents[4]
    return here.parents[4]


def main() -> None:
    project_root = _resolve_project_root()
    src_root = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from stage_partition_v10_7 import run_h_w045_scale_diagnostic_v10_7_b

    meta = run_h_w045_scale_diagnostic_v10_7_b(project_root)
    print("[V10.7_b] Summary:", meta.get("outputs", {}).get("summary"))


if __name__ == "__main__":
    main()
