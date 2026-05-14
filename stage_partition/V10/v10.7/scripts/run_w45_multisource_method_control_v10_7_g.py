from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../stage_partition/V10/v10.7/scripts/run_*.py -> project root is parents[4]
    return here.parents[4]


def main() -> None:
    project_root = _resolve_project_root()
    src = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage_partition_v10_7.w45_multisource_method_control_pipeline import run_w45_multisource_method_control_v10_7_g

    meta = run_w45_multisource_method_control_v10_7_g(project_root)
    print("V10.7_g completed")
    print(f"status={meta.get('status')}")
    print(f"output_root={meta.get('output_root')}")


if __name__ == "__main__":
    main()
