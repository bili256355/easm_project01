from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # .../stage_partition/V10/v10.7/scripts/run_*.py -> project root is parents[4]
    for parent in here.parents:
        if (parent / "foundation").exists() and (parent / "stage_partition").exists():
            return parent
    # Fallback for Windows path requested by the project convention.
    return Path(r"D:\easm_project01")


def main() -> None:
    project_root = _resolve_project_root()
    src = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage_partition_v10_7.h35_existence_pipeline import run_h35_existence_attribution_v10_7_e

    meta = run_h35_existence_attribution_v10_7_e(project_root)
    print("[V10.7_e] H35 existence attribution completed.")
    print(f"[V10.7_e] output_root = {meta.get('output_root')}")
    print("[V10.7_e] route decision table: tables/h35_existence_attribution_decision_v10_7_e.csv")


if __name__ == "__main__":
    main()
