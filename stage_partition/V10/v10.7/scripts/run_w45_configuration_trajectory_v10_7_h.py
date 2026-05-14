from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    # Expected: <root>/stage_partition/V10/v10.7/scripts/this.py
    for parent in [here.parent, *here.parents]:
        if (parent / "foundation").exists() or (parent / "stage_partition").exists():
            # If parent is v10.7/scripts or V10 etc, keep climbing to root with foundation if possible.
            cur = parent
            while cur.parent != cur and not (cur / "foundation").exists():
                cur = cur.parent
            if (cur / "foundation").exists():
                return cur
    # Hardcoded project default, matching project convention.
    return Path(r"D:\easm_project01")


def main() -> None:
    project_root = _resolve_project_root()
    src = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    from stage_partition_v10_7.w45_config_trajectory_pipeline import run_w45_configuration_trajectory_v10_7_h

    meta = run_w45_configuration_trajectory_v10_7_h(project_root)
    print("[V10.7_h] completed")
    print(f"[V10.7_h] output_root = {meta.get('output_root')}")
    print(f"[V10.7_h] status = {meta.get('status')}")


if __name__ == "__main__":
    main()
