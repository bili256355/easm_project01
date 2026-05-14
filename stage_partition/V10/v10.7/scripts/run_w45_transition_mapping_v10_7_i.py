from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "foundation").exists():
            return parent
        if (parent / "stage_partition").exists():
            cur = parent
            while cur.parent != cur and not (cur / "foundation").exists():
                cur = cur.parent
            if (cur / "foundation").exists():
                return cur
    return Path(r"D:\easm_project01")


def main() -> None:
    project_root = _resolve_project_root()
    src = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if src.exists():
        sys.path.insert(0, str(src))
    from stage_partition_v10_7.w45_transition_mapping_pipeline import run_w45_transition_mapping_v10_7_i

    meta = run_w45_transition_mapping_v10_7_i(project_root)
    print("[V10.7_i] completed")
    print(f"[V10.7_i] output_root = {meta.get('output_root')}")
    print(f"[V10.7_i] status = {meta.get('status')}")


if __name__ == "__main__":
    main()
