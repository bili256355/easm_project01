from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "foundation").exists() and (parent / "stage_partition").exists():
            return parent
    return Path(r"D:\easm_project01")


def main() -> None:
    project_root = _resolve_project_root()
    src = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from stage_partition_v10_7.w45_h_package_pipeline import run_w45_h_package_cross_object_audit_v10_7_f

    meta = run_w45_h_package_cross_object_audit_v10_7_f(project_root)
    print("[V10.7_f] W45 H-package cross-object audit completed.")
    print(f"[V10.7_f] output_root = {meta.get('output_root')}")
    print("[V10.7_f] route decision: tables/w45_H_package_route_decision_v10_7_f.csv")


if __name__ == "__main__":
    main()
