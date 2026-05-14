from __future__ import annotations

from pathlib import Path
import os
import sys


def _resolve_project_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()
    env = os.environ.get("EASM_PROJECT_ROOT") or os.environ.get("V10_7_PROJECT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.name == "easm_project01":
            return parent
        if (parent / "foundation").exists() and (parent / "stage_partition").exists():
            return parent
    return Path.cwd().resolve()


def main() -> None:
    src_root = Path(__file__).resolve().parents[1] / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from stage_partition_v10_7.h35_residual_pipeline import run_h35_residual_independence_v10_7_d
    meta = run_h35_residual_independence_v10_7_d(_resolve_project_root())
    print(f"[V10.7_d] finished: {meta.get('output_root')}")
    for item in meta.get("final_route_decision", []):
        print(f"[V10.7_d] {item.get('decision_item')}: {item.get('status')}")


if __name__ == "__main__":
    main()
