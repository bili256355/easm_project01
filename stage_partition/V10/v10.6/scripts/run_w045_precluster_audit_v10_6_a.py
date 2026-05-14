from __future__ import annotations

import sys
from pathlib import Path


def _infer_project_root() -> Path:
    # script = <root>/stage_partition/V10/v10.6/scripts/run_*.py
    return Path(__file__).resolve().parents[4]


def main() -> None:
    project_root = _infer_project_root()
    src_root = project_root / "stage_partition" / "V10" / "v10.6" / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from stage_partition_v10_6 import run_w045_precluster_audit_v10_6_a

    print("[V10.6_a] W045 precluster audit started")
    print(f"[V10.6_a] project_root = {project_root}")
    meta = run_w045_precluster_audit_v10_6_a(project_root)
    print(f"[V10.6_a] output_root = {meta['output_root']}")
    print("[V10.6_a] generated tables/figures/run_meta successfully")
    print("[V10.6_a] boundary: method-layer audit only; no causal/spatial/yearwise conclusion")


if __name__ == "__main__":
    main()
