from __future__ import annotations

from pathlib import Path
import sys


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "foundation").exists() and (p / "stage_partition").exists():
            return p
    # Expected script location: <root>/stage_partition/V10/v10.7/scripts/...
    return here.parents[4]


def main() -> None:
    project_root = _find_project_root()
    src_dir = project_root / "stage_partition" / "V10" / "v10.7" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from stage_partition_v10_7.w45_structure_transition_pipeline import run_w45_structure_transition_mapping_v10_7_k

    meta = run_w45_structure_transition_mapping_v10_7_k(project_root)
    print("[V10.7_k] completed")
    print(f"[V10.7_k] output_root = {meta.get('output_root')}")
    print(f"[V10.7_k] summary = {meta.get('summary_path')}")


if __name__ == "__main__":
    main()
