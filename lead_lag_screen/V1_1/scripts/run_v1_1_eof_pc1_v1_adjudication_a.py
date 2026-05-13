from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V11_ROOT = THIS_FILE.parents[1]
SRC = V11_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1_1.eof_pc1_v1_adjudication_pipeline import run_eof_pc1_v1_adjudication_a
from lead_lag_screen_v1_1.eof_pc1_v1_adjudication_settings import EOFPC1V1AdjudicationSettings


def main() -> None:
    parser = argparse.ArgumentParser(description="V1_1 EOF-PC1 adjudication audit against V1 old-index T3 weakening.")
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args()
    settings = EOFPC1V1AdjudicationSettings()
    if args.project_root:
        settings = EOFPC1V1AdjudicationSettings(project_root=Path(args.project_root))
    summary = run_eof_pc1_v1_adjudication_a(settings)
    print("[DONE] V1_1 EOF-PC1 V1 adjudication audit")
    print(f"[OUTPUT] {summary['output_dir']}")


if __name__ == "__main__":
    main()
