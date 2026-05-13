from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
V1_1_ROOT = THIS_FILE.parents[1]
SRC = V1_1_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lead_lag_screen_v1_1 import LeadLagScreenSettings, run_lead_lag_screen_v1_1_structural_vp_a


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V1_1 structural V→P lead-lag screen.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--n-surrogates", type=int, default=None)
    parser.add_argument("--n-audit-surrogates", type=int, default=None)
    parser.add_argument("--n-direction-bootstrap", type=int, default=None)
    parser.add_argument("--debug-fast", action="store_true", help="Use smaller resampling counts for connection tests.")
    args = parser.parse_args()

    settings = LeadLagScreenSettings()
    updates = {}
    if args.project_root:
        updates["project_root"] = Path(args.project_root)
    if args.debug_fast:
        updates.update({"n_surrogates": 100, "n_audit_surrogates": 100, "n_direction_bootstrap": 100})
    if args.n_surrogates is not None:
        updates["n_surrogates"] = args.n_surrogates
    if args.n_audit_surrogates is not None:
        updates["n_audit_surrogates"] = args.n_audit_surrogates
    if args.n_direction_bootstrap is not None:
        updates["n_direction_bootstrap"] = args.n_direction_bootstrap
    if updates:
        settings = replace(settings, **updates)
    run_lead_lag_screen_v1_1_structural_vp_a(settings)


if __name__ == "__main__":
    main()
