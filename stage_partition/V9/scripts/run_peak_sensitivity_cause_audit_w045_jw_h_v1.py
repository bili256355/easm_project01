from pathlib import Path
import sys

V9_ROOT = Path(__file__).resolve().parents[1]
SRC = V9_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stage_partition_v9.peak_sensitivity_cause_audit_w045_jw_h_v1 import (  # noqa: E402
    W045JwHPeakSensitivityCauseAuditSettings,
    run_w045_jw_h_peak_sensitivity_cause_audit_v1,
)


if __name__ == "__main__":
    settings = W045JwHPeakSensitivityCauseAuditSettings()
    run_w045_jw_h_peak_sensitivity_cause_audit_v1(V9_ROOT, settings=settings)
