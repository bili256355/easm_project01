\
from __future__ import annotations

from .core import run_screen
from .logging_utils import setup_logger
from .settings import LeadLagScreenSettings


def run_lead_lag_screen_v1(settings: LeadLagScreenSettings | None = None):
    settings = settings or LeadLagScreenSettings()
    logger = setup_logger(settings.log_dir)
    logger.info("Starting lead_lag_screen_v1")
    logger.info("Project root: %s", settings.project_root)
    logger.info("Output dir: %s", settings.output_dir)
    logger.info("Input: %s", settings.input_index_anomalies)
    summary = run_screen(settings, logger)
    logger.info("Finished lead_lag_screen_v1: %s", summary)
    return summary
