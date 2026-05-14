from __future__ import annotations

from .event_content_config import EventContentConfig, EventWindow


def iter_event_windows(cfg: EventContentConfig) -> tuple[EventWindow, ...]:
    return cfg.event_windows


def days_in_range(lo: int, hi: int) -> list[int]:
    return list(range(int(lo), int(hi) + 1))
