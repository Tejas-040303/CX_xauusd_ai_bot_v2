"""
Project MIDAS v2 — Broker server time ↔ UTC
============================================
MT5 bar and tick times are the broker server's wall clock, not UTC. Most MT5
brokers run "New York close" time: server time = New York time + 7 hours, so
the server is UTC+2 in winter and UTC+3 while the US is on daylight saving.
Configure the rule in config.py (BROKER_TIME_ZONE, BROKER_TIME_SHIFT_HOURS);
main.py checks it against the live server clock at startup.

DST switches happen on Sundays, when gold is closed, so one offset per broker
date is exact for every trading bar.
"""

from datetime import datetime, timezone

import pandas as pd

from config import BROKER_TIME_ZONE, BROKER_TIME_SHIFT_HOURS


def utc_offset_hours(broker_times: pd.Series) -> pd.Series:
    """Broker UTC offset (whole hours) for each naive broker-time timestamp."""
    days = broker_times.dt.normalize()
    unique_days = pd.Series(days.unique())
    noon = (unique_days + pd.Timedelta(hours=12)).dt.tz_localize(BROKER_TIME_ZONE)
    offsets = noon.map(lambda ts: ts.utcoffset().total_seconds() / 3600) + BROKER_TIME_SHIFT_HOURS
    return days.map(dict(zip(unique_days, offsets))).astype(int)


def to_utc(broker_times: pd.Series) -> pd.Series:
    """Naive broker-time timestamps → naive UTC timestamps."""
    return broker_times - pd.to_timedelta(utc_offset_hours(broker_times), unit="h")


def expected_offset_now() -> int:
    """What the configured rule says the broker's UTC offset is right now."""
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_convert(BROKER_TIME_ZONE)
    return int(now.utcoffset().total_seconds() / 3600 + BROKER_TIME_SHIFT_HOURS)
