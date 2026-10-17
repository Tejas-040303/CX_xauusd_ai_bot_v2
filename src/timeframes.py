"""
Project MIDAS v2 — Attaching M15 context to M5 bars without look-ahead
=======================================================================
Bar timestamps are OPEN times. An M5 bar opening at 10:05 closes at 10:10;
the M15 bar opening at 10:00 only closes at 10:15. Joining on open times
(merge_asof on "datetime") hands the 10:05 M5 bar that unfinished M15 bar —
up to 10 minutes of the future. The live bot never has that: it only sees
M15 bars that have closed.

merge_closed_m15() gives each M5 bar the last M15 bar that had CLOSED by the
time the M5 bar closed, so backtests see exactly what live trading sees.
"""

import pandas as pd

M5_BAR = pd.Timedelta(minutes=5)
M15_BAR = pd.Timedelta(minutes=15)


def merge_closed_m15(df_m5: pd.DataFrame, df_m15: pd.DataFrame,
                     m15_key: str = "m15_datetime_key") -> pd.DataFrame:
    """Attach M15 rows (keyed by their open time in `m15_key`) to M5 rows.

    Keeps `m15_key` with the M15 bar's open time, so callers can rename it
    to "m15_datetime" as before (strategies use it to spot a new M15 bar).
    """
    m5 = df_m5.sort_values("datetime").copy()
    m5["_m5_close"] = m5["datetime"] + M5_BAR
    m15 = df_m15.sort_values(m15_key).copy()
    m15["_m15_close"] = m15[m15_key] + M15_BAR

    merged = pd.merge_asof(m5, m15, left_on="_m5_close", right_on="_m15_close",
                           direction="backward")
    return merged.drop(columns=["_m5_close", "_m15_close"]).reset_index(drop=True)
