import numpy as np
import pandas as pd
import pytest
from helpers import table

from staypromise.sessions import from_hours, to_hours, validate_sessions


def test_valid_table_is_sorted_and_in_utc():
    t = table([("b", 2, 3), ("a", 0, 1)])
    t["connect_time"] = t["connect_time"].dt.tz_convert("America/Los_Angeles")
    out = validate_sessions(t)
    assert list(out["session_id"]) == ["a", "b"]
    assert str(out["connect_time"].dt.tz) == "UTC"


@pytest.mark.parametrize(
    "breaks",
    [
        lambda t: t.drop(columns="disconnect_time"),
        lambda t: t.assign(connect_time=t["connect_time"].dt.tz_localize(None)),
        lambda t: t.assign(disconnect_time=t["connect_time"]),
        lambda t: t.assign(session_id="same"),
    ],
    ids=["missing column", "no time zone", "zero-length stay", "duplicate ids"],
)
def test_invalid_tables_are_refused(breaks):
    with pytest.raises(ValueError):
        validate_sessions(breaks(table([("a", 0, 1), ("b", 2, 3)])))


def test_hours_round_trip():
    times = pd.Series(pd.to_datetime(["2019-03-10 09:15:30", "2021-09-14 23:59:59"], utc=True))
    back = from_hours(to_hours(times))
    assert np.abs((back - pd.DatetimeIndex(times)).total_seconds()).max() < 1e-3
