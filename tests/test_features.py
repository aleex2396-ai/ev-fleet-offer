import numpy as np
import pandas as pd
import pytest

from fleetoffer.academic import day_flags
from fleetoffer.data import app_entries, driver_links
from fleetoffer.engine import _view, fleet_replay
from fleetoffer.features import NUMERIC, at_decision
from fleetoffer.interface import FleetRule

PT = "America/Los_Angeles"


def test_calendar_flags_on_dates_read_off_the_registrars_calendars():
    flags = day_flags(
        pd.DatetimeIndex(
            [
                "2019-01-21",  # Martin Luther King Day, in the winter term
                "2019-02-05",  # winter term
                "2019-03-25",  # spring recess
                "2019-07-15",  # summer
                "2019-07-05",  # Independence Day, second day
                "2019-12-26",  # year-end closure, winter recess
                "2018-11-23",  # day after Thanksgiving
            ]
        )
    )
    assert flags["holiday"].tolist() == [True, False, False, False, True, True, True]
    assert flags["in_term"].tolist() == [True, True, False, False, False, False, True]
    assert flags["summer"].tolist() == [False, False, False, True, True, False, False]


def test_calendar_refuses_dates_it_does_not_cover():
    with pytest.raises(ValueError):
        day_flags(pd.DatetimeIndex(["2001-03-01"]))


def _inputs(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "session_id",
            "entry",
            "entries",
            "modified_at",
            "minutes_available",
            "requested_departure",
            "kwh_requested",
            "miles_requested",
        ],
    )


def test_a_session_is_linked_when_its_first_entry_was_recorded():
    t = pd.Timestamp("2019-02-04 08:00", tz=PT).tz_convert("UTC")
    sessions = pd.DataFrame(
        {
            "session_id": ["a", "b", "c"],
            "connect_time": [t, t, t],
            "disconnect_time": [t + pd.Timedelta(hours=8)] * 3,
            "user_id": ["u1", "u2", None],
        }
    )
    hour = pd.Timedelta(hours=1)
    inputs = _inputs(
        [
            ("a", 1, 2, t + 3 * hour, 300, t + 8 * hour, 10.0, 30.0),
            ("a", 0, 2, t + 2 * hour, 240, t + 6 * hour, 10.0, 30.0),
        ]
    )
    linked = driver_links(sessions, inputs)
    assert linked["linked_time"].iloc[0] == t + 2 * hour
    assert linked["linked_time"].iloc[1:].isna().all()  # no entry: never shown as linked
    assert app_entries(inputs)["entry"].tolist() == [0, 1]


@pytest.fixture
def day():
    """At 10:00 on Monday 4 February 2019: car x (linked at 09:00, three earlier sessions,
    plan typed at 09:00 and edited at 11:00), car y (claimed only at 12:00) and car z
    (plugged in on Saturday evening)."""
    t = pd.Timestamp("2019-02-04 10:00", tz=PT).tz_convert("UTC")
    h = pd.Timedelta(hours=1)
    d = pd.Timedelta(days=1)
    rows = [
        # id, connect, disconnect, user, linked, kwh
        ("p1", t - 7 * d - 2 * h, t - 7 * d + 6 * h, "u1", t - 7 * d - 2 * h, 6.0),
        ("p2", t - 6 * d - 2 * h, t - 6 * d + 4 * h, "u1", t - 6 * d - 2 * h, 8.0),
        ("p3", t - 3 * d - 2 * h, t - 3 * d + 2 * h, "u1", t - 3 * d - 2 * h, 10.0),
        ("x", t - 2 * h, t + 5 * h, "u1", t - h, 7.0),
        ("y", t - 1.5 * h, t + 3 * h, "u2", t + 2 * h, 5.0),
        ("z", t - 2 * d - 14 * h, t + 1 * h, None, pd.NaT, 9.0),
    ]
    table = pd.DataFrame(
        rows,
        columns=["session_id", "connect_time", "disconnect_time", "user_id", "linked_time", "kwh"],
    ).rename(columns={"kwh": "kwh_delivered"})
    table["done_charging_time"] = table["disconnect_time"]
    table["station_id"] = ["s1", "s1", "s2", "s1", "s2", "s3"]
    table["linked_time"] = pd.to_datetime(table["linked_time"], utc=True)
    entries = pd.DataFrame(
        {
            "session_id": ["x", "x", "y"],
            "entry": [0, 1, 0],
            "modified_at": [t - h, t + h, t + 2 * h],
            "minutes_available": [480.0, 600.0, 60.0],
            "requested_departure": [t + 6 * h, t + 8 * h, t + 3 * h],
            "kwh_requested": [12.0, 20.0, 5.0],
        }
    )
    present = np.flatnonzero((table["connect_time"] < t) & (table["disconnect_time"] >= t))
    ended = np.flatnonzero(table["disconnect_time"] < t)
    return _view(table, t, present, ended, entries)


def test_inputs_use_only_what_was_recorded_before_ten(day):
    feats = at_decision(day).set_index("session_id")
    assert list(feats.index) == ["x", "y", "z"]
    assert set(NUMERIC) <= set(feats.columns)
    x, y, z = feats.loc["x"], feats.loc["y"], feats.loc["z"]
    assert x["linked"] == 1 and y["linked"] == 0 and z["linked"] == 0  # y claimed after 10:00
    assert x["paid"] == 1 and y["paid"] == 0
    assert x["n_past"] == 3 and np.isnan(y["n_past"])
    assert x["stay_q50"] == pytest.approx(6.0) and x["kwh_q50"] == pytest.approx(8.0)
    assert x["days_since_last"] == pytest.approx(3 + 2 / 24)
    # the plan typed at 09:00, not its edit at 11:00
    assert x["typed"] == 1 and x["typed_remaining_h"] == pytest.approx(6.0)
    assert x["typed_480"] == 1 and x["typed_kwh"] == pytest.approx(12.0)
    assert y["typed"] == 0 and np.isnan(y["typed_remaining_h"])
    assert x["plugin_hour"] == pytest.approx(8.0)
    assert z["plugin_hour"] == pytest.approx(-52.0)  # Saturday 20:00
    assert x["in_term"] == 1 and x["holiday"] == 0 and x["weekday"] == 0
    assert x["cars_present"] == 3 and x["station"] == "s1"


def test_the_engine_shows_entries_only_once_recorded():
    t0 = pd.Timestamp("2019-02-04 07:00", tz=PT).tz_convert("UTC")
    h = pd.Timedelta(hours=1)
    sessions = pd.DataFrame(
        {
            "session_id": ["a", "b"],
            "connect_time": [t0, t0],
            "disconnect_time": [t0 + 30 * h, t0 + 30 * h],
            "user_id": ["u", None],
            "kwh_delivered": [10.0, 10.0],
            "done_charging_time": [t0 + 2 * h, t0 + 2 * h],
        }
    )
    entries = pd.DataFrame(
        {
            "session_id": ["a", "a"],
            "entry": [0, 1],
            "modified_at": [t0 + h, t0 + 26 * h],  # 08:00 Monday, 09:00 Tuesday
            "minutes_available": [60.0, 120.0],
            "requested_departure": [t0 + 2 * h, t0 + 28 * h],
            "kwh_requested": [5.0, 6.0],
        }
    )

    class Spy(FleetRule):
        name = "spy"

        def __init__(self):
            self.seen = []

        def bounds(self, view):
            self.seen.append(len(view.entries))
            return np.zeros(len(view.present)), np.zeros(len(view.present))

    spy = Spy()
    days = fleet_replay(sessions, spy, method="bounds", entries=entries).days
    assert spy.seen == [1, 2]
    assert days["size_kwh"].isna().all() if "size_kwh" in days else True
