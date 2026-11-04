import numpy as np
import pandas as pd
import pytest

from staypromise.sessions import validate_sessions
from staypromise.synthetic import Regime, make_sessions

REGIMES = [
    Regime("2001-01-01", sessions_per_weekday=40, median_stay_h=6.0, stay_spread=0.5),
    Regime("2001-04-01", sessions_per_weekday=0.0, median_stay_h=3.0),
    Regime("2001-05-01", sessions_per_weekday=40, median_stay_h=2.0, returning_share=0.3),
]


def stay_hours(table):
    return (table["disconnect_time"] - table["connect_time"]) / pd.Timedelta(hours=1)


def test_same_seed_same_table_and_a_valid_one():
    a = make_sessions(REGIMES, "2001-07-01", seed=5)
    pd.testing.assert_frame_equal(a, make_sessions(REGIMES, "2001-07-01", seed=5))
    assert not a.equals(make_sessions(REGIMES, "2001-07-01", seed=6))
    validate_sessions(a)


def test_each_regime_has_the_behaviour_it_was_given():
    t = make_sessions(REGIMES, "2001-07-01", tz="America/Los_Angeles", seed=1)
    local = t["connect_time"].dt.tz_convert("America/Los_Angeles")
    first = t[local < pd.Timestamp("2001-04-01", tz="America/Los_Angeles")]
    closed = t[
        (local >= pd.Timestamp("2001-04-01", tz="America/Los_Angeles"))
        & (local < pd.Timestamp("2001-05-01", tz="America/Los_Angeles"))
    ]
    last = t[local >= pd.Timestamp("2001-05-01", tz="America/Los_Angeles")]

    assert np.median(stay_hours(first)) == pytest.approx(6.0, rel=0.05)
    assert np.median(stay_hours(last)) == pytest.approx(2.0, rel=0.05)
    assert len(closed) == 0
    returning = set(last["user_id"]) & set(first["user_id"])
    assert len(returning) == round(0.3 * 300)


def test_drift_moves_the_median_by_the_stated_share():
    drifting = [Regime("2001-01-01", 60, median_stay_h=4.0, stay_spread=0.4, drift_per_week=0.02)]
    t = make_sessions(drifting, "2001-12-31", seed=2)
    weeks = (t["connect_time"] - t["connect_time"].min()) / pd.Timedelta(weeks=1)
    slope = np.polyfit(weeks, np.log(stay_hours(t)), 1)[0]
    assert slope == pytest.approx(0.02, abs=0.002)
