import inspect

import pandas as pd
import pytest
from helpers import (
    CountsLaterSessions,
    FixedBound,
    PeekAtDeparture,
    QuantileOfEverything,
    StepAdjust,
)

import fleetoffer.models as models_module
from fleetoffer.engine import Daily
from fleetoffer.interface import Rule
from fleetoffer.lookahead import LookaheadError, check_no_lookahead, scramble_future
from fleetoffer.models import FrozenQuantile, RollingQuantile
from fleetoffer.synthetic import Regime, make_sessions

TZ = "America/Los_Angeles"
CUTOFF = pd.Timestamp("2001-02-14 10:30", tz=TZ)


@pytest.fixture(scope="module")
def sessions():
    regimes = [
        Regime("2001-01-01", sessions_per_weekday=15, median_stay_h=6.0),
        Regime("2001-02-10", sessions_per_weekday=8, median_stay_h=2.5, returning_share=0.3),
    ]
    return make_sessions(regimes, "2001-03-15", tz=TZ, seed=3)


HONEST = {
    "fixed": lambda _: FixedBound(2.0),
    "rolling": lambda _: RollingQuantile(14),
    "frozen, fitted before the cutoff": lambda _: FrozenQuantile(pd.Timestamp("2001-02-01", tz=TZ)),
    "frozen, fitted after the cutoff": lambda _: FrozenQuantile(pd.Timestamp("2001-03-01", tz=TZ)),
    "learns from outcomes": lambda _: StepAdjust(),
}


@pytest.mark.parametrize("every_h", [None, 2.0])
@pytest.mark.parametrize("name", list(HONEST))
def test_honest_rules_pass(sessions, name, every_h):
    compared = check_no_lookahead(HONEST[name], sessions, every_h=every_h)
    assert compared > 500


@pytest.mark.parametrize("name", list(HONEST))
def test_honest_rules_pass_at_the_morning_decision(sessions, name):
    compared = check_no_lookahead(HONEST[name], sessions, daily=Daily("10:00", TZ))
    assert compared > 200


def test_every_rule_in_the_package_is_checked():
    defined = {
        cls
        for _, cls in inspect.getmembers(models_module, inspect.isclass)
        if issubclass(cls, Rule) and cls.__module__ == models_module.__name__
    }
    checked = {type(factory(None)) for factory in HONEST.values()}
    assert defined <= checked, f"add a look-ahead test for: {defined - checked}"


@pytest.mark.parametrize("cheat", [PeekAtDeparture, QuantileOfEverything, CountsLaterSessions])
def test_the_check_catches_cheating_rules(sessions, cheat):
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, sessions)
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, sessions, daily=Daily("10:00", TZ))


def test_scrambling_keeps_the_past_and_changes_the_future(sessions):
    scrambled = scramble_future(sessions, CUTOFF, seed=1).set_index("session_id")
    original = sessions.set_index("session_id")
    cutoff = CUTOFF.tz_convert("UTC")

    started_before = original[original["connect_time"] < cutoff]
    assert started_before.index.isin(scrambled.index).all()
    rows = scrambled.loc[started_before.index]
    ended_before = started_before["disconnect_time"] < cutoff
    pd.testing.assert_series_equal(
        rows.loc[ended_before, "disconnect_time"],
        started_before.loc[ended_before, "disconnect_time"],
    )
    assert (rows.loc[~ended_before, "disconnect_time"] >= cutoff).all()
    assert not rows.loc[~ended_before, "disconnect_time"].equals(
        started_before.loc[~ended_before, "disconnect_time"]
    )


def test_scrambling_changes_the_number_of_later_sessions(sessions):
    cutoff = CUTOFF.tz_convert("UTC")
    later = int((sessions["connect_time"] >= cutoff).sum())
    scrambled = scramble_future(sessions, CUTOFF, seed=1)
    assert int((scrambled["connect_time"] >= cutoff).sum()) != later
