import inspect

import pandas as pd
import pytest
from helpers import PeekAtDeparture, QuantileOfEverything, StepAdjust

import staypromise.rules as rules_module
from staypromise.checks import LookaheadError, check_no_lookahead, scramble_future
from staypromise.interface import Rule
from staypromise.rules import FixedPromise, FrozenQuantile, RollingQuantile
from staypromise.synthetic import Regime, make_sessions

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
    "fixed": lambda _: FixedPromise(2.0),
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


def test_every_rule_in_the_package_is_checked():
    defined = {
        cls
        for _, cls in inspect.getmembers(rules_module, inspect.isclass)
        if issubclass(cls, Rule) and cls.__module__ == rules_module.__name__
    }
    checked = {type(factory(None)) for factory in HONEST.values()}
    assert defined <= checked, f"add a look-ahead test for: {defined - checked}"


@pytest.mark.parametrize("cheat", [PeekAtDeparture, QuantileOfEverything])
def test_the_check_catches_cheating_rules(sessions, cheat):
    with pytest.raises(LookaheadError):
        check_no_lookahead(cheat, sessions)


def test_scrambling_keeps_the_past_and_changes_the_future(sessions):
    scrambled = scramble_future(sessions, CUTOFF, seed=1).set_index("session_id")
    original = sessions.set_index("session_id")
    cutoff = CUTOFF.tz_convert("UTC")

    started_before = original[original["connect_time"] < cutoff]
    assert started_before.index.isin(scrambled.index).all()
    kept_rows = scrambled.loc[started_before.index]
    ended_before = started_before["disconnect_time"] < cutoff
    pd.testing.assert_series_equal(
        kept_rows.loc[ended_before, "disconnect_time"],
        started_before.loc[ended_before, "disconnect_time"],
    )
    assert (kept_rows.loc[~ended_before, "disconnect_time"] >= cutoff).all()
    assert not kept_rows.loc[~ended_before, "disconnect_time"].equals(
        started_before.loc[~ended_before, "disconnect_time"]
    )
