import math

import numpy as np
import pandas as pd
import pytest
from helpers import FixedBound, Spy, hours_from_t0, table

from fleetoffer.data import from_hours, to_hours
from fleetoffer.engine import Daily, decision_times, replay
from fleetoffer.interface import Rule


def test_held_failed_and_when_each_outcome_is_revealed():
    sessions = table([("a", 0, 5), ("b", 1, 2)])
    spy = Spy(hours=3.0)
    decisions = replay(sessions, spy).set_index("session_id")

    assert decisions.loc["a", "held"] and not decisions.loc["b", "held"]
    assert decisions.loc["a", "remaining_h"] == pytest.approx(5.0)
    outcomes = {e[2].session_id: e[2] for e in spy.events if e[0] == "outcome"}
    # Held: known when the bound's time passes. Failed: known when the car leaves.
    assert outcomes["a"].revealed_h == pytest.approx(hours_from_t0(3.0))
    assert outcomes["b"].revealed_h == pytest.approx(hours_from_t0(2.0))


def test_history_shows_only_what_was_known():
    sessions = table([("a", 0, 2), ("b", 1, 10), ("c", 3, 4), ("d", 3, 5)])
    spy = Spy()
    replay(sessions, spy)
    window = spy.windows["c"]

    # Sessions plugged in strictly before 3 h: a (left at 2 h) and b (still plugged in).
    assert len(window) == 2
    assert list(window.ended) == [True, False]
    # b's true 9-hour stay is hidden: only its 2 hours so far are shown.
    assert list(window.observed_h) == pytest.approx([2.0, 2.0])


def test_events_at_exactly_the_decision_time_are_not_yet_known():
    sessions = table([("a", 0, 3), ("b", 3, 4)])
    spy = Spy()
    replay(sessions, spy)
    window = spy.windows["b"]
    assert not window.ended[0]
    assert window.observed_h[0] == pytest.approx(3.0)


def test_outcomes_arrive_in_time_order_and_never_before_a_tied_decision():
    # a's 2-hour bound holds and is revealed at 2 h, exactly when b plugs in.
    sessions = table([("a", 0, 10), ("b", 2, 3), ("c", 5, 6)])
    spy = Spy(hours=2.0)
    replay(sessions, spy)
    times = [e[1] for e in spy.events]
    assert times == sorted(times)
    kinds = [(e[0], e[2] if e[0] == "decision" else e[2].session_id) for e in spy.events]
    assert kinds.index(("decision", "b")) < kinds.index(("outcome", "a"))


def test_renewed_bounds_while_the_car_is_plugged_in():
    connect, disconnect = [0.0, 10.0], [2.5, 12.0]
    assert decision_times(connect, disconnect, None) == [(0, 0.0), (1, 10.0)]
    # A car leaving at exactly a decision time is still asked about: its departure is not
    # yet known.
    assert decision_times(connect, disconnect, 1.0) == [
        (0, 0.0),
        (0, 1.0),
        (0, 2.0),
        (1, 10.0),
        (1, 11.0),
        (1, 12.0),
    ]
    decisions = replay(table([("a", 0, 2.5)]), FixedBound(1.0), every_h=1.0)
    assert list(decisions["elapsed_h"]) == pytest.approx([0.0, 1.0, 2.0])
    assert list(decisions["held"]) == [True, True, False]


class _Abstain(Rule):
    name = "abstain"

    def bound(self, request, history):
        return None


class _Returns(Rule):
    def __init__(self, value):
        self.value, self.name = value, f"returns {value}"

    def bound(self, request, history):
        return self.value


def test_no_bound_is_recorded_as_not_given():
    decisions = replay(table([("a", 0, 1)]), _Abstain())
    assert not decisions.loc[0, "given"]
    assert math.isnan(decisions.loc[0, "bound_h"])
    assert pd.isna(decisions.loc[0, "held"])


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_bounds_are_refused(bad):
    with pytest.raises(ValueError):
        replay(table([("a", 0, 1)]), _Returns(bad))


class _KeepsHistory(Rule):
    name = "keeps history"

    def __init__(self):
        self.saved = None

    def bound(self, request, history):
        if self.saved is not None:
            self.saved.window()
        self.saved = history
        return 0.0


def test_a_history_cannot_be_read_after_its_decision():
    with pytest.raises(RuntimeError):
        replay(table([("a", 0, 1), ("b", 2, 3)]), _KeepsHistory())


# 10:00 in Los Angeles is 17:00 UTC in summer (PDT) and 18:00 UTC in winter (PST).
LA = "America/Los_Angeles"


def utc_hours(*stamps):
    return to_hours(pd.Series(pd.to_datetime(list(stamps), utc=True)))


def test_daily_decisions_cover_every_car_present_at_ten_on_weekdays():
    # Fri 7 Jun 2019 and Mon 10 Jun 2019; the weekend in between has no decision.
    connect = utc_hours(
        "2019-06-07 15:00", "2019-06-07 17:00", "2019-06-07 18:00", "2019-06-08 16:00"
    )
    disconnect = utc_hours(
        "2019-06-07 17:00", "2019-06-07 20:00", "2019-06-07 22:00", "2019-06-10 19:00"
    )
    ten_friday, ten_monday = utc_hours("2019-06-07 17:00", "2019-06-10 17:00")
    pairs = decision_times(connect, disconnect, daily=Daily("10:00", LA))
    # Car 0 leaves at exactly 10:00, so it is still present; car 1 plugs in at exactly
    # 10:00, so it is not yet present; car 2 comes later; car 3 stays over the weekend.
    assert pairs == [(0, ten_friday), (3, ten_monday)]


def test_daily_decisions_follow_daylight_saving():
    winter, summer = Daily("10:00", LA).times_h(*utc_hours("2019-01-07 12:00", "2019-07-08 23:00"))[
        [0, -1]
    ]
    assert from_hours([winter])[0] == pd.Timestamp("2019-01-07 18:00", tz="UTC")
    assert from_hours([summer])[0] == pd.Timestamp("2019-07-08 17:00", tz="UTC")


def test_daily_replay_scores_each_present_car():
    # Hours after Monday 1 January 2001, 00:00 UTC; decisions at 10 h and 34 h.
    sessions = table([("a", 0, 30), ("b", 5, 10.5), ("c", 11, 40)])
    decisions = replay(sessions, FixedBound(1.0), daily=Daily("10:00", "UTC"))
    assert list(decisions["session_id"]) == ["a", "b", "c"]
    assert list(decisions["elapsed_h"]) == pytest.approx([10.0, 5.0, 23.0])
    assert list(decisions["held"]) == [True, False, True]
    with pytest.raises(ValueError):
        decision_times(np.array([0.0]), np.array([1.0]), 1.0, Daily())
