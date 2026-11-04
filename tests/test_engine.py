import math

import pandas as pd
import pytest
from helpers import Spy, hours_from_t0, table

from staypromise.engine import decision_times, replay
from staypromise.interface import Rule
from staypromise.rules import FixedPromise


def test_kept_broken_and_when_each_outcome_is_revealed():
    sessions = table([("a", 0, 5), ("b", 1, 2)])
    spy = Spy(hours=3.0)
    decisions = replay(sessions, spy).set_index("session_id")

    assert decisions.loc["a", "kept"] and not decisions.loc["b", "kept"]
    assert decisions.loc["a", "remaining_h"] == pytest.approx(5.0)
    outcomes = {e[2].session_id: e[2] for e in spy.events if e[0] == "outcome"}
    # Kept: known when the promised time passes. Broken: known when the car leaves.
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
    # a's 2-hour promise is kept and revealed at 2 h, exactly when b plugs in.
    sessions = table([("a", 0, 10), ("b", 2, 3), ("c", 5, 6)])
    spy = Spy(hours=2.0)
    replay(sessions, spy)
    times = [e[1] for e in spy.events]
    assert times == sorted(times)
    kinds = [(e[0], e[2] if e[0] == "decision" else e[2].session_id) for e in spy.events]
    assert kinds.index(("decision", "b")) < kinds.index(("outcome", "a"))


def test_renewed_promises_while_the_car_is_plugged_in():
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
    decisions = replay(table([("a", 0, 2.5)]), FixedPromise(1.0), every_h=1.0)
    assert list(decisions["elapsed_h"]) == pytest.approx([0.0, 1.0, 2.0])
    assert list(decisions["kept"]) == [True, True, False]


class _Abstain(Rule):
    name = "abstain"

    def promise(self, request, history):
        return None


class _Returns(Rule):
    def __init__(self, value):
        self.value, self.name = value, f"returns {value}"

    def promise(self, request, history):
        return self.value


def test_no_promise_is_recorded_as_not_offered():
    decisions = replay(table([("a", 0, 1)]), _Abstain())
    assert not decisions.loc[0, "offered"]
    assert math.isnan(decisions.loc[0, "promised_h"])
    assert pd.isna(decisions.loc[0, "kept"])


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_promises_are_refused(bad):
    with pytest.raises(ValueError):
        replay(table([("a", 0, 1)]), _Returns(bad))


class _KeepsHistory(Rule):
    name = "keeps history"

    def __init__(self):
        self.saved = None

    def promise(self, request, history):
        if self.saved is not None:
            self.saved.window()
        self.saved = history
        return 0.0


def test_a_history_cannot_be_read_after_its_decision():
    with pytest.raises(RuntimeError):
        replay(table([("a", 0, 1), ("b", 2, 3)]), _KeepsHistory())
