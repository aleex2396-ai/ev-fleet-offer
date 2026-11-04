"""Small tables written by hand, and rules used only by tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from staypromise.interface import History, Outcome, Request, Rule
from staypromise.sessions import to_hours

T0 = pd.Timestamp("2001-01-01 00:00", tz="UTC")


def table(rows: list[tuple]) -> pd.DataFrame:
    """Rows of (session_id, start hour, end hour[, user_id]) counted from T0."""
    ids = [r[0] for r in rows]
    start = pd.to_timedelta([r[1] for r in rows], unit="h")
    end = pd.to_timedelta([r[2] for r in rows], unit="h")
    users = [r[3] if len(r) > 3 else None for r in rows]
    return pd.DataFrame(
        {
            "session_id": ids,
            "connect_time": T0 + start,
            "disconnect_time": T0 + end,
            "user_id": users,
        }
    )


def hours_from_t0(h: float) -> float:
    return float(to_hours(pd.Series([T0])).item() + h)


class Spy(Rule):
    """Promises a fixed number of hours and records everything it is shown, in order."""

    name = "spy"

    def __init__(self, hours: float = 1.0) -> None:
        self.hours = hours
        self.events: list[tuple] = []
        self.windows: dict[str, object] = {}

    def promise(self, request: Request, history: History) -> float:
        self.events.append(("decision", request.decision_h, request.session_id))
        self.windows[request.session_id] = history.window()
        return self.hours

    def observe(self, outcome: Outcome) -> None:
        self.events.append(("outcome", outcome.revealed_h, outcome))


class StepAdjust(Rule):
    """Learns only from outcomes: promise more after a kept promise, less after a broken one."""

    name = "step adjust"

    def __init__(self) -> None:
        self.hours = 1.0

    def promise(self, request: Request, history: History) -> float:
        return self.hours

    def observe(self, outcome: Outcome) -> None:
        self.hours = max(0.0, self.hours + (0.05 if outcome.kept else -0.45))


class PeekAtDeparture(Rule):
    """Cheats: built from the full table, it promises 99% of each car's true remaining stay."""

    name = "cheat: peek"

    def __init__(self, sessions: pd.DataFrame) -> None:
        self.departure_h = dict(
            zip(sessions["session_id"], to_hours(sessions["disconnect_time"]), strict=True)
        )

    def promise(self, request: Request, history: History) -> float:
        return 0.99 * max(self.departure_h[request.session_id] - request.decision_h, 0.0)


class QuantileOfEverything(Rule):
    """Cheats more quietly: one setting fitted on every stay, past and future."""

    name = "cheat: fitted on all data"

    def __init__(self, sessions: pd.DataFrame) -> None:
        stays = (sessions["disconnect_time"] - sessions["connect_time"]) / pd.Timedelta(hours=1)
        self.hours = float(np.quantile(stays, 0.1))

    def promise(self, request: Request, history: History) -> float:
        return self.hours
