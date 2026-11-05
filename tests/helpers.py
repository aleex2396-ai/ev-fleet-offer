"""Small tables written by hand, and rules used only by tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fleetoffer.data import to_hours
from fleetoffer.engine import energy_in_window
from fleetoffer.interface import (
    DayOutcome,
    DayView,
    FleetRule,
    History,
    Outcome,
    Request,
    Rule,
)

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


class FixedBound(Rule):
    """The same bound for every car; a reference in checks."""

    def __init__(self, hours: float) -> None:
        self.hours = float(hours)
        self.name = f"fixed {hours:g} h"

    def bound(self, request: Request, history: History) -> float:
        return self.hours


class CountsLaterSessions(Rule):
    """Cheats with counts alone: bounds each car by how many sessions the table holds."""

    name = "cheat: counts later sessions"

    def __init__(self, sessions: pd.DataFrame) -> None:
        self.hours = len(sessions) / 1000.0

    def bound(self, request: Request, history: History) -> float:
        return self.hours


class Spy(Rule):
    """Bounds every car at a fixed number of hours and records everything it is shown."""

    name = "spy"

    def __init__(self, hours: float = 1.0) -> None:
        self.hours = hours
        self.events: list[tuple] = []
        self.windows: dict[str, object] = {}

    def bound(self, request: Request, history: History) -> float:
        self.events.append(("decision", request.decision_h, request.session_id))
        self.windows[request.session_id] = history.window()
        return self.hours

    def observe(self, outcome: Outcome) -> None:
        self.events.append(("outcome", outcome.revealed_h, outcome))


class StepAdjust(Rule):
    """Learns only from outcomes: bound more after a bound held, less after one failed."""

    name = "step adjust"

    def __init__(self) -> None:
        self.hours = 1.0

    def bound(self, request: Request, history: History) -> float:
        return self.hours

    def observe(self, outcome: Outcome) -> None:
        self.hours = max(0.0, self.hours + (0.05 if outcome.held else -0.45))


class PeekAtDeparture(Rule):
    """Cheats: built from the full table, it bounds at 99% of each car's true remaining stay."""

    name = "cheat: peek"

    def __init__(self, sessions: pd.DataFrame) -> None:
        self.departure_h = dict(
            zip(sessions["session_id"], to_hours(sessions["disconnect_time"]), strict=True)
        )

    def bound(self, request: Request, history: History) -> float:
        return 0.99 * max(self.departure_h[request.session_id] - request.decision_h, 0.0)


class QuantileOfEverything(Rule):
    """Cheats more quietly: one setting fitted on every stay, past and future."""

    name = "cheat: fitted on all data"

    def __init__(self, sessions: pd.DataFrame) -> None:
        stays = (sessions["disconnect_time"] - sessions["connect_time"]) / pd.Timedelta(hours=1)
        self.hours = float(np.quantile(stays, 0.1))

    def bound(self, request: Request, history: History) -> float:
        return self.hours


# Day-level rules used only by tests.


class FleetPast(FleetRule):
    """Lower quantiles of stays and energy totals of sessions ended in the last 28 days,
    shrunk by a factor that falls each time an observed day had a failed bound. Records
    what it saw so tests can check the order in which things became known."""

    name = "past quantiles"

    def __init__(self, level: float = 0.2) -> None:
        self.level = level
        self.scale = 1.0
        self.seen: list[tuple[pd.Timestamp, list[DayOutcome]]] = []
        self.observed: list[DayOutcome] = []

    def bounds(self, day: DayView):
        self.seen.append((day.decision_time, list(self.observed)))
        recent = day.ended[day.ended["connect_time"] >= day.decision_time - pd.Timedelta(days=28)]
        n = len(day.present)
        if len(recent) < 5:
            return np.zeros(n), np.zeros(n)
        stays = (recent["disconnect_time"] - recent["connect_time"]) / pd.Timedelta(hours=1)
        stay = float(np.quantile(stays, self.level))
        energy = float(np.quantile(recent["kwh_delivered"], self.level)) * 0.5
        remaining = np.maximum(stay - day.present["elapsed_h"].to_numpy(), 0.0)
        return self.scale * remaining, np.full(n, self.scale * energy)

    def observe(self, outcome: DayOutcome) -> None:
        self.observed.append(outcome)
        failed = (outcome.learned_deadline > outcome.true_deadline) | (
            outcome.learned_energy > outcome.true_energy
        )
        if failed.any():
            self.scale *= 0.8


class FleetOracle(FleetRule):
    """Reads each present car's true remaining stay and energy: not a rule, a reference."""

    name = "oracle"

    def __init__(self, sessions: pd.DataFrame) -> None:
        self.table = sessions.set_index("session_id")

    def bounds(self, day: DayView):
        rows = self.table.loc[day.present["session_id"]]
        t = to_hours(pd.Series([day.decision_time]))[0]
        stay = to_hours(rows["disconnect_time"]) - t
        energy = energy_in_window(
            to_hours(rows["connect_time"]),
            to_hours(rows["done_charging_time"]),
            to_hours(rows["disconnect_time"]),
            rows["kwh_delivered"].to_numpy(float),
            t,
            t + 10.0,
        )
        return stay, energy


class FleetPeekEnergy(FleetPast):
    """Like FleetPast, but adds a sliver of each present car's recorded total: look-ahead."""

    name = "peeks at energy"

    def __init__(self, sessions: pd.DataFrame) -> None:
        super().__init__()
        self.table = sessions.set_index("session_id")

    def bounds(self, day: DayView):
        stay, energy = super().bounds(day)
        totals = self.table.loc[day.present["session_id"], "kwh_delivered"].to_numpy(float)
        return stay, energy + 1e-3 * totals
