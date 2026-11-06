"""Day-level rules (analysis plan, Section 5, arms 3 to 6): per-car bounds at one shared level.

Arms 4, 5 and 6 give every car present at 10:00 its bounds at one shared quantile level,
chosen from earlier days so that every present car's bounds would have held on 90% of them.

**The day score.** For a car and a target (stay or energy), its score is the largest level on
`GRID` at which its bound, as the engine uses it (the stay rounded down to the 15-minute grid
and capped at the horizon), held. The level is the standardisation the analysis plan asks
for: it measures each shortfall on the probability scale of the model's own predicted
distribution, which is fitted on training data only. The day's score is the smallest over its
cars and both targets: every bound held at level q if and only if q is at most the day's
score, because each car's bound rises with the level. Each day's score is computed from the
predictions made on that day, under the model then in force, and stored; it is never
recomputed.

**The rules.** With n scored days in the window, a rule at miss level a uses the k-th
smallest of their scores, k = floor(a (n + 1)); the trivial bound (deadline at 10:00, energy
0, level 0) if k < 1; the top of the grid if k > n.

- Primary: adaptive conformal inference (Gibbs and Candes, 2021, equation (2)) with days as
  steps. The working miss level starts at a_1 = 0.10 and moves after each scored day as
  a <- a + gamma (0.10 - err), with err = 1 if some bound failed at the level used that day.
  A working level at or below 0 gives the trivial bound, which cannot fail; one at or above 1
  gives the top of the grid, which may hold, so of Proposition 4.1 only the side that bounds
  the share of failed days from above is kept (see `ACI_GAMMA`). Updates come as outcomes are
  revealed, in that order.
- Secondary: split conformal on the window (a = 0.10), only where the window holds at least
  40 scored days (check D8); on other days the secondary arm uses its own copy of the primary
  rule, and the log says which rule ran.

**Arm 3, the union bound.** Each car's two bounds at level 0.10 / (2n) for n present cars.
Where check D10 does not support a level (a level between two grid levels needs both), that
bound is the trivial one, and the cars given it are counted.

**Calibration on delivery** (`DeliveryCalibrated`, measured on the model-selection months, not
part of the registered design). The same rules, with a different day score: the largest level
of `DELIVERY_LEVELS` such that the day's offer, made again once the day's outcome is known from
the bounds the model gave that morning, was delivered under the allocation fixed at 10:00 at
that level and at every lower one. A day on which the offer at the level used was not
delivered has a score below that level, so the share of failed days the rule controls is at
least the share of days not delivered.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fleetoffer.engine import ENERGY_REMAINING_KWH, to_steps
from fleetoffer.interface import DayOutcome, DayView, FleetRule
from fleetoffer.offer import (
    CAP_KW,
    DELIVERY_TOL_KWH,
    STEP_H,
    STEPS,
    Fleet,
    Offer,
    SegmentOffer,
    fixed_shortfall,
)
from fleetoffer.quantiles import LEVELS, PerCarRule

GRID = np.concatenate([[0.0], np.round(np.arange(0.002, 0.95 + 1e-9, 0.001), 3)])
TARGET_MISS = 0.10  # the offer may fail on at most 10% of days
WINDOW_DAYS = 90  # the calibration window, the same as the models' history window
SPLIT_MIN_DAYS = 40  # D8
ACI_START = 0.10  # a_1
# Over T scored days that begin at working level a_S, the recursion gives
# a_{S+T} = a_S + gamma * sum(0.10 - err_t), so the share of failed days among them is
# 0.10 + (a_S - a_{S+T}) / (gamma T). A level at or below 0 gives the trivial bound, which
# cannot fail, so with each outcome known before the next decision the working level never
# falls below -0.9 gamma and the share is at most 0.10 + (a_S + 0.9 gamma) / (gamma T); the
# rule starts at a_S = a_1, and a later stretch uses the level logged at its start. The step is
# the smallest, to two decimals, that keeps this within 0.05 of the target from a_1 over the
# 116 scored weekdays of the frozen development evaluation (check D3 counts 130 weekdays, 14 of
# them with fewer than five cars with energy): gamma >= a_1 / (0.05 T - 0.9) = 0.1 / 4.9 = 0.0204.
ACI_GAMMA = 0.03
ACI_HORIZON_DAYS = 116
ACI_MARGIN = 0.05
SCORED_MIN_CARS = 5  # a day is scored if at least this many cars have energy remaining
SCORED_ENERGY_KWH = ENERGY_REMAINING_KWH  # "energy remaining", as in check D3
_ENERGY_TOL_KWH = 1e-9  # the engine's tolerance on an energy bound


def at_levels(predictions: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Each row's quantile at its own level, exactly as `quantiles.at_level` reads one level
    off the grid: linear interpolation between neighbouring grid levels, 0 below the lowest,
    the top level above it; NaN rows give NaN."""
    predictions = np.asarray(predictions, dtype=float)
    levels = np.broadcast_to(np.asarray(levels, dtype=float), (len(predictions),))
    top = len(LEVELS) - 1
    clipped = np.minimum(levels, LEVELS[-1])
    k = np.clip(np.searchsorted(LEVELS, clipped, side="right") - 1, 0, top)
    at_grid = (k >= top) | (LEVELS[k] == clipped)
    k1 = np.minimum(k + 1, top)
    span = np.where(k1 > k, LEVELS[k1] - LEVELS[k], 1.0)
    share = np.where(at_grid, 0.0, (clipped - LEVELS[k]) / span)
    rows = np.arange(len(predictions))
    value = np.where(
        at_grid,
        predictions[rows, k],
        (1 - share) * predictions[rows, k] + share * predictions[rows, k1],
    )
    return np.where(levels < LEVELS[0], 0.0, value)


def _holds(bound: np.ndarray, truth: np.ndarray, target: str) -> np.ndarray:
    if target == "stay":
        steps = np.floor(np.minimum(bound, STEPS * STEP_H) / STEP_H + 1e-9)
        return steps <= truth
    return bound <= truth + _ENERGY_TOL_KWH


def held_level(predictions: np.ndarray, truth: np.ndarray, target: str) -> np.ndarray:
    """For each car, the largest level on `GRID` at which its bound held (NaN rows: the top,
    since a car without a prediction is given the trivial bound, which always holds).

    `truth` is the true deadline in steps for "stay" and the energy taken in kWh for "energy".
    """
    predictions = np.asarray(predictions, dtype=float)
    truth = np.asarray(truth, dtype=float)
    n = len(predictions)
    missing = np.isnan(predictions).any(axis=1)
    lo = np.zeros(n, dtype=int)  # level 0 always holds
    hi = np.full(n, len(GRID))  # one past the top
    while (hi - lo > 1).any():
        mid = (lo + hi) // 2
        probe = np.minimum(mid, len(GRID) - 1)
        ok = _holds(np.nan_to_num(at_levels(predictions, GRID[probe])), truth, target)
        active = hi - lo > 1
        lo = np.where(active & ok, mid, lo)
        hi = np.where(active & ~ok, mid, hi)
    out = GRID[lo]
    out[missing] = GRID[-1]
    return out


def day_score(stay, energy, true_deadline, true_energy) -> float:
    """The smallest held level over the day's cars and both targets."""
    levels = np.minimum(
        held_level(stay, true_deadline, "stay"), held_level(energy, true_energy, "energy")
    )
    return float(levels.min()) if len(levels) else float(GRID[-1])


def aci_failure_bound(
    days: int, gamma: float = ACI_GAMMA, start: float = ACI_START, target: float = TARGET_MISS
) -> float:
    """The most the share of failed days over `days` scored days can be, from working level
    `start` on the first of them, with each outcome known before the next decision:
    target + (start + (1 - target) gamma) / (gamma T)."""
    return target + (start + (1 - target) * gamma) / (gamma * days)


def aci_step(
    days: int = ACI_HORIZON_DAYS,
    margin: float = ACI_MARGIN,
    start: float = ACI_START,
    target: float = TARGET_MISS,
) -> float:
    """The smallest step keeping `aci_failure_bound` within `margin` of the target."""
    if margin * days <= 1 - target:
        raise ValueError("no step reaches that margin in that many days")
    return start / (margin * days - (1 - target))


def conformal_level(scores: np.ndarray, miss: float) -> float:
    """The k-th smallest score, k = floor(miss (n + 1)); 0 if k < 1, the top if k > n."""
    n = len(scores)
    if miss <= 0:
        return 0.0
    k = math.floor(miss * (n + 1) + 1e-12)
    if k < 1:
        return 0.0
    if k > n:
        return float(GRID[-1])
    return float(np.sort(scores)[k - 1])


def is_scored(true_energy: np.ndarray) -> bool:
    """A day is scored if at least five present cars have more than 0.5 kWh remaining."""
    return int((np.asarray(true_energy) > SCORED_ENERGY_KWH).sum()) >= SCORED_MIN_CARS


@dataclass
class _Pending:
    stay: np.ndarray
    energy: np.ndarray
    level: float
    rule: str
    fitted: bool
    miss: float  # the working miss level at the decision


class Calibrated(FleetRule):
    """Arms 4, 5 and 6: the per-car model of `per_car` (its first model), at one shared level
    chosen by adaptive conformal inference ("aci", primary) or split conformal ("split",
    secondary, with adaptive conformal inference where the window is too short)."""

    def __init__(
        self,
        per_car: PerCarRule,
        *,
        rule: str = "aci",
        gamma: float = ACI_GAMMA,
        start: float = ACI_START,
        target: float = TARGET_MISS,
        window_days: float = WINDOW_DAYS,
        min_days: int = SPLIT_MIN_DAYS,
        name: str | None = None,
    ) -> None:
        if rule not in ("aci", "split"):
            raise ValueError("rule must be 'aci' or 'split'")
        self.per_car = per_car
        self.rule = rule
        self.gamma, self.target = gamma, target
        self.window_days, self.min_days = window_days, min_days
        self.miss = start  # the working miss level of adaptive conformal inference
        self.name = name or f"{per_car.name}+{rule}"
        self.scores: list[tuple[pd.Timestamp, float]] = []  # scored days only
        self.log: list[dict] = []
        self._pending: dict[pd.Timestamp, _Pending] = {}

    def _window(self, now: pd.Timestamp) -> np.ndarray:
        since = now - pd.Timedelta(days=self.window_days)
        return np.array([s for t, s in self.scores if t >= since], dtype=float)

    def level(self, now: pd.Timestamp) -> tuple[float, str]:
        window = self._window(now)
        if self.rule == "split" and len(window) >= self.min_days:
            return conformal_level(window, self.target), "split"
        if self.miss >= 1:
            return float(GRID[-1]), "aci"
        return conformal_level(window, self.miss), "aci"

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        self.per_car.bounds(day)
        stay, energy = self.per_car.last_grid
        fitted = not np.isnan(stay).all()
        level, used = self.level(day.decision_time)
        self._pending[day.decision_time] = _Pending(stay, energy, level, used, fitted, self.miss)
        return tuple(np.nan_to_num(at_levels(p, level), nan=0.0) for p in (stay, energy))

    def _score(self, day: _Pending, outcome: DayOutcome) -> tuple[float, dict]:
        """The day's score, and anything to log beside it."""
        return day_score(day.stay, day.energy, outcome.true_deadline, outcome.true_energy), {}

    def observe(self, outcome: DayOutcome) -> None:
        self.per_car.observe(outcome)
        day = self._pending.pop(outcome.decision_time)
        row = {
            "decision_time": outcome.decision_time,
            "revealed_at": outcome.revealed_at,
            "level": day.level,
            "rule": day.rule,
            "miss_at_decision": day.miss,
            "fitted": day.fitted,
        }
        if day.fitted:
            score, extra = self._score(day, outcome)
            scored = is_scored(outcome.true_energy)
            failed = day.level > score
            row |= {"score": score, "scored": scored, "failed": failed} | extra
            if scored:
                self.scores.append((outcome.decision_time, score))
                self.miss += self.gamma * (self.target - float(failed))
        self.log.append(row)


DELIVERY_LEVELS = np.concatenate([[0.0], LEVELS])  # the levels at which a day is made again
MakeOffer = Callable[[Fleet], Offer | SegmentOffer]


def fleet_at(
    stay: np.ndarray,
    energy: np.ndarray,
    level: float,
    *,
    cap: float = CAP_KW,
    steps: int = STEPS,
    step_h: float = STEP_H,
) -> Fleet:
    """The learned sets at one shared level, as the engine builds them from the bounds."""
    stay_h = np.nan_to_num(at_levels(stay, level), nan=0.0)
    ceiling = np.nan_to_num(at_levels(energy, level), nan=0.0)
    deadline = to_steps(np.minimum(stay_h, steps * step_h), step_h, steps)
    return Fleet.of(deadline, ceiling, cap, steps, step_h)


def fixed_delivered(offer: Offer | SegmentOffer, true: Fleet) -> bool:
    """The primary check alone: every checked profile delivered under the allocation fixed at
    the decision time (`offer.check_delivery`, without the hindsight split)."""
    return all(
        float(fixed_shortfall(allocation, true).sum()) <= DELIVERY_TOL_KWH
        for _, allocation in offer.checked().values()
    )


def delivery_score(
    stay: np.ndarray,
    energy: np.ndarray,
    true: Fleet,
    make_offer: MakeOffer,
    levels: np.ndarray = DELIVERY_LEVELS,
    cap: float = CAP_KW,
) -> tuple[float, int]:
    """The largest of `levels` at which the offer, and the offer at every lower one, was
    delivered under the fixed allocation; and how many offers were made to find it. Level 0
    gives the trivial sets, whose offer charges nothing and is always delivered."""
    for k, level in enumerate(levels):
        learned = fleet_at(stay, energy, level, cap=cap, steps=true.steps, step_h=true.step_h)
        if not fixed_delivered(make_offer(learned), true):
            return float(levels[max(k - 1, 0)]), k + 1
    return float(levels[-1]), len(levels)


class DeliveryCalibrated(Calibrated):
    """Arms 4, 5 and 6 calibrated on delivery (module docstring): the day's score is the
    largest level at which `make_offer`, applied to that morning's bounds, was delivered.
    `make_offer` must be the offer the engine makes on the same days."""

    def __init__(
        self,
        per_car: PerCarRule,
        make_offer: MakeOffer,
        *,
        levels: np.ndarray = DELIVERY_LEVELS,
        cap: float = CAP_KW,
        name: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(per_car, name=name, **kwargs)
        if name is None:
            self.name = f"{per_car.name}+{self.rule}+delivery"
        self.make_offer, self.levels, self.cap = make_offer, np.asarray(levels, float), cap

    def _score(self, day: _Pending, outcome: DayOutcome) -> tuple[float, dict]:
        true = Fleet.of(outcome.true_deadline, outcome.true_energy, self.cap)
        score, offers = delivery_score(
            day.stay, day.energy, true, self.make_offer, self.levels, self.cap
        )
        return score, {"offers_made": offers}


def supported(level: float, levels: tuple[float, ...]) -> bool:
    """D10: a level is supported if it is a supported grid level, or lies between two."""
    if level < LEVELS[0]:
        return False
    if level >= LEVELS[-1]:
        return LEVELS[-1] in levels
    k = int(np.searchsorted(LEVELS, level, side="right")) - 1
    if LEVELS[k] == level:
        return float(LEVELS[k]) in levels
    return float(LEVELS[k]) in levels and float(LEVELS[k + 1]) in levels


class UnionBound(FleetRule):
    """Arm 3: each car's two bounds at level 0.10 / (2n), the trivial bound where D10 does not
    support the level."""

    def __init__(
        self,
        per_car: PerCarRule,
        supported_levels: dict[str, tuple[float, ...]],
        *,
        target: float = TARGET_MISS,
        name: str | None = None,
    ) -> None:
        self.per_car = per_car
        self.supported_levels = supported_levels
        self.target = target
        self.name = name or f"{per_car.name}+union"
        self.log: list[dict] = []

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        self.per_car.bounds(day)
        stay, energy = self.per_car.last_grid
        n = len(day.present)
        level = self.target / (2 * n)
        out = []
        for name, grid in (("stay", stay), ("energy", energy)):
            ok = supported(level, self.supported_levels[name])
            out.append(np.nan_to_num(at_levels(grid, level), nan=0.0) if ok else np.zeros(n))
        self.log.append(
            {
                "decision_time": day.decision_time,
                "cars": n,
                "level": level,
                "stay_supported": supported(level, self.supported_levels["stay"]),
                "energy_supported": supported(level, self.supported_levels["energy"]),
            }
        )
        return out[0], out[1]

    def observe(self, outcome: DayOutcome) -> None:
        self.per_car.observe(outcome)
