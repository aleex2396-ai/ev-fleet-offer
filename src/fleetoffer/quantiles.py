"""Per-car quantile models of remaining stay and of energy before the end of the horizon.

Every model here and in `baselines` answers the same question for each car present at the
decision time: lower quantiles, on the grid `LEVELS`, of (a) the hours the car will still
stay, up to the end of the horizon, and (b) the energy it will take before the end of the
horizon (analysis plan, Section 4). A quantile at level q is a lower bound that the truth
falls below with probability q.

Models are refitted at monthly origins (the first local midnight of each month). A refit at
origin R sees (D11, analysis plan Section 10a):

- the car-days of earlier decisions whose outcome had been revealed before R (the engine
  reveals a day once its last car has unplugged), excluding cars that plugged in within 24
  hours of R, and, if a history window of W days is set, decisions made within W days of R;
- the sessions that had ended before R with the same exclusions (`ended`), for the rules
  that learn from whole sessions rather than from car-days.

`PerCarRule` runs any set of models inside the day-level engine, so the models see exactly
what the engine shows at each decision and learn from outcomes only when the engine reveals
them. It records every model's predictions at every level, and gives the engine the bounds of
its primary model at one level.

The learned families: gradient-boosted decision trees with pinball loss, one model per level
and target (primary); a small neural network with one output per level, non-crossing by
construction (second); a linear quantile regression (baseline). Tree and linear predictions
that cross between levels are sorted, which can only reduce the pinball loss (Chernozhukov,
Fernandez-Val and Galichon, 2010, "Quantile and probability curves without crossing",
Econometrica 78(3)).
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import pickle
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor

from fleetoffer.academic import day_flags
from fleetoffer.engine import EMPTY_KWH
from fleetoffer.features import LOCAL_TZ, NUMERIC, at_decision
from fleetoffer.interface import DayOutcome, DayView, FleetRule
from fleetoffer.offer import STEP_H, STEPS

LEVELS = np.array(
    [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
)
HORIZON_H = STEPS * STEP_H
EXCLUDE_H = 24.0  # D11
TARGETS = ("stay", "energy")
_DAY = pd.Timedelta(days=1)


# Which data a refit at an origin may use.


def month_origin(t: pd.Timestamp, tz: str = LOCAL_TZ) -> pd.Timestamp:
    """The refit origin in force at decision time t: the first local midnight of its month."""
    local = t.tz_convert(tz)
    return pd.Timestamp(year=local.year, month=local.month, day=1, tz=tz).tz_convert("UTC")


def training_rows(
    cardays: pd.DataFrame, origin: pd.Timestamp, window_days: float | None = None
) -> pd.DataFrame:
    """Car-days a refit at `origin` may learn from (D11 and the history window)."""
    keep = (cardays["revealed_at"] < origin) & (
        cardays["connect_time"] <= origin - pd.Timedelta(hours=EXCLUDE_H)
    )
    if window_days is not None:
        keep &= cardays["decision_time"] >= origin - window_days * _DAY
    return cardays[keep].reset_index(drop=True)


def ended_rows(
    ended: pd.DataFrame, origin: pd.Timestamp, window_days: float | None = None
) -> pd.DataFrame:
    """Whole sessions a refit at `origin` may learn from (D11 and the history window)."""
    keep = (ended["disconnect_time"] < origin) & (
        ended["connect_time"] <= origin - pd.Timedelta(hours=EXCLUDE_H)
    )
    if window_days is not None:
        keep &= ended["connect_time"] >= origin - window_days * _DAY
    return ended[keep].reset_index(drop=True)


# The common form of every model.


class QuantileModel:
    """``fit`` at an origin; ``predict`` two arrays of shape (cars, len(LEVELS)), stay in hours
    (at most the horizon) and energy in kWh, with NaN where the model gives no prediction."""

    name = "model"

    def fit(self, train: pd.DataFrame, ended: pd.DataFrame, origin: pd.Timestamp) -> None:
        raise NotImplementedError

    def predict(self, day: DayView, feats: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


def tidy(stay: np.ndarray, energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort across levels (rows with NaN are left as they are) and clip to the ranges."""
    stay = np.clip(_sort_rows(stay), 0.0, HORIZON_H)
    energy = np.clip(_sort_rows(energy), 0.0, None)
    return stay, energy


def _sort_rows(values: np.ndarray) -> np.ndarray:
    values = np.array(values, dtype=float)
    ok = ~np.isnan(values).any(axis=1)
    values[ok] = np.sort(values[ok], axis=1)
    return values


def targets(train: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        "stay": train["target_stay_h"].to_numpy(float),
        "energy": train["target_energy_kwh"].to_numpy(float),
    }


class Design:
    """The input matrix of the learned families, fitted on the training rows only.

    Numeric inputs are standardised by their training mean and spread, missing values (inputs
    that do not apply, such as driver history for an anonymous car) become 0 after the
    indicator of whether they apply (``linked``, ``typed``) has been kept. For the trees
    (``one_hot=False``) the station is one integer column, missing if unseen in training. For
    the network and the linear model (``one_hot=True``) the station and the month are
    indicator columns of the values seen in training: a model that extrapolates along its
    inputs would otherwise read the month, a category, as a trend and carry it outside the
    training window. An unseen value gets weight 1/k on each of the k seen values, so the
    linear model predicts the average of their effects. All zeros would not do: every
    training row has exactly one indicator on, so the indicators add up to the constant term
    and a fit may split an effect between the two in any proportion; the prediction for an
    all-zero row would then depend on that split.
    """

    def __init__(self, train: pd.DataFrame, *, one_hot: bool) -> None:
        self.one_hot = one_hot
        self.numeric = [c for c in NUMERIC if not (one_hot and c == "month")]
        x = train[self.numeric].to_numpy(float)
        with warnings.catch_warnings():  # an input missing on every row is allowed
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.nanmean(x, axis=0)
            spread = np.nanstd(x, axis=0)
        spread = np.where(np.isnan(spread), 0.0, spread)
        self.scale = np.where(spread > 0, spread, 1.0)
        self.mean = np.where(np.isnan(self.mean), 0.0, self.mean)
        self.stations = sorted({s for s in train["station"] if isinstance(s, str)})
        self.months = sorted(set(train["month"].dropna().astype(int)))

    @staticmethod
    def _indicators(values, seen: list) -> np.ndarray:
        code = {v: k for k, v in enumerate(seen)}
        idx = np.array([code.get(v, -1) for v in values], dtype=int)
        hot = np.zeros((len(idx), len(seen)))
        hit = idx >= 0
        hot[np.flatnonzero(hit), idx[hit]] = 1.0
        if seen:
            hot[~hit] = 1.0 / len(seen)
        return hot

    def __call__(self, feats: pd.DataFrame) -> np.ndarray:
        x = (feats[self.numeric].to_numpy(float) - self.mean) / self.scale
        if self.one_hot:
            months = [int(m) if m == m else None for m in feats["month"]]
            return np.hstack(
                [
                    np.nan_to_num(x, nan=0.0),
                    self._indicators(feats["station"], self.stations),
                    self._indicators(months, self.months),
                ]
            )
        code = {s: k for k, s in enumerate(self.stations)}
        idx = np.array([code.get(s, -1) for s in feats["station"]], dtype=float)
        return np.hstack([x, np.where(idx >= 0, idx, np.nan)[:, None]])


# The learned families.

# Arms that share a per-car model fit the same trees on the same rows at the same refits, so
# a run may keep each fit once: with FIT_STORE set to a folder, a fit is saved there under a
# fingerprint of the model's settings and of every training row (values, columns and types),
# and a later fit with the same fingerprint reads it back. A lock per fingerprint makes
# parallel jobs wait for the first fit instead of repeating it. Unset, nothing is kept.
FIT_STORE = "FLEETOFFER_FIT_STORE"


def fit_fingerprint(model: QuantileModel, train: pd.DataFrame) -> str:
    """A hash of the model's class and settings, the library version and every training row."""
    digest = hashlib.sha256()
    digest.update(type(model).__name__.encode())
    digest.update(repr(sorted(asdict(model).items())).encode())
    digest.update(sklearn.__version__.encode())
    digest.update(repr([(c, str(t)) for c, t in train.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(train, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def stored_fit(model: QuantileModel, train: pd.DataFrame, fit: Callable[[], dict]) -> dict:
    """`fit()`'s fitted state, read from the fit store if this exact fit was made before."""
    folder = os.environ.get(FIT_STORE)
    if not folder:
        return fit()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = fit_fingerprint(model, train)
    path = folder / f"{key}.pkl"
    with open(folder / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return pickle.loads(path.read_bytes())
        state = fit()
        partial = path.with_suffix(".partial")
        partial.write_bytes(pickle.dumps(state))
        partial.replace(path)
        return state


# Parallel runs keep one worker process per core, so each worker runs the trees' loops on one
# thread: more threads per worker only compete for the same cores (with every core busy,
# three quantile fits took 1.2 s on one thread and 93 s on ten).
THREADS = "OMP_NUM_THREADS"


@contextmanager
def worker_pool(workers: int) -> Iterator[ProcessPoolExecutor]:
    """A process pool whose workers start with one thread for the tree models' loops."""
    before = os.environ.get(THREADS)
    os.environ[THREADS] = "1"  # each worker reads it when it starts
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield pool
    finally:
        if before is None:
            os.environ.pop(THREADS, None)
        else:
            os.environ[THREADS] = before


@dataclass
class GBDT(QuantileModel):
    """Gradient-boosted trees (scikit-learn's histogram version), pinball loss, one model per
    level and target; the station is a categorical input."""

    learning_rate: float = 0.1
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    max_iter: int = 200
    seed: int = 0
    name: str = "gbdt"

    def fit(self, train, ended, origin):
        state = stored_fit(self, train, lambda: self._fit(train))
        self.design, self.models = state["design"], state["models"]

    def _fit(self, train) -> dict:
        design = Design(train, one_hot=False)
        x = design(train)
        cat = np.zeros(x.shape[1], dtype=bool)
        cat[-1] = True
        models = {}
        for target, y in targets(train).items():
            models[target] = [
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=float(q),
                    learning_rate=self.learning_rate,
                    max_leaf_nodes=self.max_leaf_nodes,
                    min_samples_leaf=self.min_samples_leaf,
                    max_iter=self.max_iter,
                    categorical_features=cat if len(design.stations) else None,
                    early_stopping=False,
                    random_state=self.seed,
                ).fit(x if len(design.stations) else x[:, :-1], y)
                for q in LEVELS
            ]
        return {"design": design, "models": models}

    def predict(self, day, feats):
        x = self.design(feats)
        x = x if len(self.design.stations) else x[:, :-1]
        out = [np.column_stack([m.predict(x) for m in self.models[t]]) for t in TARGETS]
        return tidy(*out)


@dataclass
class LinearQuantiles(QuantileModel):
    """Linear quantile regression on the standardised inputs and station indicators, one per
    level and target, solved exactly as a linear programme (HiGHS), with no penalty."""

    name: str = "linear"

    def fit(self, train, ended, origin):
        self.design = Design(train, one_hot=True)
        x = self.design(train)
        self.models = {
            target: [
                QuantileRegressor(quantile=float(q), alpha=0.0, solver="highs").fit(x, y)
                for q in LEVELS
            ]
            for target, y in targets(train).items()
        }

    def predict(self, day, feats):
        x = self.design(feats)
        out = [np.column_stack([m.predict(x) for m in self.models[t]]) for t in TARGETS]
        return tidy(*out)


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, z)


@dataclass
class QuantileNet(QuantileModel):
    """A one-hidden-layer network (tanh) per target with one output per level.

    The first output is the lowest quantile and each further one adds a softplus increment,
    so the quantiles never cross. Trained with Adam on the mean pinball loss over levels,
    weight decay `weight_decay`, minibatches of 256, at most `epochs` passes; the weights kept
    are those of the pass with the lowest loss on the last 15% of the training car-days in
    time order. Targets are divided by a fixed scale (the horizon for stay, 20 kWh for
    energy).
    """

    hidden: int = 32
    weight_decay: float = 1e-4
    learning_rate: float = 3e-3
    epochs: int = 300
    patience: int = 20
    seed: int = 0
    name: str = "net"

    def fit(self, train, ended, origin):
        self.design = Design(train, one_hot=True)
        order = np.argsort(train["decision_time"].to_numpy(), kind="stable")
        x = self.design(train)[order]
        self.nets = {}
        for k, (target, y) in enumerate(targets(train).items()):
            scale = HORIZON_H if target == "stay" else 20.0
            self.nets[target] = (self._train(x, y[order] / scale, k), scale)

    def _train(self, x: np.ndarray, y: np.ndarray, k: int) -> list[np.ndarray]:
        rng = np.random.default_rng([self.seed, k])
        n, p = x.shape
        cut = int(round(0.85 * n))
        xt, yt, xv, yv = x[:cut], y[:cut], x[cut:], y[cut:]
        levels = len(LEVELS)
        start = np.quantile(yt, LEVELS)  # the output starts at the unconditional quantiles
        rise = np.maximum(np.diff(start), 1e-3)
        params = [
            rng.normal(0.0, 1.0 / np.sqrt(p), (p, self.hidden)),
            np.zeros(self.hidden),
            rng.normal(0.0, 0.1 / np.sqrt(self.hidden), (self.hidden, levels)),
            np.concatenate([start[:1], np.log(np.expm1(rise))]),
        ]
        moments = [(np.zeros_like(w), np.zeros_like(w)) for w in params]
        best, best_loss, waited, step = [w.copy() for w in params], np.inf, 0, 0
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(xt)), max(1, len(xt) // 256)):
                step += 1
                grads = self._grads(params, xt[batch], yt[batch])
                for w, g, (m, v) in zip(params, grads, moments, strict=True):
                    g = g + self.weight_decay * w
                    m *= 0.9
                    m += 0.1 * g
                    v *= 0.999
                    v += 0.001 * g * g
                    w -= (
                        self.learning_rate
                        * (m / (1 - 0.9**step))
                        / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
                    )
            loss = pinball(yv, self._forward(params, xv)[0]).mean() if len(xv) else 0.0
            if loss < best_loss - 1e-7:
                best, best_loss, waited = [w.copy() for w in params], loss, 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
        return best

    @staticmethod
    def _forward(params, x):
        w1, b1, w2, b2 = params
        h = np.tanh(x @ w1 + b1)
        z = h @ w2 + b2
        steps = np.concatenate([z[:, :1], _softplus(z[:, 1:])], axis=1)
        return np.cumsum(steps, axis=1), (h, z)

    def _grads(self, params, x, y):
        w1, b1, w2, b2 = params
        q, (h, z) = self._forward(params, x)
        n, levels = q.shape
        dq = ((y[:, None] < q).astype(float) - LEVELS[None, :]) / (n * levels)
        dsteps = np.cumsum(dq[:, ::-1], axis=1)[:, ::-1]  # each step moves every later level
        dz = dsteps * np.concatenate([np.ones((n, 1)), expit(z[:, 1:])], axis=1)
        dh = (dz @ w2.T) * (1.0 - h * h)
        return [x.T @ dh, dh.sum(axis=0), h.T @ dz, dz.sum(axis=0)]

    def predict(self, day, feats):
        x = self.design(feats)
        out = []
        for target in TARGETS:
            params, scale = self.nets[target]
            out.append(self._forward(params, x)[0] * scale)
        return tidy(*out)


# Scores.


def pinball(y: np.ndarray, q: np.ndarray, levels: np.ndarray = LEVELS) -> np.ndarray:
    """Pinball loss of each prediction: rows are cars, columns levels."""
    diff = np.asarray(y, dtype=float)[:, None] - np.asarray(q, dtype=float)
    return np.maximum(levels * diff, (levels - 1.0) * diff)


def at_level(predictions: np.ndarray, level: float) -> np.ndarray:
    """Each row's quantile at `level`, read off the grid by linear interpolation between
    neighbouring levels; below the lowest level the trivial bound 0; NaN rows give NaN."""
    predictions = np.asarray(predictions, dtype=float)
    if level < LEVELS[0]:
        return np.zeros(len(predictions))
    level = min(level, LEVELS[-1])
    k = int(np.searchsorted(LEVELS, level, side="right")) - 1
    if k >= len(LEVELS) - 1 or LEVELS[k] == level:
        return predictions[:, k]
    share = (level - LEVELS[k]) / (LEVELS[k + 1] - LEVELS[k])
    return (1 - share) * predictions[:, k] + share * predictions[:, k + 1]


# The models inside the engine.


class PerCarRule(FleetRule):
    """Runs per-car models in the day-level engine and records what they predict.

    At each decision the inputs are built from the day view (`features.at_decision`); at the
    first decision of a new month every model is refitted on what is known at the month's
    origin, if at least `min_rows` car-days are; until then a model gives no prediction. The
    engine is given the primary model's bounds at `level` (0 where it has none). Each day's
    inputs are kept, and its outcome is attached to them when the engine reveals it.
    """

    def __init__(
        self,
        models: list[QuantileModel],
        *,
        level: float = 0.1,
        window_days: float | None = None,
        min_rows: int = 200,
        name: str | None = None,
        calendar: Callable[..., pd.DataFrame] = day_flags,
        energy_left_only: bool = False,
    ) -> None:
        if not models:
            raise ValueError("give at least one model")
        self.models = models
        self.level = level
        self.window_days = window_days
        self.min_rows = min_rows
        self.name = name or models[0].name
        self.calendar = calendar
        self.energy_left_only = energy_left_only  # train on car-days that took energy only
        self.frozen = False  # no refit from now on (`evaluation.Frozen`)
        self.hide_history = False  # no other session's history in the inputs (JPL, track 1)
        self.keep_fit_if_short = False  # keep the last fit while a new window is too short
        self.origin: pd.Timestamp | None = None
        self.fitted: dict[str, bool] = {}
        self.fit_log: list[dict] = []
        self.records: list[pd.DataFrame] = []
        self.ended_at_origin: dict[pd.Timestamp, pd.DataFrame] = {}
        self._waiting: dict[pd.Timestamp, pd.DataFrame] = {}
        self._known: list[pd.DataFrame] = []
        self.last_grid: tuple[np.ndarray, np.ndarray] | None = None

    def cardays(self) -> pd.DataFrame:
        """Every car-day whose outcome has been revealed so far, inputs and targets."""
        if not self._known:
            return pd.DataFrame()
        return pd.concat(self._known, ignore_index=True)

    def bounds(self, day: DayView) -> tuple[np.ndarray, np.ndarray]:
        if self.hide_history:
            day = replace(day, ended=day.ended.iloc[:0])
        feats = at_decision(day, calendar=self.calendar)
        origin = month_origin(day.decision_time)
        if origin != self.origin and not self.frozen:
            self._refit(origin, day)
        self._waiting[day.decision_time] = feats
        columns: dict[str, np.ndarray] = {}
        primary = None
        for model in self.models:
            if self.fitted.get(model.name):
                stay, energy = model.predict(day, feats)
            else:
                stay = energy = np.full((len(feats), len(LEVELS)), np.nan)
            for k, q in enumerate(LEVELS):
                columns[f"{model.name}:stay:{q:g}"] = stay[:, k]
                columns[f"{model.name}:energy:{q:g}"] = energy[:, k]
            if primary is None:
                primary = (stay, energy)
        keys = feats[["decision_time", "session_id", "linked", "n_past", "typed"]].assign(
            user_id=day.present["user_id"].to_numpy(object)
        )
        self.records.append(pd.concat([keys, pd.DataFrame(columns, index=keys.index)], axis=1))
        self.last_grid = primary  # the primary model's predictions at every level, this day
        stay, energy = (np.nan_to_num(at_level(p, self.level), nan=0.0) for p in primary)
        return stay, energy

    def _refit(self, origin: pd.Timestamp, day: DayView) -> None:
        self.origin = origin
        known = self.cardays()
        train = training_rows(known, origin, self.window_days) if len(known) else known
        if self.energy_left_only and len(train):
            train = train[train["target_energy_kwh"] > EMPTY_KWH].reset_index(drop=True)
        ended = ended_rows(day.ended, origin, self.window_days)
        self.ended_at_origin[origin] = ended_rows(day.ended, origin)
        for model in self.models:
            ok = len(train) >= self.min_rows
            kept = not ok and self.keep_fit_if_short and self.fitted.get(model.name, False)
            start = time.perf_counter()
            if ok:
                model.fit(train, ended, origin)
            self.fitted[model.name] = ok or kept
            self.fit_log.append(
                {
                    "model": model.name,
                    "origin": origin,
                    "rows": len(train),
                    "fitted": ok,
                    "kept_earlier_fit": kept,
                    "seconds": time.perf_counter() - start,
                }
            )

    def forget_history(self) -> None:
        """Drop every car-day seen so far and keep the fitted models (JPL, track 2: the
        Caltech-trained pipeline then learns from JPL outcomes only)."""
        self._known = []
        self._waiting = {}
        self.keep_fit_if_short = True

    def observe(self, outcome: DayOutcome) -> None:
        feats = self._waiting.pop(outcome.decision_time)
        if not np.array_equal(feats["session_id"].to_numpy(), outcome.session_id):
            raise RuntimeError("a day's outcome does not match the cars it was asked about")
        feats = feats.copy()
        feats["target_stay_h"] = outcome.true_deadline * STEP_H
        feats["target_energy_kwh"] = outcome.true_energy
        feats["revealed_at"] = outcome.revealed_at
        self._known.append(feats)

    def predictions(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.concat(self.records, ignore_index=True)
