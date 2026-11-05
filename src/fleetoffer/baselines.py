"""The per-car baselines of the analysis plan (Section 4), in the form of `quantiles`.

- `Population`: the population rule of arms 2 and 5. Among the sessions that ended in the
  history window, those that lasted at least as long as the car has been plugged in; the
  Kaplan-Meier lower quantile of their remaining stay (`models.lower_quantile`; with every
  session ended it equals the empirical one), and the lower quantile of the energy each took
  in the horizon-long window starting that long after its plug-in.
- `Stratified`: the table of arm 6. Lower quantiles of the 10:00 targets of earlier car-days
  in the same cell of plug-in hour, weekday type and link status; a cell with fewer than
  `min_cell` car-days is merged as stated in the class.
- `DriverPast`: each driver's own earlier sessions, conditioned on the time plugged in as
  the population rule is; the population rule for anonymous cars and for drivers with fewer
  than `min_sessions` such sessions.
- `Mixture`: the Gaussian-mixture predictor of Lee, Li and Low (2019, Sections 4 and 5.1): a
  mixture over (arrival hour, stay, energy) fitted on whole sessions; given the arrival
  hour, each component's stay is Gaussian with the conditional mean of their equation (3)
  and the matching conditional variance, and the components are reweighted by the density of
  the arrival hour. Their prediction is the mean of that mixture; here it is converted to
  lower quantiles of remaining stay by conditioning on the stay being at least the time
  already plugged in. With ``per_driver`` the component weights are refitted on each linked
  driver's own sessions (their I-GMM), for drivers with at least `min_driver_sessions`. The
  plan uses this predictor for stay only, so its energy is not given.
- `TypedPlan`: the plan typed in the app before 10:00, at face value: the typed departure,
  and the typed energy spread evenly over the typed stay. The same value at every level;
  no prediction for a car without a typed plan.

"Lower quantile at level q" means the largest x with an estimated chance of at most q that
the truth is below x; the empirical version is the smallest observed value whose cumulative
share reaches q (``method="inverted_cdf"``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from fleetoffer.data import to_hours
from fleetoffer.engine import energy_in_window
from fleetoffer.features import LOCAL_TZ
from fleetoffer.models import km_curve, lower_quantile
from fleetoffer.quantiles import HORIZON_H, LEVELS, QuantileModel, tidy

_HOUR = pd.Timedelta(hours=1)


def _empirical(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.full(len(LEVELS), np.nan)
    return np.quantile(values, LEVELS, method="inverted_cdf")


@dataclass
class _Sessions:
    """Ended sessions as arrays (hours since 1970 UTC)."""

    connect: np.ndarray
    disconnect: np.ndarray
    done: np.ndarray
    kwh: np.ndarray
    user: np.ndarray

    @classmethod
    def of(cls, ended: pd.DataFrame) -> _Sessions:
        return cls(
            to_hours(ended["connect_time"]),
            to_hours(ended["disconnect_time"]),
            to_hours(ended["done_charging_time"]),
            ended["kwh_delivered"].to_numpy(float),
            ended["user_id"].to_numpy(object),
        )

    @property
    def stay(self) -> np.ndarray:
        return self.disconnect - self.connect

    def subset(self, keep: np.ndarray) -> _Sessions:
        return _Sessions(
            self.connect[keep],
            self.disconnect[keep],
            self.done[keep],
            self.kwh[keep],
            self.user[keep],
        )

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        return km_curve(self.stay, np.ones(len(self.stay), dtype=bool))

    def conditional(
        self, elapsed_h: float, curve: tuple[np.ndarray, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lower quantiles of remaining stay and of horizon energy given `elapsed_h`."""
        stay = self.stay
        if len(stay) == 0:
            return np.full(len(LEVELS), np.nan), np.full(len(LEVELS), np.nan)
        times, survival = self.curve() if curve is None else curve
        quantiles = [lower_quantile(times, survival, float(q), elapsed_h) for q in LEVELS]
        remaining = np.array([np.nan if x is None else x for x in quantiles])
        lasted = stay >= elapsed_h
        start = self.connect[lasted] + elapsed_h
        energy = energy_in_window(
            self.connect[lasted],
            self.done[lasted],
            self.disconnect[lasted],
            self.kwh[lasted],
            start,
            start + HORIZON_H,
        )
        return remaining, _empirical(energy)


@dataclass
class Population(QuantileModel):
    """Arms 2 and 5: the site's recently ended sessions, conditioned on time plugged in."""

    name: str = "population"

    def fit(self, train, ended, origin):
        self.sessions = _Sessions.of(ended)
        self.km = self.sessions.curve()

    def predict(self, day, feats):
        elapsed = feats["elapsed_h"].to_numpy(float)
        rows = [self.sessions.conditional(e, self.km) for e in elapsed]
        return tidy(np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))


HOUR_BINS = (6.0, 7.0, 8.0, 9.0)  # cells: before 06:00 (or an earlier day), 06, 07, 08, 09


def stratum(feats: pd.DataFrame) -> pd.DataFrame:
    """The three keys of the stratified table: plug-in hour bin, Friday or not, linked."""
    return pd.DataFrame(
        {
            "hour": np.searchsorted(HOUR_BINS, feats["plugin_hour"].to_numpy(float), side="right"),
            "friday": (feats["weekday"].to_numpy(float) == 4).astype(int),
            "linked": feats["linked"].to_numpy(float).astype(int),
        }
    )


@dataclass
class Stratified(QuantileModel):
    """Arm 6: quantiles of earlier 10:00 car-days in the car's cell. A cell with fewer than
    `min_cell` car-days drops the weekday type, then the plug-in hour, then the link."""

    min_cell: int = 30
    name: str = "stratified"

    def fit(self, train, ended, origin):
        keys = stratum(train)
        stay, energy = (
            train["target_stay_h"].to_numpy(float),
            train["target_energy_kwh"].to_numpy(float),
        )
        self.tables = []
        for cols in (["hour", "friday", "linked"], ["hour", "linked"], ["linked"], []):
            table = {}
            groups = keys.groupby(cols).indices if cols else {(): np.arange(len(keys))}
            for key, idx in groups.items():
                if len(idx) >= self.min_cell or not cols:
                    key = key if isinstance(key, tuple) else (key,)
                    table[key] = (_empirical(stay[idx]), _empirical(energy[idx]))
            self.tables.append((cols, table))

    def predict(self, day, feats):
        keys = stratum(feats)
        stay = np.full((len(feats), len(LEVELS)), np.nan)
        energy = np.full((len(feats), len(LEVELS)), np.nan)
        for i in range(len(feats)):
            for cols, table in self.tables:
                key = tuple(int(keys.iloc[i][c]) for c in cols)
                if key in table:
                    stay[i], energy[i] = table[key]
                    break
        return tidy(stay, energy)


@dataclass
class DriverPast(QuantileModel):
    """The driver's own ended sessions (all of them, not only the history window), given
    the time plugged in; the population rule otherwise."""

    min_sessions: int = 3
    name: str = "driver"

    def fit(self, train, ended, origin):
        self.population = Population()
        self.population.fit(train, ended, origin)

    def predict(self, day, feats):
        stay, energy = self.population.predict(day, feats)
        users = [u if isinstance(u, str) else None for u in day.present["user_id"]]
        linked = [i for i, u in enumerate(users) if u is not None]
        if not linked or len(day.ended) == 0:
            return stay, energy
        past = day.ended[day.ended["user_id"].isin({users[i] for i in linked})]
        sessions = _Sessions.of(past)
        for i in linked:
            mine = sessions.subset(sessions.user == users[i])
            elapsed = float(feats["elapsed_h"].iloc[i])
            if int((mine.stay >= elapsed).sum()) >= self.min_sessions:
                stay[i], energy[i] = mine.conditional(elapsed)
        return tidy(stay, energy)


def arrival_hour(times: pd.Series, tz: str = LOCAL_TZ) -> np.ndarray:
    """Local clock hour of plug-in, 0 to 24."""
    local = times.dt.tz_convert(tz)
    return (local - local.dt.normalize()) / _HOUR


@dataclass
class Mixture(QuantileModel):
    """The Gaussian-mixture predictor of Lee, Li and Low (2019), as lower quantiles of
    remaining stay. `components` is chosen by held-out likelihood, as in their Section 4."""

    components: int = 5
    per_driver: bool = False
    min_driver_sessions: int = 5
    n_init: int = 3
    seed: int = 0
    name: str = "mixture"

    def fit(self, train, ended, origin):
        x = self.points(ended)
        self.gmm = GaussianMixture(
            self.components, covariance_type="full", n_init=self.n_init, random_state=self.seed
        ).fit(x)
        self.weights: dict[str, np.ndarray] = {}
        if self.per_driver:
            stay = (ended["disconnect_time"] - ended["connect_time"]) / _HOUR
            whole = np.isfinite(ended["kwh_delivered"].to_numpy(float)) & np.isfinite(stay)
            users = ended["user_id"].to_numpy(object)[whole.to_numpy()]
            for user in {u for u in users if isinstance(u, str)}:
                mine = x[users == user]
                if len(mine) >= self.min_driver_sessions:
                    self.weights[user] = self._refit_weights(mine)

    @staticmethod
    def points(ended: pd.DataFrame) -> np.ndarray:
        """(arrival hour, stay in hours, energy in kWh) of each session with all three."""
        stay = (ended["disconnect_time"] - ended["connect_time"]) / _HOUR
        x = np.column_stack(
            [
                arrival_hour(ended["connect_time"]),
                stay.to_numpy(),
                ended["kwh_delivered"].to_numpy(float),
            ]
        )
        return x[np.isfinite(x).all(axis=1)]

    def _refit_weights(self, x: np.ndarray, rounds: int = 100) -> np.ndarray:
        """EM on the component weights only, means and covariances held fixed (I-GMM)."""
        dens = np.column_stack(
            [
                _gaussian(x, self.gmm.means_[k], self.gmm.covariances_[k])
                for k in range(self.components)
            ]
        )
        w = self.gmm.weights_.copy()
        for _ in range(rounds):
            resp = dens * w
            resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)
            w = resp.mean(axis=0)
        return w

    def stay_cdf(self, arrival: float, stays: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """P(stay <= s | arrival hour) for each s, under the fitted mixture."""
        mu, cov = self.gmm.means_, self.gmm.covariances_
        var_a = cov[:, 0, 0]
        w = weights * norm.pdf(arrival, mu[:, 0], np.sqrt(var_a))
        if w.sum() <= 0:
            return np.full(len(stays), np.nan)
        w = w / w.sum()
        mean = mu[:, 1] + cov[:, 0, 1] / var_a * (arrival - mu[:, 0])
        sd = np.sqrt(np.maximum(cov[:, 1, 1] - cov[:, 0, 1] ** 2 / var_a, 1e-12))
        return (w[None, :] * norm.cdf((stays[:, None] - mean[None, :]) / sd[None, :])).sum(axis=1)

    def predict(self, day, feats):
        grid = np.linspace(0.0, HORIZON_H, int(HORIZON_H * 60) + 1)
        stay = np.full((len(feats), len(LEVELS)), np.nan)
        arrivals = arrival_hour(feats["connect_time"])
        users = [u if isinstance(u, str) else None for u in day.present["user_id"]]
        for i, (a, e) in enumerate(zip(arrivals, feats["elapsed_h"].to_numpy(float), strict=True)):
            weights = (
                self.weights.get(users[i], self.gmm.weights_) if users[i] else self.gmm.weights_
            )
            cdf = self.stay_cdf(a, np.concatenate([[e], e + grid]), weights)
            left = 1.0 - cdf[0]
            if not np.isfinite(left) or left < 1e-9:
                continue
            conditional = (cdf[1:] - cdf[0]) / left  # P(remaining <= r | stayed e)
            for k, q in enumerate(LEVELS):
                below = np.flatnonzero(conditional > q)
                stay[i, k] = grid[below[0]] if len(below) else HORIZON_H
        return tidy(stay, np.full_like(stay, np.nan))

    def held_out_loglik(self, ended: pd.DataFrame) -> float:
        return float(self.gmm.score(self.points(ended)))


def _gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = x - mean
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", d, inv, d)) / np.sqrt((2 * np.pi) ** 3 * det)


@dataclass
class TypedPlan(QuantileModel):
    """The plan typed before 10:00, taken at face value."""

    name: str = "typed"

    def fit(self, train, ended, origin):
        return None

    def predict(self, day, feats):
        has = feats["typed"].to_numpy(float) == 1
        remaining = np.clip(feats["typed_remaining_h"].to_numpy(float), 0.0, HORIZON_H)
        typed_h = feats["typed_minutes"].to_numpy(float) / 60.0
        with np.errstate(divide="ignore", invalid="ignore"):
            energy = np.where(
                typed_h > 0, feats["typed_kwh"].to_numpy(float) * remaining / typed_h, 0.0
            )
        stay = np.where(has[:, None], remaining[:, None], np.nan) * np.ones(len(LEVELS))
        energy = np.where(has[:, None], energy[:, None], np.nan) * np.ones(len(LEVELS))
        return tidy(stay, energy)
