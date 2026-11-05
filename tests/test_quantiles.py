import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

from fleetoffer.baselines import DriverPast, Mixture, Population, Stratified, TypedPlan
from fleetoffer.checks import check_no_lookahead_days
from fleetoffer.engine import fleet_replay
from fleetoffer.features import NUMERIC
from fleetoffer.quantiles import (
    GBDT,
    LEVELS,
    LinearQuantiles,
    PerCarRule,
    QuantileModel,
    QuantileNet,
    at_level,
    pinball,
    training_rows,
)
from fleetoffer.synthetic import Regime, make_sessions, with_app_entries

T = pd.Timestamp("2019-03-01", tz="UTC")


def no_flags(dates):
    n = len(dates)
    return pd.DataFrame({"in_term": np.zeros(n), "summer": np.zeros(n), "holiday": np.zeros(n)})


def cardays(n, rng, stay=None, energy=None, x=None):
    """Made-up car-days with every input present; targets given or drawn."""
    frame = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    if x is not None:
        frame["plugin_hour"] = x
    frame["station"] = rng.choice(["s1", "s2", "s3"], n)
    frame["decision_time"] = T - pd.to_timedelta(rng.uniform(2, 60, n), unit="D")
    frame["connect_time"] = frame["decision_time"] - pd.Timedelta(hours=2)
    frame["revealed_at"] = frame["decision_time"] + pd.Timedelta(hours=10)
    frame["target_stay_h"] = rng.uniform(0, 10, n) if stay is None else stay
    frame["target_energy_kwh"] = rng.uniform(0, 20, n) if energy is None else energy
    frame["linked"] = 1.0
    frame["weekday"] = 1.0
    return frame


def test_a_refit_sees_only_revealed_car_days_that_began_a_day_before_it():
    rng = np.random.default_rng(0)
    rows = cardays(6, rng)
    rows["decision_time"] = T - pd.to_timedelta([40, 20, 3, 1, 1, 0.5], unit="D")
    rows["connect_time"] = rows["decision_time"] - pd.Timedelta(hours=2)
    rows["revealed_at"] = rows["decision_time"] + pd.Timedelta(hours=10)
    rows.loc[2, "revealed_at"] = T + pd.Timedelta(hours=1)  # its last car is still plugged in
    kept = training_rows(rows, T)
    # rows 3 and 4 plugged in 26 hours before the origin, row 5 in the last 24 hours
    assert kept["decision_time"].tolist() == rows["decision_time"].iloc[[0, 1, 3, 4]].tolist()
    assert len(training_rows(rows, T, window_days=30)) == 3


def test_pinball_and_reading_a_level_off_the_grid():
    assert pinball(np.array([2.0]), np.array([[1.0, 3.0]]), np.array([0.1, 0.9])).tolist() == [
        [pytest.approx(0.1), pytest.approx(0.1)]
    ]
    grid = np.tile(np.arange(len(LEVELS), dtype=float), (2, 1))
    assert at_level(grid, 0.1) == pytest.approx([5.0, 5.0])
    assert at_level(grid, 0.125) == pytest.approx([5.5, 5.5])
    assert at_level(grid, 0.001) == pytest.approx([0.0, 0.0])  # below the grid: trivial


@pytest.mark.parametrize(
    "model", [GBDT(max_iter=100), QuantileNet(epochs=150, hidden=16), LinearQuantiles()]
)
def test_learned_families_recover_quantiles_that_depend_on_an_input(model):
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.uniform(0, 1, n)
    stay = 2 + 4 * x + rng.normal(0, 0.5, n)  # quantile q: 2 + 4x + 0.5 z_q
    energy = rng.exponential(5.0, n)  # quantile q: -5 log(1 - q), whatever the inputs
    train = cardays(n, rng, stay=stay, energy=energy, x=x)
    model.fit(train, None, T)
    test = cardays(200, rng, x=np.full(200, 0.5))
    pred_stay, pred_energy = model.predict(None, test)
    assert (np.diff(pred_stay, axis=1) >= -1e-9).all()  # never crossing
    for q in (0.1, 0.5, 0.9):
        k = int(np.flatnonzero(np.isclose(LEVELS, q))[0])
        assert np.median(pred_stay[:, k]) == pytest.approx(4 + 0.5 * norm.ppf(q), abs=0.35)
        assert np.median(pred_energy[:, k]) == pytest.approx(-5 * np.log(1 - q), rel=0.15, abs=0.3)


def test_an_unseen_month_gets_the_average_of_the_months_seen_in_training():
    rng = np.random.default_rng(3)
    n = 2000
    train = cardays(n, rng)
    train["month"] = rng.choice([10.0, 11.0], n)
    train["target_stay_h"] = np.where(train["month"] == 10, 3.0, 5.0) + rng.normal(0, 0.3, n)
    model = LinearQuantiles()
    model.fit(train, None, T)
    test = cardays(50, rng)
    test["month"] = 1.0  # not seen in training
    pred, _ = model.predict(None, test)
    k = int(np.flatnonzero(np.isclose(LEVELS, 0.5))[0])
    assert np.median(pred[:, k]) == pytest.approx(4.0, abs=0.15)
    hot = model.design(test)[:, -2:]
    assert hot == pytest.approx(np.full((50, 2), 0.5))


def test_network_gradient_matches_finite_differences():
    rng = np.random.default_rng(2)
    net = QuantileNet(hidden=4)
    x, y = rng.normal(size=(7, 3)), rng.uniform(0, 1, 7)
    params = [
        rng.normal(size=(3, 4)),
        rng.normal(size=4),
        rng.normal(size=(4, len(LEVELS))),
        rng.normal(size=len(LEVELS)),
    ]
    grads = net._grads(params, x, y)

    def loss(ps):
        return pinball(y, net._forward(ps, x)[0]).mean()

    for w, g in zip(params, grads, strict=True):
        flat = w.reshape(-1)
        for i in range(0, flat.size, max(1, flat.size // 5)):
            keep = flat[i]
            flat[i] = keep + 1e-6
            up = loss(params)
            flat[i] = keep - 1e-6
            down = loss(params)
            flat[i] = keep
            assert g.reshape(-1)[i] == pytest.approx((up - down) / 2e-6, abs=1e-5)


def _ended(stays, kwh=None, users=None):
    n = len(stays)
    start = T - pd.Timedelta(days=30)
    connect = start + pd.to_timedelta(np.arange(n), unit="h")
    disconnect = connect + pd.to_timedelta(np.asarray(stays, dtype=float), unit="h")
    return pd.DataFrame(
        {
            "connect_time": connect,
            "disconnect_time": disconnect,
            "done_charging_time": disconnect,
            "kwh_delivered": np.ones(n) if kwh is None else kwh,
            "user_id": [None] * n if users is None else users,
        }
    )


def test_population_rule_conditions_on_time_plugged_in():
    model = Population()
    model.fit(None, _ended(np.arange(1, 101, dtype=float)), T)  # stays of 1, 2, ..., 100 h
    feats = pd.DataFrame({"elapsed_h": [0.0, 50.0]})
    stay, energy = model.predict(None, feats)
    # P(stay < 1 + X) <= 0.1 for X < 10, so the 0.1 quantile is 10 more hours (then capped)
    assert stay[0, np.isclose(LEVELS, 0.1)][0] == pytest.approx(10.0)
    # of the 51 stays from 50 to 100 h, the 0.1 quantile of the remaining stay is 5 h
    assert stay[1, np.isclose(LEVELS, 0.1)][0] == pytest.approx(5.0)
    # each session charged 1 kWh evenly over its stay: energy after 50 h of the 100 h stay
    assert energy[1, 0] == pytest.approx(0.0) and energy[0, -1] > 0


def test_stratified_table_merges_small_cells():
    rng = np.random.default_rng(3)
    train = cardays(100, rng, stay=np.r_[np.full(60, 2.0), np.full(40, 8.0)])
    train["plugin_hour"] = np.r_[np.full(60, 8.5), np.full(40, 7.5)]
    train["weekday"] = np.r_[np.full(55, 1.0), np.full(5, 4.0), np.full(40, 1.0)]
    model = Stratified(min_cell=30)
    model.fit(train, None, T)
    feats = train.iloc[[0, 56, 60]].copy()  # an 08:00 Tuesday, an 08:00 Friday, a 07:00 car
    stay, _ = model.predict(None, feats)
    assert stay[0, 0] == stay[1, 0] == 2.0  # the Friday cell has 5 car-days: hour cell used
    assert stay[2, 0] == 8.0


def test_driver_past_uses_the_drivers_own_sessions_when_there_are_enough():
    ended = _ended([9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0], users=["u"] * 3 + [None] * 4)
    model = DriverPast(min_sessions=3)
    model.fit(None, ended, T)

    class Day:
        present = pd.DataFrame({"user_id": ["u", None]})

    Day.ended = ended
    stay, _ = model.predict(Day, pd.DataFrame({"elapsed_h": [2.0, 0.5]}))
    assert stay[0, 0] == pytest.approx(7.0)  # own stays of 9 h, 2 h in
    assert stay[1, 0] == pytest.approx(0.5)  # everyone's: mostly 1 h stays


def test_mixture_with_one_component_is_the_conditional_normal():
    rng = np.random.default_rng(4)
    cov = np.array([[1.0, 0.8, 0.1], [0.8, 4.0, 0.2], [0.1, 0.2, 9.0]])
    x = rng.multivariate_normal([8.0, 6.0, 10.0], cov, 20000)
    model = Mixture(components=1, n_init=1)
    model.gmm = GaussianMixture(1).fit(x)
    model.weights = {}
    mu, c = model.gmm.means_[0], model.gmm.covariances_[0]
    mean = mu[1] + c[0, 1] / c[0, 0] * (9.0 - mu[0])
    sd = np.sqrt(c[1, 1] - c[0, 1] ** 2 / c[0, 0])
    got = model.stay_cdf(9.0, np.array([5.0, 7.0]), model.gmm.weights_)
    assert got == pytest.approx(norm.cdf([5.0, 7.0], mean, sd))

    class Day:
        present = pd.DataFrame({"user_id": [None]})

    connect = pd.Series([pd.Timestamp("2019-02-04 09:00", tz="America/Los_Angeles")])
    feats = pd.DataFrame({"connect_time": connect.dt.tz_convert("UTC"), "elapsed_h": [1.0]})
    stay, energy = model.predict(Day, feats)
    # remaining stay given it is at least 1 h: lower quantiles of the normal truncated at 1 h
    tail = 1 - norm.cdf(1.0, mean, sd)
    for k in (5, 10):
        want = norm.ppf(norm.cdf(1.0, mean, sd) + LEVELS[k] * tail, mean, sd) - 1.0
        assert stay[0, k] == pytest.approx(want, abs=1 / 60 + 1e-9)
    assert np.isnan(energy).all()


def test_typed_plan_at_face_value():
    feats = pd.DataFrame(
        {
            "typed": [1.0, 0.0],
            "typed_remaining_h": [4.0, np.nan],
            "typed_minutes": [480.0, np.nan],
            "typed_kwh": [16.0, np.nan],
        }
    )
    stay, energy = TypedPlan().predict(None, feats)
    assert np.allclose(stay[0], 4.0) and np.allclose(energy[0], 8.0)
    assert np.isnan(stay[1]).all()


@pytest.fixture(scope="module")
def claimed():
    sessions = make_sessions(
        [Regime("2001-01-01", sessions_per_weekday=12, median_stay_h=6.0, drivers=40)],
        "2001-03-20",
        seed=5,
    )
    return with_app_entries(sessions, seed=5)


class Spy(QuantileModel):
    """Records what each refit was given."""

    name = "spy"

    def __init__(self):
        self.fits = []

    def fit(self, train, ended, origin):
        self.fits.append((origin, train.copy(), ended.copy()))

    def predict(self, day, feats):
        n = len(feats)
        return np.ones((n, len(LEVELS))), np.ones((n, len(LEVELS)))


def test_refits_happen_monthly_on_what_was_known_at_the_origin(claimed):
    sessions, entries = claimed
    spy = Spy()
    rule = PerCarRule([spy], min_rows=1, calendar=no_flags)
    fleet_replay(sessions, rule, method="bounds", entries=entries)
    origins = [f[0] for f in spy.fits]
    assert [o.tz_convert("America/Los_Angeles").month for o in origins] == [2, 3]
    for origin, train, ended in spy.fits:
        assert (train["revealed_at"] < origin).all()
        assert (train["connect_time"] <= origin - pd.Timedelta(hours=24)).all()
        assert (ended["disconnect_time"] < origin).all()
        assert len(train) > 100


def test_per_car_models_do_not_look_ahead(claimed):
    """Every model's prediction at every level, not only the bounds the engine is given."""
    sessions, entries = claimed
    rules = []

    def factory(table):
        models = [
            GBDT(max_iter=20),
            QuantileNet(epochs=5, hidden=4),
            LinearQuantiles(),
            Population(),
            Stratified(),
            DriverPast(),
            Mixture(components=2, n_init=1),
            Mixture(components=2, n_init=1, per_driver=True, name="mixture-driver"),
            TypedPlan(),
        ]
        rules.append(PerCarRule(models, min_rows=50, calendar=no_flags))
        return rules[-1]

    late = sessions["connect_time"].min() + pd.Timedelta(days=45)
    cutoffs = [late + pd.Timedelta(hours=4), late + pd.Timedelta(days=12, hours=-1)]
    compared = check_no_lookahead_days(factory, sessions, cutoffs, entries=entries, method="bounds")
    assert compared > 40
    for k, cutoff in enumerate(cutoffs):
        real, scrambled = (r.predictions() for r in rules[2 * k : 2 * k + 2])
        real = real[real["decision_time"] < cutoff].reset_index(drop=True)
        scrambled = scrambled[scrambled["decision_time"] < cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(real, scrambled)
        filled = [c for c in real.columns if ":" in c and real[c].notna().any()]
        assert len(filled) == len(real.columns) - 6 - 32  # all but the two mixtures' energy


def test_the_offer_from_learned_bounds_does_not_look_ahead(claimed):
    sessions, entries = claimed
    first = sessions["connect_time"].min()
    short = sessions[sessions["connect_time"] < first + pd.Timedelta(days=40)]

    def factory(table):
        return PerCarRule([GBDT(max_iter=10), Population()], min_rows=50, calendar=no_flags)

    cutoff = first + pd.Timedelta(days=37, hours=19)
    assert check_no_lookahead_days(factory, short, [cutoff], entries=entries) > 20
