import numpy as np
import pandas as pd
import pytest

from fleetoffer import percar
from fleetoffer.data import SealedPeriodError
from fleetoffer.quantiles import LEVELS


def frame(stay_bounds, true_deadline, users, energy=None, true_energy=None):
    """A scored frame for one model 'm' with the same bound at every level."""
    n = len(stay_bounds)
    out = pd.DataFrame(
        {
            "decision_time": pd.Timestamp("2019-06-03 17:00", tz="UTC"),
            "session_id": [f"s{k}" for k in range(n)],
            "linked": [1.0 if u else 0.0 for u in users],
            "user_id": users,
            "true_deadline": true_deadline,
            "true_energy_kwh": np.zeros(n) if true_energy is None else true_energy,
        }
    )
    out["true_stay_h"] = out["true_deadline"] * 0.25
    for q in LEVELS:
        out[f"m:stay:{q:g}"] = stay_bounds
        out[f"m:energy:{q:g}"] = np.zeros(n) if energy is None else energy
    out["month"] = pd.Period("2019-06", "M")
    out["period"] = percar.PERIODS[2][0]
    out["driver"] = np.where(out["linked"] == 1, "known", "anonymous")
    return out


def test_a_stay_bound_misses_only_if_it_rounds_above_the_true_deadline():
    f = frame([2.2, 2.3, 4.0], [8, 9, 15], ["u", None, "v"])
    ok, missed = percar.misses(f, "m", "stay")
    # 2.2 h rounds down to 8 steps (held); 2.3 h to 9 (held); 4.0 h is 16 steps > 15 (missed)
    assert ok.all() and missed[:, 0].tolist() == [False, False, True]


def test_figure_one_counts_hours_kept_on_the_engines_grid():
    f = frame([2.2, 2.3, 4.0], [8, 9, 15], ["u", None, "v"])
    monthly = percar.figure_one(f, {"m": "model"})
    row = monthly.iloc[0]
    assert row["given"] == 3 and row["held_share"] == pytest.approx(2 / 3)
    assert row["mean_bound_h"] == pytest.approx((2.0 + 2.25 + 4.0) / 3)


def test_smape_averages_within_each_driver_first():
    f = frame([1.0, 1.0, 3.0], [4, 12, 4], ["u", "u", "v"])  # truths 1 h, 3 h, 1 h
    table = percar.smape(f, ["m"]).set_index("target")
    # driver u: 0 and |3 - 1| / 4 = 0.5, mean 0.25; driver v: |1 - 3| / 4 = 0.5
    assert table.loc["stay", "smape_percent"] == pytest.approx(100 * (0.25 + 0.5) / 2)
    assert table.loc["stay", "drivers"] == 2


def test_the_d10_reading_uses_the_tighter_tolerance_at_0005():
    rows = [
        {
            "period": percar.PERIODS[2][0],
            "drivers": "all",
            "model": "gbdt",
            "target": "stay",
            "level": q,
            "car_days": 100,
            "miss_rate": q + 0.008,
        }
        for q in LEVELS
    ]
    reading = percar.d10(pd.DataFrame(rows)).set_index("level")
    assert not reading.loc[0.005, "within"] and reading.loc[0.01, "within"]


def test_months_after_august_2019_need_the_tag(tmp_path):
    with pytest.raises(SealedPeriodError):
        percar.load(tmp_path, until=pd.Timestamp("2019-10-01", tz="America/Los_Angeles"))


def test_h4_reads_each_models_bound_at_the_level_its_own_past_car_days_give():
    day1 = frame([2.0] * 10, [8] * 9 + [7], ["u"] * 10)  # one bound of ten fails (8 > 7 steps)
    day2 = frame([3.0] * 4, [20] * 4, ["u"] * 4)
    day2["decision_time"] = day1["decision_time"].iloc[0] + pd.Timedelta(days=1)
    both = pd.concat([day1, day2], ignore_index=True)
    revealed = pd.DataFrame(
        {
            "decision_time": both["decision_time"].unique(),
            "revealed_at": both["decision_time"].unique() + pd.Timedelta(hours=10),
        }
    )
    rows = percar.matched(both, revealed, {"m": "model"}).set_index("session_id")
    first = rows[rows["decision_time"] == day1["decision_time"].iloc[0]]
    assert (first["level"] == 0).all() and (first["bound_h"] == 0).all()  # nothing known yet
    second = rows[rows["decision_time"] == day2["decision_time"].iloc[0]]
    # ten past scores, k = floor(0.1 * 11) = 1: the smallest, 0 (the failed car): trivial
    assert (second["level"] == 0).all() and (second["past_car_days"] == 10).all()

    day1["true_deadline"] = 8  # now every bound of day 1 held: k = 1 gives the top level
    both = pd.concat([day1, day2], ignore_index=True)
    rows = percar.matched(both, revealed, {"m": "model"})
    second = rows[rows["decision_time"] == day2["decision_time"].iloc[0]]
    assert (second["level"] == 0.95).all() and (second["bound_h"] == 3.0).all()
    assert second["held"].all()
