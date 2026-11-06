import pandas as pd
import pytest

from fleetoffer import options
from fleetoffer.baselines import Population
from fleetoffer.daylevel import DeliveryCalibrated
from fleetoffer.engine import fleet_replay, training_base_set
from fleetoffer.quantiles import PerCarRule
from fleetoffer.synthetic import Regime, make_sessions


def no_flags(dates):
    n = len(dates)
    return pd.DataFrame({"in_term": [0] * n, "summer": [0] * n, "holiday": [0] * n})


@pytest.fixture(scope="module")
def sessions():
    return make_sessions(
        [Regime("2001-01-01", sessions_per_weekday=12, median_stay_h=6.0)], "2001-03-20", seed=7
    )


@pytest.mark.parametrize("form", options.FORMS)
def test_the_rule_scores_the_offer_the_engine_makes(sessions, form):
    base = training_base_set(sessions, pd.Timestamp("2001-01-20", tz="UTC"))
    per_car = PerCarRule([Population()], min_rows=50, calendar=no_flags)
    rule = DeliveryCalibrated(per_car, options.make_offer(form, base), gamma=0.05)
    days = fleet_replay(sessions, rule, **options.engine_options(form, base)).days
    log = pd.DataFrame(rule.log)
    log = log[log["fitted"]]
    merged = log.merge(days[["decision_time", "fixed_delivered"]], on="decision_time")
    assert len(merged) > 20
    # a day whose offer was not delivered at the level used has a score below that level
    assert merged.loc[~merged["fixed_delivered"].astype(bool), "failed"].astype(bool).all()


def test_every_planned_run_names_a_known_arm_and_form():
    for name, form in options.RUNS:
        assert form in options.FORMS
        assert name in {"1", "4", "5"} or (name.endswith("d") and name[0] in options.MODELS)
