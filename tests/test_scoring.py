import numpy as np
import pandas as pd
import pytest

from staypromise.scoring import monthly_summary, overall_summary


def decisions():
    times = pd.to_datetime(
        ["2001-01-05 12:00", "2001-01-06 12:00", "2001-01-07 12:00", "2001-02-01 03:00"], utc=True
    )
    return pd.DataFrame(
        {
            "rule": "r",
            "decision_time": times,
            "offered": [True, True, False, True],
            "promised_h": [2.0, 3.0, np.nan, 1.0],
            "kept": pd.array([True, False, pd.NA, True], dtype="boolean"),
        }
    )


def test_monthly_scores():
    jan = monthly_summary(decisions()).iloc[0]
    assert (jan["decisions"], jan["offered"], jan["kept"]) == (3, 2, 1)
    assert jan["kept_share"] == pytest.approx(0.5)  # of the promises made
    assert jan["mean_promised_h"] == pytest.approx(5.0 / 3.0)  # no promise counts as 0 h
    assert jan["offered_share"] == pytest.approx(2.0 / 3.0)


def test_months_follow_the_site_time_zone():
    # 1 Feb 03:00 UTC is still 31 Jan in California.
    by_month = monthly_summary(decisions(), tz="America/Los_Angeles")
    assert list(by_month["month"].astype(str)) == ["2001-01"]
    assert by_month.iloc[0]["decisions"] == 4


def test_overall_scores():
    row = overall_summary(decisions()).iloc[0]
    assert row["rule"] == "r"
    assert row["kept_share"] == pytest.approx(2.0 / 3.0)
    assert overall_summary(decisions().drop(columns="rule")).shape[0] == 1
