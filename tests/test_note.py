import re

import numpy as np
import pandas as pd
import pytest

from fleetoffer import figures, note


def test_a_placeholder_without_a_number_is_an_error():
    assert note.fill("size ${size} kWh", {"size": "1.5"}) == "size 1.5 kWh"
    with pytest.raises(KeyError):
        note.fill("size ${size} kWh, delivered ${rate}", {"size": "1.5"})


def test_the_template_names_only_plain_placeholders():
    text = note.TEMPLATE.read_text()
    names = re.findall(r"\$\{([^}]*)\}", text)
    assert names and all(re.fullmatch(r"[a-z_0-9]+", n) for n in names)
    assert "$" not in re.sub(r"\$\{[a-z_0-9]+\}", "", text)


def test_the_zero_ceiling_figure_reads_both_runs_of_each_model(tmp_path):
    levels = [0.0, 0.01, 0.1, 0.5]
    rows = []
    for model in ("4", "4k", "5"):
        for form in ("average", "segment"):
            share = np.linspace(1.0, 0.0, len(levels))
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "form": form,
                        "level": levels,
                        "zero_ceiling_share_with_energy": share,
                        "zero_stay_share": share / 2,
                    }
                )
            )
    path = figures.plot_zero_ceilings(
        pd.concat(rows),
        tmp_path / "zero.png",
        models={"4": ("learned", 0), "5": ("population", 1)},
        reached={"learned rule": 0.003},
        title="test",
    )
    assert path.exists() and path.stat().st_size > 0


def test_the_accounting_gives_the_registered_and_the_calibrated_rule_a_row_each():
    frame = pd.DataFrame(
        {
            "form": "segment",
            "step": "day-level rule as run",
            "who": ["4", "5", "4d", "5d"],
            "size_mean_kwh": [0.0, 0.0, 1.5, 2.5],
            "share_of_perfect": [0.0, 0.0, 0.1, 0.2],
            "delivered": [1.0, 0.9, 0.95, 0.9],
        }
    )
    rows = note.accounting_html(frame, "segment").splitlines()
    assert len(rows) == 2
    assert any("calibrated on delivery" in r and "1.50" in r for r in rows)
    assert any("registered rule" in r and "0.00" in r for r in rows)


def test_a_level_between_grid_levels_reads_the_upper_one():
    assert note.grid_level_at_or_above(0.003) == 0.005
    assert note.grid_level_at_or_above(0.01) == 0.01
    assert note.grid_level_at_or_above(0.028) == 0.05


def _sweep(shares: dict[str, list[float]], levels=(0.0, 0.002, 0.005, 0.01, 0.05)):
    return pd.concat(
        pd.DataFrame(
            {
                "model": model,
                "form": "average",
                "level": list(levels),
                "zero_ceiling_share_with_energy": share,
            }
        )
        for model, share in shares.items()
    )


def test_zero_ceiling_readings_at_each_rules_level():
    table = _sweep(
        {
            "4": [1.0, 1.0, 0.9, 0.5, 0.1],
            "5": [1.0, 0.8, 0.8, 0.7, 0.7],
            "6": [1.0, 1.0, 1.0, 0.6, 0.2],
            "4k": [1.0, 0.02, 0.01, 0.0, 0.0],
        }
    )
    reached = {"learned rule": 0.003, "population rule": 0.0, "stratified rule": 0.009}
    r = note.zero_ceilings(table, reached)
    assert r["at_4"] == 0.9  # read at 0.005, the grid level above 0.003
    assert r["at_5"] == 1.0  # below the grid every bound is zero
    assert r["at_6"] == 0.6
    assert r["falls"] and r["known_max"] == 0.02
    rising = table.copy()
    rising.loc[
        (rising["model"] == "5") & (rising["level"] == 0.05), "zero_ceiling_share_with_energy"
    ] = 0.75
    assert not note.zero_ceilings(rising, reached)["falls"]


def test_the_accounting_table_has_a_dash_where_a_period_has_no_step():
    rows = []
    for period, has_sweep in (
        ("development", True),
        ("evaluation", True),
        ("sealed", True),
        ("jpl", False),
    ):
        rows += [
            {
                "period": period,
                "form": "average",
                "step": "exact band of the true sets",
                "who": "1",
                "size_mean_kwh": 100.0,
                "share_of_perfect": 2.5,
                "delivered": np.nan,
            },
            {
                "period": period,
                "form": "average",
                "step": "perfect knowledge",
                "who": "1",
                "size_mean_kwh": 40.0,
                "share_of_perfect": 1.0,
                "delivered": 1.0,
            },
        ]
        if has_sweep:
            rows += [
                {
                    "period": period,
                    "form": "average",
                    "step": "forecasts at 90% delivery",
                    "who": m,
                    "size_mean_kwh": 0.0,
                    "share_of_perfect": 0.0,
                    "delivered": 1.0,
                }
                for m in ("4", "5", "6")
            ]
    html = note.accounting_table(pd.DataFrame(rows))
    forecasts = next(line for line in html.splitlines() if "Forecasts" in line)
    assert forecasts.count("<td>0</td>") == 9 and forecasts.count("<td>–</td>") == 3
    assert "40.0 (40%)" in html


def test_offer_days_counts_scored_days_above_the_zero_line():
    days = pd.DataFrame(
        {
            "arm": ["4", "4", "5", "5", "5"],
            "size_kwh": [0.0, 4e-14, 0.0, 0.003355, 2.0],
            "cars_with_energy": [6, 6, 6, 6, 3],  # the last day is not scored (fewer than 5)
            "decision_time": pd.Timestamp("2019-09-23 17:00", tz="UTC"),
        }
    )
    got = note.offer_days(days, {"4": "a", "5": "b"}, "k")
    assert got == {"k_pos_a": "0", "k_max_a": "0", "k_pos_b": "1", "k_max_b": "0.0034"}
