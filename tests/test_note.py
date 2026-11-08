import json
import re

import numpy as np
import pandas as pd
import pytest

from fleetoffer import figures, note


def test_a_placeholder_without_a_number_is_an_error():
    assert note.fill("size {{size}} kWh", {"size": "1.5"}) == "size 1.5 kWh"
    with pytest.raises(KeyError):
        note.fill("size {{size}} kWh, delivered {{rate}}", {"size": "1.5"})


def test_the_template_names_only_plain_placeholders():
    text = note.TEMPLATE.read_text()
    names = note.FIELD.findall(text)
    assert names and all(re.fullmatch(r"[a-z_0-9]+", n) for n in names)
    rest = note.FIELD.sub("", text)
    assert "{{" not in rest and "}}" not in rest


def test_the_page_says_where_it_comes_from_and_drops_the_template_comment():
    out = note.page("<!-- edit me -->\n\n# {{title}}\n", {"title": "T"})
    assert out.startswith("<!-- Written by `uv run fleetoffer report`")
    assert "edit me" not in out and out.endswith("# T\n")


def test_every_figure_the_template_shows_is_copied():
    shown = set(re.findall(r"\]\(([\w-]+\.png)\)", note.TEMPLATE.read_text()))
    assert shown == set(note.FIGURES)


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
    assert figures.bare(path).exists()  # the version without heading or note, for the report


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
        ("sealed, 2021 only", True),
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
    table = note.accounting_table(pd.DataFrame(rows))
    forecasts = next(line for line in table.splitlines() if "Forecasts" in line)
    assert forecasts.count("| 0 / 0 / 0 ") == 4 and forecasts.endswith("| – |")
    assert "40.0 (40%)" in table


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


def comparison(a, b, reading, low=-1.0, high=-0.5):
    return {
        "mean_a_kwh": a,
        "mean_b_kwh": b,
        "delivered_a": 0.9,
        "delivered_b": 0.95,
        "reading": reading,
        "absolute_kwh": {"estimate": a - b, "low": low, "high": high},
        "relative": {"estimate": -1.0, "low": -1.0, "high": -1.0},
    }


def test_the_hypotheses_table_reads_in_words():
    hyp = {
        "evaluation": {
            "h1_registered": {"one_week": comparison(4.9e-14, 2.9e-5, "fail", -9e-5, 1e-13)},
            "h1_secondary": {"one_week": comparison(0.18, 1.76, "fail", -1.78, -1.36)},
            "h2": {"delivered_3": 0.98, "size_3_kwh": 0.0, "size_4_kwh": 0.0, "reading": "pass"},
            "h4": {
                "learned_hours": 2.93,
                "population_hours": 2.01,
                "learned_coverage": 0.907,
                "population_coverage": 0.907,
                "car_days": 2055,
                "reading": "pass",
            },
        },
        "sealed": {
            "h1_registered": {
                "one_week": {
                    "mean_a_kwh": 0.0,
                    "mean_b_kwh": 0.0,
                    "delivered_a": 1.0,
                    "delivered_b": 1.0,
                    "reading": "not testable: both arms offer zero",
                }
            },
            "reopening": "2021-08-02",
            "h3": {
                "4 against 4f": {
                    "reopening": "2021-08-02",
                    "frozen_first20_delivered": None,
                    "adaptive_reached_at_scored_weekday": None,
                    "adaptive_scored_weekdays": 15,
                }
            },
        },
    }
    table = note.hypotheses_table(hyp)
    # no internal arm labels, relative changes of a zero offer, or negative zeros
    assert "arm" not in table and "relative" not in table
    assert "-0.00" not in table and "\N{MINUS SIGN}0.00" not in table
    # an offer too small to show at two decimals gets no difference; a visible one does
    assert "fail: 0.00 against 0.00&#160;kWh, delivered on 90% and 95% of days</td>" in table
    m = "\N{MINUS SIGN}"
    assert f"difference {m}1.58&#160;kWh ({m}1.78 to {m}1.36)" in table
    assert "not testable: both offers are zero (delivered on 100% and 100% of days)" in table
    assert "2 August 2021" in table and "15 scored weekdays" in table
    assert "JPL" not in table  # no row for a run that is absent


def test_the_sealed_sentence_says_when_the_months_cannot_test_the_proposition():
    at_zero = {f"level_{m}": 0.0 for m in ("4", "5", "6")} | {
        f"at_{m}": 1.0 for m in ("4", "5", "6")
    }
    assert "median level of every rule was 0" in note.sealed_zero_ceiling_sentence(at_zero)
    tested = {"level_4": 0.003, "level_5": 0.03, "level_6": 0.01}
    tested |= {"at_4": 0.88, "at_5": 0.69, "at_6": 0.99}
    assert note.sealed_zero_ceiling_sentence(tested).endswith("88%, 99% and 69%.")


def test_the_order_sentence_names_where_the_models_change_places():
    rows = []
    for period, fc, hind in (
        ("development", (0.0, 0.036, 0.0), (0.56, 0.21, 0.33)),
        ("evaluation", (0.005, 0.036, 0.012), (0.74, 0.32, 0.26)),
        ("sealed", (0.003, 0.027, 0.014), (0.20, 0.56, 0.39)),
    ):
        for step, shares in (("forecasts at 90% delivery", fc), ("hindsight split at 90%", hind)):
            for who, share in zip("456", shares, strict=True):
                rows.append(
                    {
                        "period": period,
                        "form": "average",
                        "step": step,
                        "who": who,
                        "share_of_perfect": share,
                    }
                )
    text = note.model_order_sentence(pd.DataFrame(rows))
    assert "56% and 21%; 74% and 32%, in the order" in text
    assert text.endswith("the order differs in the sealed months.")
    assert note.model_order_sentence(pd.DataFrame(rows[:0], columns=list(rows[0]))) == ""


def test_the_baseline_sentence_says_when_the_trees_only_tie_a_baseline():
    def loss(stay: dict, energy: dict) -> pd.Series:
        rows = {("all", m, "stay"): v for m, v in stay.items()}
        rows |= {("all", m, "energy"): v for m, v in energy.items()}
        return pd.Series(rows)

    clear = loss(
        {"gbdt": 0.4, "driver": 0.5, "stratified": 0.5},
        {"gbdt": 0.6, "driver": 0.7, "stratified": 1.0},
    )
    assert note.baseline_sentence(clear).startswith("The trees beat")
    tied = loss(
        {"gbdt": 0.4, "driver": 0.5, "stratified": 0.5},
        {"gbdt": 0.683, "driver": 0.684, "stratified": 1.0},
    )
    text = note.baseline_sentence(tied)
    level = "on energy the trees are level with each driver's own past (0.683 against 0.684 kWh)"
    assert level in text
    assert "the plan fixes no margin" in text
    assert "on stay" not in text and "stratified table (0" not in text
    anonymous = pd.concat(
        [
            tied,
            pd.Series(
                {
                    ("anonymous", "gbdt", "energy"): 0.49,
                    ("anonymous", "stratified", "energy"): 0.484,
                }
            ),
        ]
    )
    text = note.baseline_sentence(anonymous)
    assert "for anonymous drivers the stratified table is lower than the trees on energy" in text
    assert "(0.484 against 0.490 kWh)" in text


def test_late_outcomes_count_days_known_before_the_next_decision(tmp_path):
    t = pd.date_range("2019-09-02 17:00", periods=4, freq="D", tz="UTC")
    days = pd.DataFrame({"decision_time": t, "arm": "4", "size_kwh": 1.0, "cars_with_energy": 9})
    revealed = [
        t[0] + pd.Timedelta(hours=10),
        t[3] + pd.Timedelta(hours=1),
        t[2] + pd.Timedelta(hours=10),
        t[3] + pd.Timedelta(hours=10),
    ]
    logs = pd.DataFrame(
        {
            "decision_time": t,
            "arm": "4",
            "revealed_at": revealed,
            "scored": True,
            "miss_at_decision": 0.10,
        }
    )
    days.to_csv(tmp_path / "days.csv.gz", index=False)
    logs.to_csv(tmp_path / "logs.csv.gz", index=False)
    late = note.late_outcomes(tmp_path, None)
    # day 2 is revealed after days 3 and 4 were decided: known on 2 of the 3 days with a next
    assert late["known_min"] == pytest.approx(2 / 3)
    assert late["pending"] == 1  # at day 4's decision only day 2 is still open
    assert late["bound_min"] == pytest.approx(0.10 + (0.10 + 0.9 * 0.03 * 2) / (0.03 * 4))


def test_monthly_lines_break_at_a_gap():
    x = pd.Series(
        pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15", "2021-08-15", "2021-09-15"])
    )
    assert figures._runs(x) == [slice(0, 3), slice(3, 5)]


def test_a_job_left_out_of_a_run_is_named_and_nothing_is_said_otherwise(tmp_path):
    assert note.left_out_sentence(tmp_path) == ""
    manifest = {"jobs": [["jpl", "jpl", "4d", "average", "updating", True]] * 36}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert note.left_out_sentence(tmp_path) == ""
    manifest |= {"jobs_left_out": ["jpl-jpl-6d-average-updating-True"], "registered_jobs": 37}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    text = note.left_out_sentence(tmp_path)
    assert text.startswith(" One of its 37 runs, the stratified model calibrated on delivery,")
    assert "updating on JPL days, did not finish and is left out" in text
