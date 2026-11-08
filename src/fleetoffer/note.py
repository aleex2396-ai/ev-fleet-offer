"""The note on the results (`note/note.html`): every result in it is read here from the result
tables, the template is filled, and the page is printed to PDF with a headless Chrome if one
is installed (otherwise the filled page is written alone and can be printed from any browser).

Each function below says which table its numbers come from. The settings, and the numbers of
the checks made before the tag that `docs/` already states, are typed in the template.
"""

from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from fleetoffer import daylevel, evaluation, figures, levels
from fleetoffer.devchecks import SELECTION_START
from fleetoffer.quantiles import LEVELS

TEMPLATE = Path(__file__).parents[2] / "note" / "note.html"
CHROME = "FLEETOFFER_CHROME"  # path of a Chrome or Chromium binary, if not the default below
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MODELS = {"4": ("learned", 0), "5": ("population", 1), "6": ("stratified", 2)}
TITLE = "Fleet Charging Offers from Forecast Per-Car Sets: Where the Flexibility Is Lost"
AUTHOR = "Weixuan (Alex) Lu"


def pct(x: float, digits: int = 1) -> str:
    """A share in percent; exactly zero is "0%", not "0.0%"."""
    return "0%" if x == 0 else f"{100 * x:.{digits}f}%"


def development(levels_dir: Path, checks_dir: Path) -> dict[str, str]:
    """Numbers from the model-selection months (May to August 2019): the level sweep
    (`fleetoffer levels`) and the checks (`fleetoffer checks`)."""
    checks = json.loads((Path(checks_dir) / "checks.json").read_text())
    sweep = pd.read_csv(Path(levels_dir) / "sweep.csv.gz", dtype={"model": str})
    sweep["decision_time"] = pd.to_datetime(sweep["decision_time"], utc=True)
    days = levels.scored(sweep[(sweep["model"] == "5") & (sweep["level"] == 0)])
    days = days.drop_duplicates("decision_time")
    median_cars = float(days["cars"].median())
    return {
        "empty_share": pct(checks["D4"]["zero_share"]),
        "median_cars": f"{median_cars:g}",
        "union_level": pct(0.10 / (2 * median_cars), 2),
        "selection_days": str(len(days)),
    }


def zero_ceiling_figure(levels_dir: Path, out: Path, reached: dict[str, float]) -> Path:
    table = pd.read_csv(Path(levels_dir) / "table.csv", dtype={"model": str})
    return figures.plot_zero_ceilings(
        table,
        out,
        models=MODELS,
        reached=reached,
        title=f"Zero ceilings by shared level, scored weekdays from {SELECTION_START:%B %Y}"
        " to August 2019",
    )


def fill(template: str, numbers: dict[str, str]) -> str:
    """The template with every placeholder replaced; a placeholder without a number is an
    error, so no unfilled field reaches the page."""
    return string.Template(template).substitute(numbers)


def print_pdf(html: Path, pdf: Path) -> Path | None:
    """Print `html` to `pdf` with a headless Chrome; None if no Chrome is found."""
    chrome = os.environ.get(CHROME) or DEFAULT_CHROME
    if not Path(chrome).exists() and shutil.which(chrome) is None:
        return None
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}",
            html.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    return pdf


def rule_levels(checks_dir: Path) -> dict[str, float]:
    """The median level each registered day-level rule used on the model-selection months
    (`fleetoffer checks`, one log per arm), to mark on the zero-ceiling figure."""
    names = {
        "3": "union bound",
        "4": "learned rule",
        "5": "population rule",
        "6": "stratified rule",
    }
    out = {}
    for arm, name in names.items():
        log = pd.read_csv(Path(checks_dir) / "fleet" / f"log_arm{arm}.csv")
        log["decision_time"] = pd.to_datetime(log["decision_time"], utc=True)
        keep = log["decision_time"] >= SELECTION_START
        if "fitted" in log:
            keep &= log["fitted"].astype(str) == "True"
        out[name] = float(log.loc[keep, "level"].median())
    return out


def build(results: Path, out: Path, *, rehearsal: bool = False) -> dict[str, Path | None]:
    """Fill the note from `results` (the folders the other commands write) into `out`: the
    zero-ceiling figure, the filled page and, if a Chrome is found, its PDF."""
    results, out = Path(results), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    numbers = development(results / "levels", results / "checks")
    reached = rule_levels(results / "checks")
    numbers |= {f"level_{k.split()[0]}": pct(v, 2) for k, v in reached.items()}
    zero_ceiling_figure(results / "levels", out / "zero-ceilings.png", reached)
    report = results / ("report-rehearsal" if rehearsal else "report")
    for figure in sorted(report.glob("figure*.png")):
        shutil.copy(figure, out / figure.name)
    numbers |= later(results, report, reached, numbers, rehearsal=rehearsal)
    numbers |= {"title": TITLE, "author": AUTHOR}
    page = out / "note.html"
    page.write_text(fill(TEMPLATE.read_text(), numbers))
    return {"html": page, "pdf": print_pdf(page, out / "note.pdf")}


ACCOUNTING_ROWS = {
    "exact band of the true sets": "Exact band of the true sets",
    "perfect knowledge": "Perfect knowledge",
    "forecasts at 90% delivery": "Forecasts, delivered on 90% of days",
    "empty cars known": "The same, empty cars known",
    "hindsight split at 90%": "The same, split chosen with hindsight",
    "day-level rule as run": "Level chosen from past days",
}
ACCOUNTING_WHO = {"4": "4", "5": "5", "6": "6", "4k": "4", "5k": "5", "6k": "6"}
ACCOUNTING_WHO |= {"4d": "4", "5d": "5", "6d": "6"}


def accounting_html(frame: pd.DataFrame, form: str) -> str:
    """Table rows of the accounting (`levels.accounting`) for one offer form: each step's mean
    offer size in kWh, its share of the perfect-knowledge offer, and the delivery rate it was
    read at, one column per model."""

    def cell(row) -> str:
        size = f"{row['size_mean_kwh']:.2f}"
        share = row["share_of_perfect"]
        rate = row["delivered"]
        extra = [f"{100 * share:.0f}%"] if pd.notna(share) else []
        extra += [f"d {100 * rate:.0f}%"] if pd.notna(rate) else []
        return size + (f" ({', '.join(extra)})" if extra else "")

    rows = frame[frame["form"] == form]
    out = []
    for step, label in ACCOUNTING_ROWS.items():
        mine = rows[rows["step"] == step]
        if mine.empty:
            continue
        if step in ("exact band of the true sets", "perfect knowledge"):
            first = mine.iloc[0]
            text = cell(first) if step == "perfect knowledge" else f"{first['size_mean_kwh']:.2f}"
            out.append(f'<tr><td>{label}</td><td colspan="3">{text}</td></tr>')
            continue
        # the registered rule (4, 5, 6) and the rule calibrated on delivery (4d, 5d, 6d) can
        # both appear for one form: one row each, so neither overwrites the other
        groups: dict[str, dict[str, str]] = {}
        for _, r in mine.iterrows():
            who = str(r["who"])
            kind = " (calibrated on delivery)" if who.endswith("d") else ""
            groups.setdefault(kind, {})[ACCOUNTING_WHO.get(who, who)] = cell(r)
        if len(groups) > 1 and "" in groups:
            groups[" (registered rule)"] = groups.pop("")
        for kind, cells in sorted(groups.items()):
            out.append(
                f"<tr><td>{label}{kind}</td>"
                + "".join(f"<td>{cells.get(m, '–')}</td>" for m in MODELS)
                + "</tr>"
            )
    return "\n".join(out)


# Table I: the accounting across periods, on the day-average base set.
PERIODS = {
    "development": "May–Aug 2019",
    "evaluation": "Sep 2019–Feb 2020",
    "sealed": "Sealed months",
    "sealed, 2021 only": "2021 only",
    "jpl": "JPL",
}
SHARE_ROWS = {
    "forecasts at 90% delivery": "Forecasts, 90% of days",
    "empty cars known": "Empty cars known, 90% of days",
    "hindsight split at 90%": "Hindsight split, 90% of days",
    "day-level rule as run": "Level from past days",
}


def accounts(results: Path, report: Path) -> pd.DataFrame:
    """The accounting of every period in one table (`period` column): the model-selection
    months from `fleetoffer accounting`, the later periods from `fleetoffer report`."""
    dev = pd.read_csv(Path(results) / "levels" / "accounting.csv", dtype={"who": str})
    later = pd.read_csv(Path(report) / "accounting.csv", dtype={"who": str})
    return pd.concat([dev.assign(period="development"), later], ignore_index=True)


def _share(x: float) -> str:
    """A share of the perfect-knowledge offer, in percent: '0' when exactly zero."""
    if pd.isna(x):
        return "–"
    return "0" if x == 0 else f"{100 * x:.1f}"


def accounting_table(frame: pd.DataFrame, form: str = "average") -> str:
    """Table I's rows: per period, the exact band and the perfect-knowledge offer in kWh, then
    each later step as a share of the perfect-knowledge offer for the learned, population and
    stratified models, and the delivery rate of the rule as run. A step a period does not have
    (the second site sweeps the one-dimensional offer only) is a dash."""
    rows = frame[frame["form"] == form]
    head = "".join(f'<th colspan="3">{label}</th>' for label in PERIODS.values())
    models = "".join("<th>L</th><th>P</th><th>S</th>" for _ in PERIODS)
    out = [
        f"<thead><tr><th></th>{head}</tr><tr><th></th>{models}</tr></thead>",
        "<tbody>",
    ]

    def first(period: str, step: str) -> pd.Series | None:
        mine = rows[(rows["period"] == period) & (rows["step"] == step)]
        return mine.iloc[0] if len(mine) else None

    band, perfect = [], []
    for period in PERIODS:
        b, p = first(period, "exact band of the true sets"), first(period, "perfect knowledge")
        band.append(f'<td colspan="3">{b["size_mean_kwh"]:.1f}</td>' if b is not None else "")
        band[-1] = band[-1] or '<td colspan="3">–</td>'
        if p is None:
            perfect.append('<td colspan="3">–</td>')
        else:
            of_band = f" ({100 / b['share_of_perfect']:.0f}%)" if b is not None else ""
            perfect.append(f'<td colspan="3">{p["size_mean_kwh"]:.1f}{of_band}</td>')
    counts = []
    for period in PERIODS:
        mine = rows[rows["period"] == period]
        if "days" not in mine.columns or mine["days"].isna().all():
            counts.append('<td colspan="3">–</td>')
            continue
        every = mine.loc[mine["step"] == "perfect knowledge", "days"]
        fewest = mine.loc[mine["step"].isin(levels.SWEPT_STEPS), "days"].min()
        text = f"{int(every.iloc[0])}" if len(every) else "–"
        if pd.notna(fewest) and len(every) and fewest < every.iloc[0]:
            text += f" ({int(fewest)})"
        counts.append(f'<td colspan="3">{text}</td>')
    out.append("<tr><td>Scored weekdays (fewest in a row)</td>" + "".join(counts) + "</tr>")
    out.append("<tr><td>Exact band (kWh)</td>" + "".join(band) + "</tr>")
    out.append("<tr><td>Perfect knowledge (kWh)</td>" + "".join(perfect) + "</tr>")
    for step, label in SHARE_ROWS.items():
        cells, rates = [], []
        for period in PERIODS:
            mine = rows[(rows["period"] == period) & (rows["step"] == step)]
            by_model = {str(w)[0]: r for w, r in zip(mine["who"], mine.itertuples(), strict=True)}
            for model in MODELS:
                r = by_model.get(model)
                cells.append(f"<td>{_share(r.share_of_perfect) if r is not None else '–'}</td>")
                rate = f"{100 * r.delivered:.0f}" if r is not None else "–"
                rates.append(f"<td>{rate}</td>")
        out.append(f"<tr><td>{label}</td>" + "".join(cells) + "</tr>")
        if step == "day-level rule as run":
            out.append("<tr><td>&#8195;days delivered (%)</td>" + "".join(rates) + "</tr>")
    out.append("</tbody>")
    return "\n".join(out)


def grid_level_at_or_above(q: float) -> float:
    """The quantile grid level at or above `q`. A bound between two grid levels is interpolated
    linearly, so it is zero exactly when the bound at the upper level is zero (both are
    non-negative and the lower is no larger); the zero-ceiling share at `q` is the share there."""
    return float(min(level for level in LEVELS if level >= q - 1e-12))


def run_rule_levels(folder: Path) -> dict[str, float]:
    """The median shared level each registered rule (arms 4, 5 and 6) used on a run's scored
    weekdays, from its decision logs."""
    folder = Path(folder)
    days = pd.read_csv(folder / "days.csv.gz", dtype={"arm": str})
    days["decision_time"] = pd.to_datetime(days["decision_time"], utc=True)
    times = set(evaluation.scored_days(days)["decision_time"])
    logs = pd.read_csv(folder / "logs.csv.gz", dtype={"arm": str})
    logs["decision_time"] = pd.to_datetime(logs["decision_time"], utc=True)
    keep = logs["decision_time"].isin(times) & (logs["fitted"].astype(str) == "True")
    names = {"4": "learned rule", "5": "population rule", "6": "stratified rule"}
    return {
        name: float(logs.loc[keep & (logs["arm"] == arm), "level"].median())
        for arm, name in names.items()
    }


def zero_ceilings(table: pd.DataFrame, reached: dict[str, float]) -> dict[str, float | bool]:
    """The three readings of the zero-ceiling proposition (`docs/theory.md`) from a sweep table:
    at the level each registered rule reached, the share of cars that took energy but were
    given a zero energy ceiling (`at_<model>`; every car below the lowest grid level, where a
    bound is zero by definition); whether that share never rises with the level,
    for every model and offer form (`falls`); and its largest value at any positive level once
    the empty cars are known (`known_max`)."""
    rows = table[table["level"] > 0]
    form = sorted(rows["form"].unique())[0]  # the ceilings do not depend on the offer form
    column = "zero_ceiling_share_with_energy"
    out: dict[str, float | bool] = {}
    for model, rule in (("4", "learned rule"), ("5", "population rule"), ("6", "stratified rule")):
        out[f"level_{model}"] = reached[rule]
        if reached[rule] < LEVELS[0] - 1e-12:  # below the grid every bound is zero by definition
            out[f"at_{model}"] = 1.0
            continue
        level = grid_level_at_or_above(reached[rule])
        mine = rows[(rows["form"] == form) & (rows["model"] == model)]
        out[f"at_{model}"] = float(mine.loc[np.isclose(mine["level"], level), column].iloc[0])
    out["falls"] = all(
        (curve.sort_values("level")[column].diff().dropna() <= 1e-12).all()
        for _, curve in rows.groupby(["model", "form"])
    )
    out["known_max"] = float(rows.loc[rows["model"].str.endswith("k"), column].max())
    return out


def comparison_words(c: dict) -> str:
    """One paired comparison of offers (`evaluation.paired`, one-week blocks) in words: its
    reading, both mean offers with their delivery rates, and the difference with its interval
    when either offer is visible at two decimals."""
    if not c:
        return "–"
    sizes = f"{_kwh(c.get('mean_a_kwh'))} against {_kwh(c.get('mean_b_kwh'))}&#160;kWh"
    rates = f"delivered on {_rate(c.get('delivered_a'))} and {_rate(c.get('delivered_b'))} of days"
    reading = str(c.get("reading", "–")).replace("both arms offer zero", "both offers are zero")
    if reading.startswith("not testable"):
        return f"{reading} ({rates})"
    text = f"{reading}: {sizes}, {rates}"
    kwh = c.get("absolute_kwh", {})
    visible = max(abs(c.get("mean_a_kwh") or 0.0), abs(c.get("mean_b_kwh") or 0.0)) >= 0.005
    if kwh and visible:
        text += (
            f"; difference {_kwh(kwh.get('estimate'))}&#160;kWh "
            f"({_kwh(kwh.get('low'))} to {_kwh(kwh.get('high'))})"
        )
    return text


def hypotheses_table(hyp: dict) -> str:
    """Table II's rows, in words: each registered hypothesis, and H1 under the design
    calibrated on delivery, on the frozen evaluation and the sealed months; H1 at JPL on both
    tracks when that run is present. The 2022 theorem is left to the text."""
    ev, se, jpl = hyp["evaluation"], hyp["sealed"], hyp.get("jpl", {})

    def h2(h: dict) -> str:
        if not h:
            return "–"
        return (
            f"{h.get('reading', '–')}: delivered on {_rate(h.get('delivered_3'))} of days, "
            f"{_kwh(h.get('size_3_kwh'))}&#160;kWh against {_kwh(h.get('size_4_kwh'))} for the "
            "learned offer"
        )

    def h4(h: dict) -> str:
        if "learned_hours" not in h:
            return "–"
        days = h.get("car_days")
        return (
            f"{h.get('reading', '–')}: {h['learned_hours']:.2f} against "
            f"{h['population_hours']:.2f}&#160;h, coverage {_rate(h.get('learned_coverage'))} "
            f"and {_rate(h.get('population_coverage'))}"
            + (f", {days:,} car-days" if isinstance(days, int) else "")
        )

    h3 = se.get("h3", {}).get("4 against 4f", {})
    h3a, h3b = evaluation.h3_readings(h3) if "reopening" in h3 else ("–", "–")
    when = pd.Timestamp(se["reopening"]).strftime("%-d %B %Y") if se.get("reopening") else "–"
    h3_text = h3a if h3a == h3b else f"H3a {h3a}; H3b {h3b}"
    rows = [
        (
            "H1, learned offer at least 10% larger than population",
            "registered",
            comparison_words(ev.get("h1_registered", {}).get("one_week", {})),
            comparison_words(se.get("h1_registered", {}).get("one_week", {})),
        ),
        (
            "H1",
            "calibrated on delivery",
            comparison_words(ev.get("h1_secondary", {}).get("one_week", {})),
            comparison_words(se.get("h1_secondary", {}).get("one_week", {})),
        ),
        (
            "H2, union bound delivered on at least 98% of days, at most half the learned offer",
            "registered",
            h2(ev.get("h2", {})),
            h2(se.get("h2", {})),
        ),
        (f"H3a and H3b, after the reopening ({when})", "both", "–", h3_text),
        (
            "H4, learned stay bound at least 0.5&#160;h longer at matched 90% coverage",
            "per car",
            h4(ev.get("h4", {})),
            h4(se.get("h4", {})),
        ),
    ]
    out = [
        "<thead><tr><th>Hypothesis</th><th>Design</th><th>Caltech, Sep 2019–Feb 2020</th>"
        "<th>Caltech, from Mar 2020</th></tr></thead><tbody>"
    ]
    out += [f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows]
    for track, label in (("frozen", "Caltech pipeline frozen"), ("updating", "updating on JPL")):
        for key, design in (("h1_registered", "registered"), ("h1_secondary", "on delivery")):
            c = jpl.get(track, {}).get(key, {}).get("one_week", {})
            if c:
                out.append(
                    f"<tr><td>H1 at JPL, {label} (descriptive)</td><td>{design}</td>"
                    f'<td colspan="2">{comparison_words(c)}</td></tr>'
                )
    out.append("</tbody>")
    return "\n".join(out)


PREFIX = {
    "development": "dev",
    "evaluation": "ev",
    "sealed": "se",
    "sealed, 2021 only": "late",
    "jpl": "jpl",
}


def accounting_numbers(frame: pd.DataFrame) -> dict[str, str]:
    """The accounting's numbers the text quotes, per period (prefix `dev`, `ev`, `se`, `jpl`):
    on the day-average base set the exact band and perfect-knowledge offer (kWh) and their
    ratio; the smallest and largest share kept by the three models at each later step; the rule
    calibrated on delivery as run, per model, with its delivery rate. The same steps on the
    one-dimensional offer carry `_seg`. A step a period does not have gives a dash."""
    out: dict[str, str] = {}
    steps = {
        "forecasts at 90% delivery": "fc",
        "empty cars known": "known",
        "hindsight split at 90%": "hind",
    }
    for period, p in PREFIX.items():
        for form, suffix in (("average", ""), ("segment", "_seg")):
            rows = frame[(frame["period"] == period) & (frame["form"] == form)]
            band = rows.loc[rows["step"] == "exact band of the true sets", "size_mean_kwh"]
            perfect = rows.loc[rows["step"] == "perfect knowledge", "size_mean_kwh"]
            out[f"{p}{suffix}_pk"] = f"{perfect.iloc[0]:.1f}" if len(perfect) else "–"
            if form == "average":
                out[f"{p}_band"] = f"{band.iloc[0]:.1f}" if len(band) else "–"
                ok = len(band) and len(perfect)
                out[f"{p}_pk_of_band"] = pct(perfect.iloc[0] / band.iloc[0], 0) if ok else "–"
            for step, key in steps.items():
                shares = rows.loc[rows["step"] == step, "share_of_perfect"]
                out[f"{p}{suffix}_{key}_min"] = pct(shares.min()) if len(shares) else "–"
                out[f"{p}{suffix}_{key}_max"] = pct(shares.max()) if len(shares) else "–"
            run = rows[rows["step"] == "day-level rule as run"]
            for _, r in run.iterrows():
                who = str(r["who"])
                key = f"{p}{suffix}_rule_{who}"
                out[key] = f"{r['size_mean_kwh']:.2f}"
                out[key + "_share"] = pct(r["share_of_perfect"])
                out[key + "_share0"] = pct(r["share_of_perfect"], 0)
                out[key + "_delivered"] = pct(r["delivered"], 0)
    return out


WHERE = {
    "development": "on May to August 2019",
    "evaluation": "from September 2019 to February 2020",
    "sealed": "in the sealed months",
    "sealed, 2021 only": "in 2021 alone",
    "jpl": "at JPL",
}


def _leader(rows: pd.DataFrame, step: str) -> str | None:
    """The model (first character of `who`) keeping strictly the largest share at `step`."""
    shares = rows[rows["step"] == step].dropna(subset=["share_of_perfect"])
    if len(shares) < 2:
        return None
    ranked = shares.sort_values("share_of_perfect", ascending=False)
    first, second = ranked["share_of_perfect"].iloc[0], ranked["share_of_perfect"].iloc[1]
    return str(ranked["who"].iloc[0])[0] if first > second else None


def model_order_sentence(frame: pd.DataFrame) -> str:
    """Which model keeps the most of the perfect-knowledge offer under the fixed allocation and
    under the hindsight split (day-average base set), in each period the frame holds."""
    reversed_in, other_in, pairs = [], [], []
    for period, where in WHERE.items():
        rows = frame[(frame["period"] == period) & (frame["form"] == "average")]
        fixed = _leader(rows, "forecasts at 90% delivery")
        hind = _leader(rows, "hindsight split at 90%")
        if fixed is None and hind is None:
            continue
        if fixed == "5" and hind == "4":
            reversed_in.append(where)
            h = rows[rows["step"] == "hindsight split at 90%"]
            share = {str(w)[0]: s for w, s in zip(h["who"], h["share_of_perfect"], strict=True)}
            pairs.append(f"{pct(share['4'], 0)} and {pct(share['5'], 0)}")
        else:
            other_in.append(where)
    if not reversed_in:
        return ""
    text = (
        "The order of the models reverses between the two: under the fixed allocation the "
        "population model keeps the most, under the hindsight split the learned model does "
        f"(learned and population under the hindsight split: {'; '.join(pairs)}, in the order "
        "of Table I's columns)"
    )
    if other_in:
        text += f"; the order differs {', '.join(other_in)}"
    return text + "."


def zero_ceiling_numbers(readings: dict[str, dict]) -> dict[str, str]:
    """The zero-ceiling readings the text quotes, per period (`zero_ceilings` for each)."""
    out: dict[str, str] = {}
    for period, r in readings.items():
        p = PREFIX[period]
        for model in MODELS:
            out[f"{p}_zc_{model}"] = pct(r[f"at_{model}"], 0)
            out[f"{p}_level_{model}"] = pct(r[f"level_{model}"], 1)
        out[f"{p}_zc_falls"] = "holds" if r["falls"] else "does not hold"
        out[f"{p}_zc_known_max"] = pct(r["known_max"], 1)
    return out


def aci_bounds(folder: Path, start: pd.Timestamp | None) -> dict[str, float]:
    """For each adaptive day-level arm of a run (4, 5, 6 and 4d, 5d, 6d), the bound on its share
    of failed days over the run's scored weekdays, from the working level logged on the first
    of them (`daylevel.aci_failure_bound`; each outcome known before the next decision)."""
    folder = Path(folder)
    days = pd.read_csv(folder / "days.csv.gz", dtype={"arm": str}, low_memory=False)
    logs = pd.read_csv(folder / "logs.csv.gz", dtype={"arm": str}, low_memory=False)
    for frame in (days, logs):
        frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    out = {}
    for arm in ("4", "5", "6", "4d", "5d", "6d"):
        scored = evaluation.scored_days(days[days["arm"] == arm], start)
        mine = logs[(logs["arm"] == arm) & logs["decision_time"].isin(scored["decision_time"])]
        if mine.empty:
            continue
        first = float(mine.sort_values("decision_time")["miss_at_decision"].iloc[0])
        out[arm] = daylevel.aci_failure_bound(len(scored), start=first)
    return out


def sealed_composition(folder: Path, start: pd.Timestamp | None) -> dict[str, str]:
    """How the sealed months' scored weekdays split around the closure: those in 2020 (before
    it) and those from the reopening year on, from the perfect-knowledge arm's days."""
    days = pd.read_csv(Path(folder) / "days.csv.gz", dtype={"arm": str}, low_memory=False)
    days["decision_time"] = pd.to_datetime(days["decision_time"], utc=True)
    scored = evaluation.scored_days(days[days["arm"] == "1a"], start)
    year = scored["decision_time"].dt.tz_convert(evaluation.LOCAL_TZ).dt.year
    return {
        "se_days_2020": str(int((year == 2020).sum())),
        "se_days_2021": str(int((year > 2020).sum())),
    }


def sealed_zero_ceiling_sentence(r: dict) -> str:
    """The sentence on the sealed months' zero ceilings (`zero_ceilings` for that period). A
    rule whose median level is below the lowest quantile level gives every car a zero bound by
    construction, so those months do not test the proposition; the sentence says so."""
    medians = [r[f"level_{model}"] for model in MODELS]
    if all(q < LEVELS[0] - 1e-12 for q in medians):
        where = "0" if all(q == 0 for q in medians) else f"below the lowest level, {pct(LEVELS[0])}"
        return (
            f"In the sealed months the median level of every rule was {where}, where every bound "
            "is zero by construction, so those months do not test the proposition."
        )
    shares = ", ".join(pct(r[f"at_{m}"], 0) for m in ("4", "6")) + f" and {pct(r['at_5'], 0)}"
    return f"In the sealed months the shares are {shares}."


def _kwh(x) -> str:
    if x is None or pd.isna(x):
        return "–"
    return f"{0.0 if abs(x) < 0.005 else x:.2f}"  # no "-0.00"


def _rate(x) -> str:
    return "–" if x is None or pd.isna(x) else pct(x, 0)


def comparison_numbers(c: dict, key: str) -> dict[str, str]:
    """One paired comparison (`evaluation.paired`, one-week blocks) as the text quotes it."""
    kwh = c.get("absolute_kwh", {})
    return {
        f"{key}_reading": c.get("reading", "–"),
        f"{key}_days": str(c.get("days", "–")),
        f"{key}_a": _kwh(c.get("mean_a_kwh")),
        f"{key}_b": _kwh(c.get("mean_b_kwh")),
        f"{key}_da": _rate(c.get("delivered_a")),
        f"{key}_db": _rate(c.get("delivered_b")),
        f"{key}_diff": _kwh(kwh.get("estimate")),
        f"{key}_lo": _kwh(kwh.get("low")),
        f"{key}_hi": _kwh(kwh.get("high")),
    }


def hypothesis_numbers(hyp: dict, requirements: list[dict], available: list[dict]) -> dict:
    """The registered hypotheses' numbers the text quotes, from `fleetoffer report`."""
    ev, se, jpl = hyp["evaluation"], hyp["sealed"], hyp.get("jpl", {})
    out: dict[str, str] = {}
    for p, h in (("ev", ev), ("se", se)):
        for name, key in (("h1_registered", "h1_reg"), ("h1_secondary", "h1_sec")):
            out |= comparison_numbers(h.get(name, {}).get("one_week", {}), f"{p}_{key}")
        h4 = h.get("h4", {})
        out |= {
            f"{p}_h4_reading": h4.get("reading", "–"),
            f"{p}_h4_learned": f"{h4['learned_hours']:.2f}" if "learned_hours" in h4 else "–",
            f"{p}_h4_population": f"{h4['population_hours']:.2f}"
            if "population_hours" in h4
            else "–",
            f"{p}_h4_learned_cov": _rate(h4.get("learned_coverage")),
            f"{p}_h4_population_cov": _rate(h4.get("population_coverage")),
            f"{p}_h4_car_days": f"{h4.get('car_days', '–'):,}"
            if isinstance(h4.get("car_days"), int)
            else str(h4.get("car_days", "–")),
        }
    h2 = ev.get("h2", {})
    out |= {
        "h2_reading": h2.get("reading", "–"),
        "h2_delivered": _rate(h2.get("delivered_3")),
        "h2_size3": _kwh(h2.get("size_3_kwh")),
        "h2_size4": _kwh(h2.get("size_4_kwh")),
    }
    h3 = se.get("h3", {}).get("4 against 4f", {})
    a, b = evaluation.h3_readings(h3) if "reopening" in h3 else ("–", "–")
    out |= {
        "h3_reopening": pd.Timestamp(se.get("reopening")).strftime("%-d %B %Y")
        if se.get("reopening")
        else "–",
        "h3a_reading": a,
        "h3b_reading": b,
        "h3_days": str(h3.get("frozen_scored_weekdays", h3.get("adaptive_scored_weekdays", "–"))),
    }
    for track in ("frozen", "updating"):
        for name, key in (("h1_registered", "reg"), ("h1_secondary", "sec")):
            c = jpl.get(track, {}).get(name, {}).get("one_week", {})
            out |= comparison_numbers(c, f"jpl_{track}_{key}")
    need = [r["theorem4_from"] for r in requirements]
    out["theorem_days"] = " and ".join(f"{n:,}" for n in need)
    out["days_before_change"] = f"{available[0]['weekdays']:,}"
    return out


def offer_days(days: pd.DataFrame, arms: dict[str, str], key: str) -> dict[str, str]:
    """For each arm (as `{arm: name}`), on the scored weekdays of a run's `days.csv.gz`: how
    many carry an offer above the registered zero line of 1e-9 kWh, and the largest offer
    (kWh, two significant figures). Shows what a reading between two near-zero means rests on."""
    scored = evaluation.scored_days(days)
    out: dict[str, str] = {}
    for arm, name in arms.items():
        size = scored.loc[scored["arm"].astype(str) == arm, "size_kwh"]
        largest = float(size.max()) if len(size) else float("nan")
        out[f"{key}_pos_{name}"] = str(int((size > 1e-9).sum()))
        if pd.isna(largest):
            out[f"{key}_max_{name}"] = "–"
        else:
            out[f"{key}_max_{name}"] = "0" if largest <= 1e-9 else f"{largest:.2g}"
    return out


def mechanism_numbers(table: pd.DataFrame, median_cars: float) -> dict[str, str]:
    """From the model-selection sweep: how often every bound held beside delivery with the
    allocation fixed and with the hindsight split, at level 0.15 for the learned model, and at
    the lowest level with the empty cars known, beside the union bound's floor for that level."""
    rows = table[table["form"] == "average"]

    def at(model: str, level: float, column: str) -> float:
        mine = rows[(rows["model"] == model) & np.isclose(rows["level"], level)]
        return float(mine[column].iloc[0])

    low = float(LEVELS[0])
    return {
        "dev_held_4_015": pct(at("4", 0.15, "all_bounds_held"), 0),
        "dev_fixed_4_015": pct(at("4", 0.15, "fixed_delivered"), 0),
        "dev_hind_4_015": pct(at("4", 0.15, "hindsight_delivered"), 0),
        "dev_held_4k_low": pct(at("4k", low, "all_bounds_held"), 0),
        "dev_union_floor": pct(1 - 2 * median_cars * low, 0),
        "lowest_level": pct(low, 1),
    }


def later(
    results: Path, report: Path, reached: dict[str, float], numbers: dict, *, rehearsal: bool
) -> dict[str, str]:
    """Every number and table of Sections V and VI and the abstract, from the model-selection
    sweep, the runs after the tag and `fleetoffer report`."""
    results, report = Path(results), Path(report)
    made = json.loads((report / "report.json").read_text())
    frame = accounts(results, report)
    table = pd.read_csv(results / "levels" / "table.csv", dtype={"model": str})
    readings = {"development": zero_ceilings(table, reached)}
    for period in ("evaluation", "sealed"):
        folder = results / f"evaluate-{'rehearsal-' if rehearsal else ''}{period}"
        sweep = pd.read_csv(folder / "sweep_table.csv", dtype={"model": str})
        readings[period] = zero_ceilings(sweep, run_rule_levels(folder))
    out = accounting_numbers(frame) | zero_ceiling_numbers(readings)
    out["model_order_sentence"] = model_order_sentence(frame)
    out["se_zc_sentence"] = sealed_zero_ceiling_sentence(readings["sealed"])
    prefix = "rehearsal-" if rehearsal else ""
    bounds = aci_bounds(
        results / f"evaluate-{prefix}evaluation", evaluation.OFFERS_FROM[f"{prefix}evaluation"]
    )
    out["ev_aci_min"], out["ev_aci_max"] = pct(min(bounds.values())), pct(max(bounds.values()))
    out |= sealed_composition(
        results / f"evaluate-{prefix}sealed", evaluation.OFFERS_FROM[f"{prefix}sealed"]
    )
    out |= hypothesis_numbers(made["hypotheses"], made["theorem"], made["days_available"])
    frozen = results / f"evaluate-{'rehearsal-' if rehearsal else ''}evaluation"
    days = pd.read_csv(frozen / "days.csv.gz", dtype={"arm": str})
    out |= offer_days(days, {"4": "a", "5": "b"}, "ev_h1_reg")
    out |= mechanism_numbers(table, float(numbers["median_cars"]))
    out["table_accounting"] = accounting_table(frame)
    out["table_hypotheses"] = hypotheses_table(made["hypotheses"])
    return out
