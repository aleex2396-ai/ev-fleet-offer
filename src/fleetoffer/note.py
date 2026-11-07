"""The note on the results (`note/note.html`): every number in it is read here from the result
tables, the template is filled, and the page is printed to PDF with a headless Chrome if one
is installed (otherwise the filled page is written alone and can be printed from any browser).

`values` names each number the template uses and says which table it comes from.
"""

from __future__ import annotations

import json
import os
import shutil
import string
import subprocess
from pathlib import Path

import pandas as pd

from fleetoffer import figures, levels
from fleetoffer.devchecks import SELECTION_START

TEMPLATE = Path(__file__).parents[2] / "note" / "note.html"
CHROME = "FLEETOFFER_CHROME"  # path of a Chrome or Chromium binary, if not the default below
DEFAULT_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MODELS = {"4": ("learned", 0), "5": ("population", 1), "6": ("stratified", 2)}


def pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def development(levels_dir: Path, checks_dir: Path) -> dict[str, str]:
    """Numbers from the model-selection months (May to August 2019): the level sweep
    (`fleetoffer levels`) and the checks (`fleetoffer checks`)."""
    checks = json.loads((Path(checks_dir) / "checks.json").read_text())
    sweep = pd.read_csv(Path(levels_dir) / "sweep.csv.gz", dtype={"model": str})
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
        cells = {}
        for _, r in mine.iterrows():
            cells[ACCOUNTING_WHO.get(str(r["who"]), str(r["who"]))] = cell(r)
        out.append(
            f"<tr><td>{label}</td>"
            + "".join(f"<td>{cells.get(m, '–')}</td>" for m in MODELS)
            + "</tr>"
        )
    return "\n".join(out)
