import json

import pandas as pd

from staypromise.cli import main
from staypromise.pipeline import run_demo
from staypromise.synthetic import Regime

SHORT = [
    Regime("2001-01-01", sessions_per_weekday=20, median_stay_h=6.0),
    Regime("2001-03-01", sessions_per_weekday=20, median_stay_h=2.0, returning_share=0.3),
]


def test_demo_writes_every_output_and_a_manifest(tmp_path):
    manifest = run_demo(tmp_path, regimes=SHORT, end="2001-05-01", fit_until="2001-02-01")
    for name in ["sessions.csv", "decisions.csv.gz", "monthly.csv", "overall.csv", "monthly.png"]:
        assert (tmp_path / name).stat().st_size > 0
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["parameters"]["seed"] == 0
    assert on_disk["inputs"]["sessions"] == manifest["inputs"]["sessions"] > 0

    # After the change, the frozen rule keeps far fewer promises than a rolling one.
    monthly = pd.read_csv(tmp_path / "monthly.csv")
    april = monthly[monthly["month"] == "2001-04"].set_index("rule")["kept_share"]
    assert april["frozen"] < 0.6 < 0.8 < april["rolling 30 d"]


def test_command_line(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path / "demo")]) == 0
    assert "manifest.json" in capsys.readouterr().out
