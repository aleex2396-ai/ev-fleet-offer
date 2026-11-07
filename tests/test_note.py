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
