"""Figures. Every figure is drawn from a table that is also written to disk."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical colours in a fixed order, validated for colour-blind separation as a set of
# three. Colour follows the order the caller lists the rules in, so a rule keeps its colour
# across figures.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK = "#0b0b0b"
INK_2 = "#52514e"
RULE_LINE = "#c3c2b7"
GRID = "#e9e8e3"
SURFACE = "#fcfcfb"
LABEL_GAP_PT = 9.0


def plot_monthly(
    monthly: pd.DataFrame,
    level: float,
    path: Path,
    *,
    rules: list[str],
    title: str,
    note: str | None = None,
    min_offered: int = 20,
) -> Path:
    """Two stacked panels sharing the time axis: share of promises kept, and hours promised.

    Months with fewer than `min_offered` promises are left blank rather than drawn from a
    handful of cars.
    """
    if len(rules) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} rules per figure; got {len(rules)}")

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 5.8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_facecolor(SURFACE)
    halo = [pe.Stroke(linewidth=3.4, foreground=SURFACE), pe.Normal()]
    ends: dict = {top: [], bottom: []}

    for k, rule in enumerate(rules):
        m = monthly[monthly["rule"] == rule].sort_values("month")
        x = m["month"].dt.to_timestamp() + pd.Timedelta(days=14)
        enough = m["offered"] >= min_offered
        for ax, y in (
            (top, (100 * m["kept_share"]).where(enough)),
            (bottom, m["mean_promised_h"].where(enough)),
        ):
            ax.plot(
                x, y, color=SERIES[k], lw=1.6, solid_capstyle="round", label=rule, path_effects=halo
            )
            valid = y.notna().to_numpy().nonzero()[0]
            if valid.size:
                ends[ax].append((x.iloc[valid[-1]], float(y.iloc[valid[-1]]), rule))

    target = 100 * (1 - level)
    top.axhline(target, color=INK_2, lw=0.9, zorder=1)
    top.annotate(
        f"{target:.0f}% target",
        xy=(0.0, target),
        xycoords=("axes fraction", "data"),
        xytext=(2, 3),
        textcoords="offset points",
        color=INK_2,
        fontsize=8,
    )
    top.set_ylabel("promises kept (%)", color=INK_2, fontsize=9)
    top.set_ylim(0, 100)
    bottom.set_ylabel("hours promised\n(mean per car)", color=INK_2, fontsize=9)
    bottom.set_ylim(bottom=0)
    for ax in (top, bottom):
        _quiet_axes(ax)
    bottom.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    top.legend(loc="lower left", frameon=False, fontsize=8, ncol=len(rules), labelcolor=INK_2)

    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    bottom_margin = 0.0
    if note:
        wrapped = textwrap.fill(note, width=120)
        fig.text(0.01, 0.01, wrapped, fontsize=7.5, color=INK_2, ha="left", va="bottom")
        bottom_margin = 0.025 * (wrapped.count("\n") + 1) + 0.01
    fig.tight_layout(rect=(0, bottom_margin, 0.9, 0.97))

    # Line-end labels are placed after layout, so their spacing is measured on the final axes.
    for ax, items in ends.items():
        _label_line_ends(fig, ax, items)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def _label_line_ends(fig, ax, items) -> None:
    """Label each line at its last point, nudging labels apart so none overlap."""
    if not items:
        return
    to_pt = 72.0 / fig.dpi
    placed = []
    for x, y, text in items:
        y_pt = ax.transData.transform((mdates.date2num(x), y))[1] * to_pt
        placed.append([y_pt, y_pt, x, y, text])
    placed.sort(key=lambda p: p[0])
    for i in range(1, len(placed)):
        placed[i][1] = max(placed[i][1], placed[i - 1][1] + LABEL_GAP_PT)
    shift = (sum(p[0] for p in placed) - sum(p[1] for p in placed)) / len(placed)
    for want, got, x, y, text in placed:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(5, got + shift - want),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
            color=INK_2,
            annotation_clip=False,
        )


def _quiet_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE_LINE)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
