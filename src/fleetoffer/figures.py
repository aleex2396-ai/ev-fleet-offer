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
    min_given: int = 20,
) -> Path:
    """Two stacked panels sharing the time axis: share of bounds that held, and hours bounded.

    Months with fewer than `min_given` bounds are left blank rather than drawn from a
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
        enough = m["given"] >= min_given
        for ax, y in (
            (top, (100 * m["held_share"]).where(enough)),
            (bottom, m["mean_bound_h"].where(enough)),
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
    top.set_ylabel("bounds that held (%)", color=INK_2, fontsize=9)
    top.set_ylim(0, 100)
    bottom.set_ylabel("hours bounded\n(mean per car)", color=INK_2, fontsize=9)
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


def _finish(fig, path: Path, title: str, note: str | None, right: float = 0.97) -> Path:
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    bottom_margin = 0.0
    if note:
        wrapped = textwrap.fill(note, width=120)
        fig.text(0.01, 0.01, wrapped, fontsize=7.5, color=INK_2, ha="left", va="bottom")
        bottom_margin = 0.025 * (wrapped.count("\n") + 1) + 0.01
    fig.tight_layout(rect=(0, bottom_margin, right, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_offers_monthly(
    monthly: pd.DataFrame,
    path: Path,
    *,
    arms: dict[str, tuple[str, int, str]],
    title: str,
    note: str | None = None,
    target: float = 0.90,
) -> Path:
    """Figure 2: per month, the share of scored weekdays delivered under the fixed allocation
    and the mean offer size, one line per arm. `arms` maps an arm to (label, colour index,
    line style)."""
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.0, 5.8), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2, 3]}
    )
    fig.patch.set_facecolor(SURFACE)
    for arm, (label, colour, style) in arms.items():
        m = monthly[monthly["arm"] == arm].sort_values("month")
        if m.empty:
            continue
        x = pd.PeriodIndex(m["month"], freq="M").to_timestamp() + pd.Timedelta(days=14)
        line = {"color": SERIES[colour], "ls": style, "lw": 1.5, "marker": "o", "ms": 2.5}
        top.plot(x, 100 * m["fixed_delivered"], label=label, **line)
        bottom.plot(x, m["size_mean_kwh"], label=label, **line)
    top.axhline(100 * target, color=INK_2, lw=0.9, zorder=1)
    top.set_ylabel("days delivered (%)", color=INK_2, fontsize=9)
    top.set_ylim(0, 102)
    bottom.set_ylabel("offer size (kWh, mean)", color=INK_2, fontsize=9)
    bottom.set_ylim(bottom=0)
    for ax in (top, bottom):
        _quiet_axes(ax)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    bottom.legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK_2)
    return _finish(fig, path, title, note)


def plot_frontier(
    panels: dict[str, pd.DataFrame],
    path: Path,
    *,
    models: dict[str, tuple[str, int, str]],
    form: str,
    title: str,
    note: str | None = None,
    measure: str = "fixed_delivered",
    oracle: dict[str, float] | None = None,
    target: float = 0.90,
    monthly: pd.DataFrame | None = None,
    monthly_arms: dict[str, tuple[str, int, str]] | None = None,
) -> Path:
    """Figure 3: mean offer size against the share of scored weekdays delivered, one curve per
    model across the swept levels, one panel per period. `models` maps a model to (label,
    colour index, line style); `oracle` gives each panel's perfect-knowledge size. With
    `monthly`, a last panel gives each arm of `monthly_arms` its share of days delivered by
    month, with the interval from resampling whole weeks."""
    extra = monthly is not None and bool(monthly_arms)
    count = len(panels) + int(extra)
    fig, axes = plt.subplots(1, count, figsize=(4.2 * count, 4.2), dpi=150)
    axes = [axes] if count == 1 else list(axes)
    fig.patch.set_facecolor(SURFACE)
    if extra:
        ax = axes[-1]
        for arm, (label, colour, style) in monthly_arms.items():
            m = monthly[monthly["arm"] == arm].sort_values("month")
            if m.empty:
                continue
            x = pd.PeriodIndex(m["month"], freq="M").to_timestamp() + pd.Timedelta(days=14)
            ax.plot(
                x,
                100 * m["fixed_delivered"],
                color=SERIES[colour],
                ls=style,
                lw=1.3,
                marker="o",
                ms=2.5,
            )
            ax.fill_between(
                x,
                100 * m["fixed_delivered_low"],
                100 * m["fixed_delivered_high"],
                color=SERIES[colour],
                alpha=0.15,
                lw=0,
                label=label,
            )
        ax.axhline(100 * target, color=INK_2, lw=0.8)
        ax.set_ylim(0, 102)
        ax.set_title("days delivered by month, 95% week-block interval", fontsize=9, loc="left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
        ax.legend(loc="lower left", frameon=False, fontsize=7, labelcolor=INK_2)
        _quiet_axes(ax)
    for ax, (name, table) in zip(axes, panels.items(), strict=False):
        rows = table[table["form"] == form]
        for model, (label, colour, style) in models.items():
            curve = rows[rows["model"] == model].sort_values("level")
            if curve.empty:
                continue
            ax.plot(
                100 * curve[measure],
                curve["size_mean_kwh"],
                color=SERIES[colour],
                ls=style,
                lw=1.4,
                marker="o",
                ms=2.5,
                label=label,
            )
        if oracle and name in oracle:
            ax.axhline(oracle[name], color=INK_2, lw=0.8, ls=":")
            ax.annotate(
                f"perfect knowledge {oracle[name]:.1f} kWh",
                xy=(0.0, oracle[name]),
                xycoords=("axes fraction", "data"),
                xytext=(2, 3),
                textcoords="offset points",
                color=INK_2,
                fontsize=7,
            )
        ax.axvline(100 * target, color=INK_2, lw=0.8)
        ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.set_xlabel("days delivered (%)", color=INK_2, fontsize=9)
        ax.set_xlim(right=101)
        # Linear below 1 kWh, logarithmic above: the large offers of high levels, never
        # delivered, would otherwise flatten the region near the delivery target.
        ax.set_yscale("symlog", linthresh=1.0, linscale=0.5)
        ax.set_ylim(bottom=0)
        ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        _quiet_axes(ax)
    axes[0].set_ylabel("offer size (kWh, mean)", color=INK_2, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5, labelcolor=INK_2)
    return _finish(fig, path, title, note)


def plot_failures(
    per_day: pd.DataFrame, path: Path, *, title: str, note: str | None = None, seed: int = 0
) -> Path:
    """Figure 4: for each day, the share of present cars whose learned set was not inside
    their true set, against whether the offer was delivered under each check."""
    import numpy as np

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (column, label) in zip(
        axes,
        (
            ("fixed_delivered", "allocation fixed at 10:00"),
            ("hindsight_delivered", "hindsight split"),
        ),
        strict=True,
    ):
        y = per_day[column].astype(float).to_numpy()
        jitter = y + rng.uniform(-0.15, 0.15, len(y))
        ax.scatter(100 * per_day["share_mispredicted"], jitter, s=6, color=SERIES[0], alpha=0.6)
        ax.set_yticks([0, 1], ["not delivered", "delivered"])
        ax.set_xlabel("present cars mispredicted (%)", color=INK_2, fontsize=9)
        ax.set_title(label, fontsize=9, color=INK, loc="left")
        ax.set_xlim(-2, 102)
        _quiet_axes(ax)
    return _finish(fig, path, title, note)


def plot_zero_ceilings(
    table: pd.DataFrame,
    path: Path,
    *,
    models: dict[str, tuple[str, int]],
    reached: dict[str, float] | None = None,
    title: str,
    note: str | None = None,
) -> Path:
    """The check of the zero-ceiling proposition (`docs/theory.md`): at each swept level, the
    share of cars that did take energy but were given an energy ceiling of zero, and the share
    of all cars given a deadline of zero, one colour per model; solid with the models as run,
    dashed with the cars that take nothing known (rows of model `<model>k`). The ceilings do
    not depend on the offer form, so one form's rows are read. `reached` marks levels the
    day-level rules reached, as vertical lines."""
    form = sorted(table["form"].unique())[0]
    rows = table[(table["form"] == form) & (table["level"] > 0)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=150, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    panels = (
        ("zero_ceiling_share_with_energy", "energy ceiling zero, among cars that took energy"),
        ("zero_stay_share", "deadline zero (leaves within 15 minutes), among all cars"),
    )
    for ax, (column, label) in zip(axes, panels, strict=True):
        for model, (name, colour) in models.items():
            for suffix, style in (("", "-"), ("k", "--")):
                curve = rows[rows["model"] == model + suffix].sort_values("level")
                if curve.empty:
                    continue
                ax.plot(
                    100 * curve["level"],
                    100 * curve[column],
                    color=SERIES[colour],
                    ls=style,
                    lw=1.3,
                    marker="o",
                    ms=2.2,
                    label=name + (", empty cars known" if suffix else ""),
                )
        for reach_label, level in (reached or {}).items():
            ax.axvline(100 * level, color=INK_2, lw=0.6, ls=":")
            ax.annotate(
                reach_label,
                xy=(100 * level, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                va="top",
                fontsize=6.5,
                color=INK_2,
            )
        ax.set_xscale("log")
        ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 95]
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("shared level q (%)", color=INK_2, fontsize=9)
        ax.set_title(label, fontsize=8.5, color=INK, loc="left")
        ax.set_ylim(-2, 102)
        _quiet_axes(ax)
    axes[0].set_ylabel("share of cars (%)", color=INK_2, fontsize=9)
    axes[1].legend(loc="upper right", frameon=False, fontsize=7, labelcolor=INK_2)
    return _finish(fig, path, title, note)
