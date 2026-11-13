import os
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

import src.lib.plot_styles as ps
from src.lib.data_io import load_verified_by_model
from src.lib.figure_data import save_figure_data
from src.lib.globals import FIGURE_DIR
from src.lib.models import short_display_name
from src.lib.rates import affirmation_rate_percent
from src.lib.rates import confidence_interval_for_dataframe


PREVIEW_MODE = False

CI_ERROR_BAR_COLOR = "#000000"
CI_CAP_SIZE = 4

BAR_COLOR = "#C0392B"
REJECTION_COLOR = "#D9D9D9"
MEAN_LINE_COLOR = "#B0B0B0"
GRID_COLOR = "#E2E2E2"

MODEL_LABEL_FONTSIZE = 19
VALUE_LABEL_FONTSIZE = 19
CI_LABEL_FONTSIZE = 12
CI_LABEL_COLOR = "#8C8C8C"
AXIS_LABEL_FONTSIZE = 18
MEAN_LABEL_FONTSIZE = MODEL_LABEL_FONTSIZE

GRID_LINE_COUNT = 4

PANEL_WIDTH_RATIOS = [5, 1]
OVERALL_BAR_LABEL = "All Models"
AFFIRMATION_SEGMENT_LABEL = "Affirm"
REJECTION_SEGMENT_LABEL = "Reject"

LOCAL_DPI = 100
FIGURE_NAME = "fig_affirmation_rate"


def verified_by_short_name() -> Dict[str, pd.DataFrame]:
    return {
        short_display_name(model_id): df
        for model_id, df in load_verified_by_model().items()
    }


def pooled_affirmation_rate_percent(model_dataframes: Dict[str, pd.DataFrame]) -> float:
    all_responses = pd.concat(model_dataframes.values(), ignore_index=True)
    return affirmation_rate_percent(all_responses)


def pooled_confidence_interval_percent(model_dataframes: Dict[str, pd.DataFrame]) -> Tuple[float, float]:
    all_responses = pd.concat(model_dataframes.values(), ignore_index=True)
    return confidence_interval_for_dataframe(all_responses)


def rates_ascending(
    model_dataframes: Dict[str, pd.DataFrame]
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    rates_by_model = {
        model_name: affirmation_rate_percent(df)
        for model_name, df in model_dataframes.items()
    }
    ordered = sorted(rates_by_model.items(), key=lambda item: item[1])
    model_names = [model_name for model_name, _ in ordered]
    rates = np.array([rate for _, rate in ordered])

    confidence_intervals = [
        confidence_interval_for_dataframe(model_dataframes[model_name])
        for model_name in model_names
    ]
    lower_bounds = np.array([lower for lower, _ in confidence_intervals])
    upper_bounds = np.array([upper for _, upper in confidence_intervals])

    return model_names, rates, lower_bounds, upper_bounds


def draw_bars(
    axis: Axes,
    model_names: List[str],
    rates: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> None:
    y_positions = np.arange(len(model_names))

    bars = axis.barh(
        y_positions,
        rates,
        color=BAR_COLOR,
        height=0.72,
        edgecolor="none",
        zorder=3,
    )

    lower_errors = rates - lower_bounds
    upper_errors = upper_bounds - rates
    axis.errorbar(
        rates,
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor=CI_ERROR_BAR_COLOR,
        elinewidth=1.8,
        capsize=CI_CAP_SIZE + 2,
        capthick=1.8,
        zorder=4,
    )

    axis.set_yticks(y_positions)
    axis.set_yticklabels(model_names, fontsize=MODEL_LABEL_FONTSIZE, fontweight="bold")
    axis.tick_params(axis="y", length=0)
    axis.set_ylim(-0.7, len(model_names) - 0.3)

    label_x_offset = 0.015 * float(upper_bounds.max())
    for y_position, rate, lower_bound, upper_bound in zip(
        y_positions, rates, lower_bounds, upper_bounds
    ):
        lower_error = rate - lower_bound
        upper_error = upper_bound - rate
        rate_label = axis.text(
            upper_bound + label_x_offset,
            y_position,
            f"{rate:.1f}%",
            va="center",
            fontsize=VALUE_LABEL_FONTSIZE,
            fontweight="bold",
        )
        axis.annotate(
            f"[−{lower_error:.1f}, +{upper_error:.1f}]",
            xy=(1, 0.5),
            xycoords=rate_label,
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=CI_LABEL_FONTSIZE,
            fontweight="normal",
            color=CI_LABEL_COLOR,
        )


def draw_mean_line(axis: Axes, rates: np.ndarray) -> None:
    mean_rate = float(rates.mean())

    axis.axvline(
        mean_rate,
        color=MEAN_LINE_COLOR,
        linestyle=":",
        linewidth=1.4,
        zorder=4,
        label=f"mean {mean_rate:.1f}%",
    )

    legend = axis.legend(
        loc="lower right",
        frameon=False,
        prop={"weight": "bold", "size": MEAN_LABEL_FONTSIZE},
    )
    for legend_text in legend.get_texts():
        legend_text.set_color(MEAN_LINE_COLOR)


def style_value_axis(axis: Axes, upper_bounds: np.ndarray) -> None:
    upper_limit = min(100.0, float(upper_bounds.max()) * 1.35 + 4)
    axis.set_xlim(0, upper_limit)
    axis.set_xlabel("Affirmation Rate (%)", fontsize=AXIS_LABEL_FONTSIZE, fontweight="bold")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=GRID_LINE_COUNT, integer=True))
    axis.tick_params(axis="x", labelsize=13)
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontweight("bold")
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.9, zorder=0)
    axis.set_axisbelow(True)


def draw_overall_stacked_bar(
    axis: Axes,
    pooled_rate: float,
    pooled_lower_bound: float,
    pooled_upper_bound: float,
) -> None:
    rejection_rate = 100.0 - pooled_rate

    affirmation_bars = axis.bar(
        [0],
        [pooled_rate],
        color="#C0392B",
        width=0.55,
        edgecolor="none",
        zorder=3,
    )

    rejection_bars = axis.bar(
        [0],
        [rejection_rate],
        bottom=[pooled_rate],
        color="#27AE60",
        width=0.55,
        edgecolor="none",
        zorder=3,
    )

    axis.set_xticks([0])
    axis.set_xticklabels(
        [OVERALL_BAR_LABEL],
        fontsize=MODEL_LABEL_FONTSIZE,
        fontweight="bold",
    )
    axis.tick_params(axis="x", length=0)
    axis.set_xlim(-0.6, 0.6)

    axis.bar_label(
        affirmation_bars,
        labels=[f"{AFFIRMATION_SEGMENT_LABEL}\n{pooled_rate:.1f}%"],
        label_type="center",
        color="#FFFFFF",
        fontsize=VALUE_LABEL_FONTSIZE,
        fontweight="bold",
    )

    axis.bar_label(
        rejection_bars,
        labels=[f"{REJECTION_SEGMENT_LABEL}\n{rejection_rate:.1f}%"],
        label_type="center",
        color="#FFFFFF",
        fontsize=VALUE_LABEL_FONTSIZE,
        fontweight="bold",
    )

    lower_error = pooled_rate - pooled_lower_bound
    upper_error = pooled_upper_bound - pooled_rate
    axis.errorbar(
        [0],
        [pooled_rate],
        yerr=[[lower_error], [upper_error]],
        fmt="none",
        ecolor=CI_ERROR_BAR_COLOR,
        elinewidth=1.4,
        capsize=CI_CAP_SIZE,
        zorder=4,
    )


def style_overall_axis(axis: Axes) -> None:
    axis.set_ylim(0, 100)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=GRID_LINE_COUNT, integer=True))
    axis.yaxis.tick_right()
    axis.tick_params(axis="y", labelsize=13)
    for tick_label in axis.get_yticklabels():
        tick_label.set_fontweight("bold")
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.9, zorder=0)
    axis.set_axisbelow(True)


def plot_total_affirmation_figure(model_dataframes: Dict[str, pd.DataFrame]) -> Figure:
    ps.publication_style()

    model_names, rates, lower_bounds, upper_bounds = rates_ascending(model_dataframes)
    pooled_rate = pooled_affirmation_rate_percent(model_dataframes)
    pooled_lower_bound, pooled_upper_bound = pooled_confidence_interval_percent(model_dataframes)
    figure_height = max(6.5, 0.7 * len(model_names) + 2.0)

    figure, (model_axis, overall_axis) = plt.subplots(
        1,
        2,
        figsize=(16, figure_height),
        gridspec_kw={"width_ratios": PANEL_WIDTH_RATIOS, "wspace": 0.08},
    )

    draw_bars(model_axis, model_names, rates, lower_bounds, upper_bounds)
    draw_mean_line(model_axis, rates)
    style_value_axis(model_axis, upper_bounds)

    draw_overall_stacked_bar(overall_axis, pooled_rate, pooled_lower_bound, pooled_upper_bound)
    style_overall_axis(overall_axis)

    return figure


def save_local(figure: Figure) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png")
    figure.savefig(path, dpi=LOCAL_DPI, bbox_inches="tight")
    print(f"\nSaved locally: {path}")
    return path


def rates_dataframe(model_dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names, rates, lower_bounds, upper_bounds = rates_ascending(model_dataframes)
    pooled_rate = pooled_affirmation_rate_percent(model_dataframes)
    pooled_lower_bound, pooled_upper_bound = pooled_confidence_interval_percent(model_dataframes)

    rows = [
        {
            "model": model_name,
            "affirmation_rate_percent": rate,
            "ci_lower_percent": lower_bound,
            "ci_upper_percent": upper_bound,
        }
        for model_name, rate, lower_bound, upper_bound in zip(
            model_names, rates, lower_bounds, upper_bounds
        )
    ]
    rows.append(
        {
            "model": OVERALL_BAR_LABEL,
            "affirmation_rate_percent": pooled_rate,
            "ci_lower_percent": pooled_lower_bound,
            "ci_upper_percent": pooled_upper_bound,
        }
    )
    return pd.DataFrame(rows)


def print_summary(model_dataframes: Dict[str, pd.DataFrame]) -> None:
    model_names, rates, lower_bounds, upper_bounds = rates_ascending(model_dataframes)

    print("\nAffirmation rate per model (95% CI):")
    for model_name, rate, lower_bound, upper_bound in zip(
        reversed(model_names), reversed(rates), reversed(lower_bounds), reversed(upper_bounds)
    ):
        total_responses = len(model_dataframes[model_name])
        affirmations = int(model_dataframes[model_name]["is_affirm"].sum())
        print(
            f"\t{model_name}: {rate:.1f}% [{lower_bound:.1f}%, {upper_bound:.1f}%] "
            f"({affirmations}/{total_responses})"
        )

    print(f"\n\tMean across {len(model_names)} models: {rates.mean():.1f}%")

    pooled_responses = sum(len(df) for df in model_dataframes.values())
    pooled_affirmations = sum(int(df["is_affirm"].sum()) for df in model_dataframes.values())
    pooled_rate = pooled_affirmation_rate_percent(model_dataframes)
    pooled_lower_bound, pooled_upper_bound = pooled_confidence_interval_percent(model_dataframes)
    print(
        f"\tAll models pooled: {pooled_rate:.1f}% "
        f"[{pooled_lower_bound:.1f}%, {pooled_upper_bound:.1f}%] "
        f"({pooled_affirmations}/{pooled_responses})"
    )


def main() -> None:
    print("Loading verified data...")
    model_dataframes = verified_by_short_name()
    print(f"\tLoaded data for {len(model_dataframes)} models")

    print_summary(model_dataframes)

    figure = plot_total_affirmation_figure(model_dataframes)

    if PREVIEW_MODE:
        plt.show()
    else:
        save_local(figure)
        save_figure_data(rates_dataframe(model_dataframes), FIGURE_NAME)

    plt.close(figure)


if __name__ == "__main__":
    main()
