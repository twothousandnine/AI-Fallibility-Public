import os
import math
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import src.lib.plot_styles as ps
from src.lib.data_io import load_verified_by_model
from src.lib.figure_data import save_figure_data
from src.lib.globals import FIGURE_DIR
from src.lib.globals import NUMBER_OF_REPETITIONS
from src.lib.models import is_open_weights_model
from src.lib.models import short_display_name


PREVIEW_MODE = False

BAR_COLOR = "#C0392B"
EMPTY_COLOR = "#ECECEC"

GRID_COLUMNS = 4

TITLE_FONTSIZE = 28
AXIS_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 20
COUNT_LABEL_FONTSIZE = 24
COUNT_LABEL_COLOR = "#C0392B"

SECTION_LABEL_FONTSIZE = 34
SECTION_DIVIDER_COLOR = "#333333"
SECTION_DIVIDER_WIDTH = 2.0

SECTION_GAP_ROW_HEIGHT = 0.6
LEFT_MARGIN = 0.12
SECTION_LABEL_X = 0.01

OPEN_SECTION_LABEL = "Open\nweights"
CLOSED_SECTION_LABEL = "Closed\nSource"

GRIDLINE_VALUE = 25
GRIDLINE_COLOR = "#B0B0B0"

MINIMUM_BAR_SLOTS = 8

LOCAL_DPI = 100
FIGURE_NAME = "fig_affirmed_prompts"


def affirmations_per_prompt_ascending(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("prompt")["is_affirm"].sum().astype(int)
    return np.sort(counts.to_numpy())


def wrong_prompt_counts(df: pd.DataFrame) -> np.ndarray:
    counts = affirmations_per_prompt_ascending(df)
    return counts[counts > 0]


def draw_panel(axis: Axes, counts: np.ndarray, title: str, shared_slot_count: int, is_left_edge: bool) -> None:
    positions = shared_slot_count - len(counts) + np.arange(len(counts))

    axis.bar(positions, counts, color=BAR_COLOR, width=0.7, edgecolor="none")

    axis.set_title(
        title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
        pad=8,
    )
    axis.text(
        0.03,
        0.95,
        f"{len(counts)} Prompts",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=COUNT_LABEL_FONTSIZE,
        fontweight="bold",
        color=COUNT_LABEL_COLOR,
    )
    axis.set_ylim(0, NUMBER_OF_REPETITIONS)
    axis.set_yticks([0, GRIDLINE_VALUE, NUMBER_OF_REPETITIONS])
    if is_left_edge:
        axis.tick_params(axis="y", labelsize=TICK_FONTSIZE, labelleft=True)
    else:
        axis.tick_params(axis="y", labelsize=TICK_FONTSIZE, labelleft=False)
    axis.set_xticks([])
    axis.set_xlim(-1, shared_slot_count)
    axis.margins(x=0)
    axis.grid(False)
    axis.axhline(GRIDLINE_VALUE, color=GRIDLINE_COLOR, linewidth=1.0, zorder=0)
    axis.set_facecolor(EMPTY_COLOR)


def split_model_ids_by_source(counts_by_model: Dict[str, np.ndarray]) -> tuple[List[str], List[str]]:
    def by_affirmations_descending(model_ids: List[str]) -> List[str]:
        return sorted(model_ids, key=lambda model_id: counts_by_model[model_id].sum(), reverse=True)

    open_model_ids = [model_id for model_id in counts_by_model if is_open_weights_model(model_id)]
    closed_model_ids = [model_id for model_id in counts_by_model if not is_open_weights_model(model_id)]

    return by_affirmations_descending(open_model_ids), by_affirmations_descending(closed_model_ids)


def section_row_count(model_count: int) -> int:
    return math.ceil(model_count / GRID_COLUMNS)


def draw_section(
        figure: Figure,
        grid,
        model_ids: List[str],
        counts_by_model: Dict[str, np.ndarray],
        first_row: int,
        shared_slot_count: int,
    ) -> List[Axes]:
    section_axes: List[Axes] = []

    for index, model_id in enumerate(model_ids):
        row = first_row + index // GRID_COLUMNS
        column = index % GRID_COLUMNS
        axis = figure.add_subplot(grid[row, column])
        draw_panel(
            axis,
            counts_by_model[model_id],
            short_display_name(model_id),
            shared_slot_count,
            is_left_edge=(column == 0),
        )
        section_axes.append(axis)

    return section_axes


def draw_section_label(figure: Figure, text: str, y: float) -> None:
    figure.text(
        SECTION_LABEL_X,
        y,
        text,
        fontsize=SECTION_LABEL_FONTSIZE,
        fontweight="bold",
        ha="left",
        va="center",
        rotation=0,
    )


def plot_grid(dataframes_by_model: Dict[str, pd.DataFrame]) -> Tuple[Figure, Dict[str, np.ndarray]]:
    ps.publication_style()

    counts_by_model = {
        model_id: wrong_prompt_counts(df)
        for model_id, df in dataframes_by_model.items()
    }
    open_model_ids, closed_model_ids = split_model_ids_by_source(counts_by_model)
    shared_slot_count = max(MINIMUM_BAR_SLOTS, max(len(counts) for counts in counts_by_model.values()))

    open_row_count = section_row_count(len(open_model_ids))
    closed_row_count = section_row_count(len(closed_model_ids))
    total_row_count = open_row_count + closed_row_count

    figure = plt.figure(
        figsize=(5.5 * GRID_COLUMNS, 4.2 * (total_row_count + SECTION_GAP_ROW_HEIGHT)),
        layout="constrained",
    )
    figure.get_layout_engine().set(rect=(LEFT_MARGIN, 0.0, 1.0 - LEFT_MARGIN, 1.0))
    row_heights = [1.0] * open_row_count + [SECTION_GAP_ROW_HEIGHT] + [1.0] * closed_row_count
    closed_first_row = open_row_count + 1
    grid = figure.add_gridspec(len(row_heights), GRID_COLUMNS, height_ratios=row_heights)

    open_axes = draw_section(figure, grid, open_model_ids, counts_by_model, 0, shared_slot_count)
    closed_axes = draw_section(
        figure, grid, closed_model_ids, counts_by_model, closed_first_row, shared_slot_count
    )

    figure.canvas.draw()
    all_axes = open_axes + closed_axes
    left_axes_x = min(ax.get_position().x0 for ax in all_axes)

    open_bottom = min(ax.get_position().y0 for ax in open_axes)
    open_top = max(ax.get_position().y1 for ax in open_axes)
    closed_top = max(ax.get_position().y1 for ax in closed_axes)
    closed_bottom = min(ax.get_position().y0 for ax in closed_axes)
    divider_y = (open_bottom + closed_top) / 2

    figure.set_layout_engine("none")

    draw_section_label(figure, OPEN_SECTION_LABEL, y=(open_top + open_bottom) / 2)
    draw_section_divider(figure, divider_y, left_axes_x)
    draw_section_label(figure, CLOSED_SECTION_LABEL, y=(closed_top + closed_bottom) / 2)
    return figure, counts_by_model


def draw_section_divider(figure: Figure, divider_y: float, left_x: float) -> None:
    figure.add_artist(
        plt.Line2D(
            [left_x, 0.99],
            [divider_y, divider_y],
            color=SECTION_DIVIDER_COLOR,
            linewidth=SECTION_DIVIDER_WIDTH,
            transform=figure.transFigure,
        )
    )


def save_local(figure: Figure) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png")
    figure.savefig(path, dpi=LOCAL_DPI, bbox_inches="tight")
    print(f"\nSaved locally: {path}")
    return path


def counts_dataframe(counts_by_model: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_id, counts in counts_by_model.items():
        for rank, count in enumerate(counts, start=1):
            rows.append({
                "model": short_display_name(model_id),
                "rank": rank,
                "affirmations_for_prompt": int(count),
            })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading verified data...")
    dataframes_by_model = load_verified_by_model()
    print(f"\tLoaded data for {len(dataframes_by_model)} models")

    figure, counts_by_model = plot_grid(dataframes_by_model)

    if PREVIEW_MODE:
        plt.show()
    else:
        save_local(figure)
        save_figure_data(counts_dataframe(counts_by_model), FIGURE_NAME)

    plt.close(figure)


if __name__ == "__main__":
    main()
