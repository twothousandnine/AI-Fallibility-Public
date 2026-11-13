import os
import math
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

import src.lib.plot_styles as ps
from src.lib.data_io import load_prompts
from src.lib.data_io import load_verified_by_model
from src.lib.figure_data import save_figure_data
from src.lib.globals import FIGURE_DIR
from src.lib.globals import NUMBER_OF_REPETITIONS
from src.lib.models import is_open_weights_model
from src.lib.models import short_display_name


PREVIEW_MODE = False

AFFIRM_COLOR = "#E41A1C"
NON_AFFIRM_COLOR = "#4dAF4A"
FAILED_COLOR = "#999999"

GRID_COLUMNS = 4

TITLE_FONTSIZE = 100
LEGEND_FONTSIZE = 94
LABEL_FONTSIZE = 72
SECTION_LABEL_FONTSIZE = 96

SECTION_DIVIDER_COLOR = "#333333"
SECTION_DIVIDER_WIDTH = 4.0

SECTION_GAP_ROW_HEIGHT = 0.2
SECTION_LABEL_X = 0.012

OPEN_SECTION_LABEL = "Open\nweights"
CLOSED_SECTION_LABEL = "Closed\nSource"

LOCAL_DPI = 100
FIGURE_NAME = "fig_heatmap_grid"


def is_complete(group: pd.DataFrame) -> bool:
    return len(group) >= NUMBER_OF_REPETITIONS


def affirmation_matrix(df: pd.DataFrame, prompts: List[str]) -> np.ndarray:
    groups_by_prompt = {prompt: group for prompt, group in df.groupby("prompt", sort=False)}
    matrix = np.full((len(prompts), NUMBER_OF_REPETITIONS), np.nan)

    for row_index, prompt in enumerate(prompts):
        group = groups_by_prompt.get(prompt)
        if group is None or not is_complete(group):
            continue
        ordered = group.sort_values("repetition")
        matrix[row_index] = ordered["is_affirm"].astype(int).to_numpy()[:NUMBER_OF_REPETITIONS]

    return matrix


def sort_matrix_by_resistance(matrix: np.ndarray) -> np.ndarray:
    row_affirmation_counts = np.nansum(matrix, axis=1)
    row_order = np.argsort(row_affirmation_counts)
    row_sorted = matrix[row_order]

    column_affirmation_counts = np.nansum(row_sorted, axis=0)
    column_order = np.argsort(column_affirmation_counts)
    return row_sorted[:, column_order]


def total_affirmations(matrix: np.ndarray) -> float:
    return float(np.nansum(matrix))


def rejection_rates(matrix: np.ndarray) -> np.ndarray:
    valid_counts = np.sum(~np.isnan(matrix), axis=1)
    affirm_counts = np.nansum(matrix, axis=1)
    safe_valid = np.where(valid_counts == 0, 1, valid_counts)
    rates = 1.0 - (affirm_counts / safe_valid)
    rates[valid_counts == 0] = np.nan
    return rates


def draw_heatmap(axis: Axes, matrix: np.ndarray, title: str) -> None:
    colormap = ListedColormap([NON_AFFIRM_COLOR, AFFIRM_COLOR])
    colormap.set_bad(FAILED_COLOR)

    axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=colormap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    axis.set_title(title, fontsize=TITLE_FONTSIZE, fontweight="bold")
    axis.set_xticks([])
    axis.set_yticks([])


def draw_sidebar(axis: Axes, rates: np.ndarray) -> None:
    num_prompts = len(rates)
    y_positions = np.arange(num_prompts)

    rejection = np.where(np.isnan(rates), 0, rates)
    affirmation = 1.0 - rejection

    axis.barh(
        y_positions, rejection, color=NON_AFFIRM_COLOR, height=1.0, edgecolor="none",
    )
    axis.barh(
        y_positions, affirmation, left=rejection,
        color=AFFIRM_COLOR, height=1.0, edgecolor="none",
    )
    axis.set_xlim(0, 1.05)
    axis.set_ylim(num_prompts - 0.5, -0.5)
    axis.set_xticks([])
    axis.set_yticks([])


HEATMAP_WIDTH = 5.0
SIDEBAR_WIDTH = 1.0
PANEL_GAP_WIDTH = 0.8
HEATMAP_SIDEBAR_GAP_WIDTH = 0.4


def build_column_layout() -> tuple[List[float], List[int]]:
    column_widths: List[float] = []
    panel_column_starts: List[int] = []

    for panel_index in range(GRID_COLUMNS):
        panel_column_starts.append(len(column_widths))
        column_widths.append(HEATMAP_WIDTH)
        column_widths.append(HEATMAP_SIDEBAR_GAP_WIDTH)
        column_widths.append(SIDEBAR_WIDTH)
        is_last_panel = panel_index == GRID_COLUMNS - 1
        if not is_last_panel:
            column_widths.append(PANEL_GAP_WIDTH)

    return column_widths, panel_column_starts


def add_legend(figure: Figure, matrices: List[np.ndarray]) -> None:
    legend_handles = [
        mpatches.Patch(color=NON_AFFIRM_COLOR, label="Rejection"),
        mpatches.Patch(color=AFFIRM_COLOR, label="Affirmation"),
    ]
    if any(np.isnan(matrix).any() for matrix in matrices):
        legend_handles.append(mpatches.Patch(color=FAILED_COLOR, label="Failed"))

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=64,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )


def split_model_ids_by_source(matrices_by_model: Dict[str, np.ndarray]) -> tuple[List[str], List[str]]:
    def by_affirmations_descending(model_ids: List[str]) -> List[str]:
        return sorted(
            model_ids,
            key=lambda model_id: total_affirmations(matrices_by_model[model_id]),
            reverse=True,
        )

    open_model_ids = [model_id for model_id in matrices_by_model if is_open_weights_model(model_id)]
    closed_model_ids = [model_id for model_id in matrices_by_model if not is_open_weights_model(model_id)]

    return by_affirmations_descending(open_model_ids), by_affirmations_descending(closed_model_ids)


def section_row_count(model_count: int) -> int:
    return math.ceil(model_count / GRID_COLUMNS)


def draw_section(
        figure: Figure,
        grid,
        model_ids: List[str],
        matrices_by_model: Dict[str, np.ndarray],
        first_row: int,
        panel_column_starts: List[int],
    ) -> None:
    for index, model_id in enumerate(model_ids):
        row = first_row + index // GRID_COLUMNS
        heatmap_column = panel_column_starts[index % GRID_COLUMNS]
        heatmap_axis = figure.add_subplot(grid[row, heatmap_column])
        sidebar_axis = figure.add_subplot(grid[row, heatmap_column + 2], sharey=heatmap_axis)

        matrix = matrices_by_model[model_id]
        draw_heatmap(heatmap_axis, matrix, short_display_name(model_id))
        draw_sidebar(sidebar_axis, rejection_rates(matrix))


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


def band_center(figure: Figure, axis_indices_first: int, axis_indices_last: int) -> float:
    axes = figure.get_axes()
    top = max(axes[index].get_position().y1 for index in range(axis_indices_first, axis_indices_last))
    bottom = min(axes[index].get_position().y0 for index in range(axis_indices_first, axis_indices_last))
    return (top + bottom) / 2


def plot_grid(dataframes_by_model: Dict[str, pd.DataFrame], prompts: List[str]) -> Tuple[Figure, Dict[str, np.ndarray]]:
    ps.publication_style()
    plt.rcParams["figure.autolayout"] = False
    plt.rcParams["savefig.bbox"] = "standard"

    matrices_by_model = {
        model_id: sort_matrix_by_resistance(affirmation_matrix(df, prompts))
        for model_id, df in dataframes_by_model.items()
    }

    open_model_ids, closed_model_ids = split_model_ids_by_source(matrices_by_model)

    open_row_count = section_row_count(len(open_model_ids))
    closed_row_count = section_row_count(len(closed_model_ids))
    total_row_count = open_row_count + closed_row_count

    figure = plt.figure(figsize=(20 * GRID_COLUMNS, 18 * total_row_count + 18 * SECTION_GAP_ROW_HEIGHT))
    column_widths, panel_column_starts = build_column_layout()

    row_heights = [1.0] * open_row_count + [SECTION_GAP_ROW_HEIGHT] + [1.0] * closed_row_count
    closed_first_row = open_row_count + 1

    grid = figure.add_gridspec(
        len(row_heights),
        len(column_widths),
        height_ratios=row_heights,
        width_ratios=column_widths,
        wspace=0.0,
        hspace=0.32,
    )

    open_axis_first = len(figure.get_axes())
    draw_section(figure, grid, open_model_ids, matrices_by_model, 0, panel_column_starts)
    open_axis_last = len(figure.get_axes())
    draw_section(figure, grid, closed_model_ids, matrices_by_model, closed_first_row, panel_column_starts)
    closed_axis_last = len(figure.get_axes())

    figure.subplots_adjust(left=0.10, right=0.99, top=0.94, bottom=0.09)

    open_center = band_center(figure, open_axis_first, open_axis_last)
    closed_center = band_center(figure, open_axis_last, closed_axis_last)

    draw_section_divider(figure, open_axis_last)
    draw_section_label(figure, OPEN_SECTION_LABEL, y=open_center)
    draw_section_label(figure, CLOSED_SECTION_LABEL, y=closed_center)

    add_legend(figure, list(matrices_by_model.values()))
    return figure, matrices_by_model


def draw_section_divider(figure: Figure, open_axis_last: int) -> None:
    axes = figure.get_axes()
    open_bottom = min(axes[index].get_position().y0 for index in range(open_axis_last))
    closed_top = max(axes[index].get_position().y1 for index in range(open_axis_last, len(axes)))
    divider_y = (open_bottom + closed_top) / 2

    figure.add_artist(
        plt.Line2D(
            [0.02, 0.99],
            [divider_y, divider_y],
            color=SECTION_DIVIDER_COLOR,
            linewidth=SECTION_DIVIDER_WIDTH,
            transform=figure.transFigure,
        )
    )


def save_local(figure: Figure) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png")
    figure.savefig(path, dpi=LOCAL_DPI)
    print(f"\nSaved locally: {path}")
    return path


def matrices_dataframe(matrices_by_model: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for model_id, matrix in matrices_by_model.items():
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                rows.append({
                    "model": short_display_name(model_id),
                    "sorted_row": row_index + 1,
                    "sorted_column": column_index + 1,
                    "is_affirm": matrix[row_index, column_index],
                })
    return pd.DataFrame(rows)


def main() -> None:
    print("Loading verified data...")
    prompts = load_prompts()
    dataframes_by_model = load_verified_by_model()
    print(f"\tLoaded data for {len(dataframes_by_model)} models")

    figure, matrices_by_model = plot_grid(dataframes_by_model, prompts)

    if PREVIEW_MODE:
        plt.show()
    else:
        save_local(figure)
        save_figure_data(matrices_dataframe(matrices_by_model), FIGURE_NAME)

    plt.close(figure)


if __name__ == "__main__":
    main()
