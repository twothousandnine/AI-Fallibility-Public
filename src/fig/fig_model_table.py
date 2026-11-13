import os
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import src.lib.plot_styles as ps
from src.lib.figure_data import save_figure_data
from src.lib.globals import FIGURE_DIR


PREVIEW_MODE = False

FIGURE_NAME = "fig_model_table"
LOCAL_DPI = 200

HEADER_FONT_SIZE = 13
BODY_FONT_SIZE = 12
CAPTION_FONT_SIZE = 14

TEXT_COLOR = "#000000"

BODY_FONT = "cmr10"
HEADER_FONT = "cmb10"
ITALIC_FONT = "cmti10"

HEAVY_RULE_WIDTH = 1.4
LIGHT_RULE_WIDTH = 0.7

ROW_HEIGHT_INCHES = 0.28
MARGIN_HEIGHT_INCHES = 1.3
FIGURE_WIDTH_INCHES = 14.0

TOP_RULE_Y = 0.995
HEADER_Y = 0.955
HEADER_RULE_Y = 0.915
FIRST_ROW_Y = 0.865
LAST_ROW_Y = 0.06
BOTTOM_RULE_Y = 0.01

COLUMN_HEADERS = ["Model", "Provider", "Parameters", "Access", "Location", "Notable features"]
COLUMN_POSITIONS = [0.000, 0.210, 0.310, 0.430, 0.545, 0.685]

UNDISCLOSED_LABEL = "Not disclosed"
OPEN_LABEL = "Open weights"
CLOSED_LABEL = "Closed"


@dataclass
class ModelRow:
    model: str
    provider: str
    parameters: str | None
    open_weights: bool
    location: str
    notable: str


MODEL_ROWS = [
    ModelRow(
        model="Claude Opus 4.7",
        provider="Anthropic",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Adaptive extended thinking; 1M context",
    ),
    ModelRow(
        model="Claude Sonnet 4.6",
        provider="Anthropic",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Mid-tier; extended thinking; 1M context",
    ),
    ModelRow(
        model="DeepSeek V3.2",
        provider="DeepSeek",
        parameters="685B",
        open_weights=True,
        location="OUS China",
        notable="Mixture-of-experts; 164K context",
    ),
    ModelRow(
        model="DeepSeek-R1",
        provider="DeepSeek",
        parameters="671B",
        open_weights=True,
        location="OUS China",
        notable="Chain-of-thought reasoning; 128K context",
    ),
    ModelRow(
        model="Gemma 4 31B",
        provider="Google",
        parameters="30.7B",
        open_weights=True,
        location="US",
        notable="Dense; native function calling; 256K context",
    ),
    ModelRow(
        model="GPT-5.4",
        provider="OpenAI",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Reasoning-first; computer use; 272K context",
    ),
    ModelRow(
        model="GPT-5.6 Luna",
        provider="OpenAI",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Agentic flagship; 272K context",
    ),
    ModelRow(
        model="Grok 4.3",
        provider="xAI",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Always-on configurable reasoning; 1M context",
    ),
    ModelRow(
        model="Llama 3.3 70B Instruct",
        provider="Meta",
        parameters="70B",
        open_weights=True,
        location="US",
        notable="Efficiency-focused; 128K context",
    ),
    ModelRow(
        model="Llama 4 Maverick 17B Instruct",
        provider="Meta",
        parameters="17B active",
        open_weights=True,
        location="US",
        notable="Mixture-of-experts, 128 experts; 1M context",
    ),
    ModelRow(
        model="Llama 4 Scout 17B Instruct",
        provider="Meta",
        parameters="17B active",
        open_weights=True,
        location="US",
        notable="Mixture-of-experts, 16 experts; 10M context",
    ),
    ModelRow(
        model="Mistral Large 3",
        provider="Mistral AI",
        parameters="675B",
        open_weights=True,
        location="OUS France",
        notable="Multilingual and code focused; 256K context",
    ),
    ModelRow(
        model="Nova Pro",
        provider="Amazon",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Multimodal; 300K context",
    ),
    ModelRow(
        model="Qwen3 Next 80B A3B",
        provider="Alibaba Cloud",
        parameters="80B / 3B active",
        open_weights=True,
        location="OUS China",
        notable="Sparse mixture-of-experts; 256K context",
    ),
    ModelRow(
        model="GLM 5.3",
        provider="Z.ai",
        parameters="355B",
        open_weights=True,
        location="OUS China",
        notable="Large-scale reasoning model; 1M context",
    ),
    ModelRow(
        model="DeepSeek V4 Pro",
        provider="DeepSeek",
        parameters="1.6T",
        open_weights=True,
        location="OUS China",
        notable="Intelligent mixture-of-experts; 1M context",
    ),
    ModelRow(
        model="Kimi K3",
        provider="Moonshot AI",
        parameters="2.8T",
        open_weights=True,
        location="OUS China",
        notable="Comprised of Kimi Delta Attention (KDA) architecture; 1M context",
    ),
    ModelRow(
        model="Qwen3.8 Max",
        provider="Alibaba Cloud",
        parameters="2.4T",
        open_weights=True,
        location="OUS China",
        notable="Mixture-of-experts multi-modal; 1M context",
    ),
    ModelRow(
        model="Muse Glimmer 30B",
        provider="Meta",
        parameters="30B",
        open_weights=True,
        location="US",
        notable="Dense distilled & multilingual model; 131k context",
    ),
    ModelRow(
        model="Gemini 3.8 Flash",
        provider="Google",
        parameters=None,
        open_weights=False,
        location="US",
        notable="Software engineering & agentic reasoning gains; 1M context",
    ),
]


def open_weight_rows_sorted() -> List[ModelRow]:
    open_rows = [row for row in MODEL_ROWS if row.open_weights]
    return sorted(open_rows, key=lambda row: row.model.lower())


def closed_rows_sorted() -> List[ModelRow]:
    closed_rows = [row for row in MODEL_ROWS if not row.open_weights]
    return sorted(closed_rows, key=lambda row: row.model.lower())


def parameters_cell(parameters: str | None) -> tuple[str, str]:
    if parameters is None:
        return UNDISCLOSED_LABEL, ITALIC_FONT
    return parameters, BODY_FONT


def access_label(open_weights: bool) -> str:
    if open_weights:
        return OPEN_LABEL
    return CLOSED_LABEL


def draw_rule(axis: Axes, y: float, linewidth: float) -> None:
    axis.plot(
        [0.0, 1.0],
        [y, y],
        color=TEXT_COLOR,
        linewidth=linewidth,
        solid_capstyle="butt",
    )


def draw_cell(axis: Axes, x: float, y: float, text: str, font: str, size: float) -> None:
    axis.text(
        x,
        y,
        text,
        fontsize=size,
        fontfamily=font,
        color=TEXT_COLOR,
        va="center",
    )


def draw_header(axis: Axes) -> None:
    for x, header in zip(COLUMN_POSITIONS, COLUMN_HEADERS):
        draw_cell(axis, x, HEADER_Y, header, HEADER_FONT, HEADER_FONT_SIZE)


def draw_row(axis: Axes, y: float, row: ModelRow) -> None:
    parameters_text, parameters_font = parameters_cell(row.parameters)

    cells = [
        (row.model, BODY_FONT),
        (row.provider, BODY_FONT),
        (parameters_text, parameters_font),
        (access_label(row.open_weights), BODY_FONT),
        (row.location, BODY_FONT),
        (row.notable, BODY_FONT),
    ]

    for x, (text, font) in zip(COLUMN_POSITIONS, cells):
        draw_cell(axis, x, y, text, font, BODY_FONT_SIZE)


def plot_model_table() -> Figure:
    ps.publication_style()

    open_rows = open_weight_rows_sorted()
    closed_rows = closed_rows_sorted()
    total_rows = len(open_rows) + len(closed_rows)
    row_step = (FIRST_ROW_Y - LAST_ROW_Y) / (total_rows - 1)
    figure_height = ROW_HEIGHT_INCHES * total_rows + MARGIN_HEIGHT_INCHES

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH_INCHES, figure_height))
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    draw_rule(axis, TOP_RULE_Y, HEAVY_RULE_WIDTH)
    draw_header(axis)
    draw_rule(axis, HEADER_RULE_Y, LIGHT_RULE_WIDTH)

    row_index = 0
    for row in open_rows:
        draw_row(axis, FIRST_ROW_Y - row_index * row_step, row)
        row_index += 1

    section_rule_y = FIRST_ROW_Y - (row_index - 0.5) * row_step
    draw_rule(axis, section_rule_y, LIGHT_RULE_WIDTH)

    for row in closed_rows:
        draw_row(axis, FIRST_ROW_Y - row_index * row_step, row)
        row_index += 1

    draw_rule(axis, BOTTOM_RULE_Y, HEAVY_RULE_WIDTH)

    axis.set_title(
        "Models evaluated.",
        loc="center",
        pad=12,
        fontfamily=BODY_FONT,
        fontsize=CAPTION_FONT_SIZE,
        fontweight="normal",
    )
    return figure


def save_local(figure: Figure) -> str:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    path = os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png")
    figure.savefig(path, dpi=LOCAL_DPI, bbox_inches="tight")
    print(f"Saved locally: {path}")
    return path


def model_rows_dataframe() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": row.model,
            "provider": row.provider,
            "parameters": row.parameters,
            "open_weights": row.open_weights,
            "location": row.location,
            "notable": row.notable,
        }
        for row in MODEL_ROWS
    ])


def main() -> None:
    figure = plot_model_table()

    open_weight_count = sum(1 for row in MODEL_ROWS if row.open_weights)
    disclosed_count = sum(1 for row in MODEL_ROWS if row.parameters is not None)

    print(f"\nRendered {len(MODEL_ROWS)} models")
    print(f"\tOpen weights: {open_weight_count}")
    print(f"\tClosed: {len(MODEL_ROWS) - open_weight_count}")
    print(f"\tParameter count published: {disclosed_count}")

    if PREVIEW_MODE:
        plt.show()
    else:
        save_local(figure)
        save_figure_data(model_rows_dataframe(), FIGURE_NAME)

    plt.close(figure)


if __name__ == "__main__":
    main()
