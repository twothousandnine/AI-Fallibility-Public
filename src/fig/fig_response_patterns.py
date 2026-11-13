import os
import textwrap
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import src.lib.plot_styles as ps
from src.lib.data_io import load_prompts
from src.lib.data_io import load_verified
from src.lib.figure_data import save_figure_data
from src.lib.globals import FIGURE_DIR
from src.lib.globals import NUMBER_OF_REPETITIONS


PREVIEW_MODE = False

MODEL_ID = "qwen.qwen3-next-80b-a3b"

LINE_COLOR = "#377EB8"
FILL_COLOR = "#C5B3E6"
MIDLINE_COLOR = "#9A9A9A"

PROMPT_FONTSIZE = 21
STANCE_FONTSIZE = 14
REPETITION_LABEL_FONTSIZE = 16
TICK_FONTSIZE = 12

PROMPT_WRAP_WIDTH = 70

LOCAL_DPI = 100
FIGURE_NAME = "fig_response_patterns"

PATTERN_KEYS = [
    "consistent_disagreement",
    "reverberation",
    "rapid_oscillation",
    "consistent_agreement",
]


def build_trajectory(group: pd.DataFrame) -> np.ndarray:
    ordered = group.sort_values("repetition")
    return ordered["is_affirm"].astype(int).to_numpy()[:NUMBER_OF_REPETITIONS]


def count_transitions(trajectory: np.ndarray) -> int:
    return int(np.sum(np.diff(trajectory) != 0))


def classify_prompts(df: pd.DataFrame, prompts: List[str]) -> dict:
    consistent_disagreement = []
    reverberation = []
    rapid_oscillation = []
    consistent_agreement = []

    groups_by_prompt = {prompt: group for prompt, group in df.groupby("prompt", sort=False)}

    for prompt in prompts:
        group = groups_by_prompt.get(prompt)
        if group is None or len(group) < NUMBER_OF_REPETITIONS:
            continue

        trajectory = build_trajectory(group)
        affirm_rate = trajectory.mean()
        transitions = count_transitions(trajectory)

        if affirm_rate == 0.0:
            consistent_disagreement.append((prompt, trajectory, transitions))
        elif affirm_rate == 1.0:
            consistent_agreement.append((prompt, trajectory, transitions))
        elif transitions >= 3 and affirm_rate > 0.3:
            first_half_rate = trajectory[:NUMBER_OF_REPETITIONS // 2].mean()
            if first_half_rate < 0.2:
                reverberation.append((prompt, trajectory, transitions))
            else:
                rapid_oscillation.append((prompt, trajectory, transitions))
        elif transitions >= 2:
            rapid_oscillation.append((prompt, trajectory, transitions))
        else:
            if affirm_rate > 0.5:
                consistent_agreement.append((prompt, trajectory, transitions))
            else:
                consistent_disagreement.append((prompt, trajectory, transitions))

    return {
        "consistent_disagreement": consistent_disagreement,
        "reverberation": reverberation,
        "rapid_oscillation": rapid_oscillation,
        "consistent_agreement": consistent_agreement,
    }


def pick_representative(candidates: List[Tuple[str, np.ndarray, int]], pattern: str) -> Tuple[str, np.ndarray]:
    if not candidates:
        blank = np.zeros(NUMBER_OF_REPETITIONS, dtype=int)
        return ("(no example found)", blank)

    if pattern == "consistent_disagreement":
        best = min(candidates, key=lambda x: x[2])
    elif pattern == "reverberation":
        best = max(candidates, key=lambda x: x[2])
    elif pattern == "rapid_oscillation":
        best = max(candidates, key=lambda x: x[2])
    elif pattern == "consistent_agreement":
        best = max(candidates, key=lambda x: (x[1].mean(), -x[2]))
    else:
        best = candidates[0]

    return (best[0], best[1])


def wrap_prompt(prompt: str) -> str:
    cleaned = prompt.rstrip().rstrip(".")
    lines = textwrap.wrap(cleaned, width=PROMPT_WRAP_WIDTH, max_lines=2, placeholder="...")
    return "\n".join(lines)


def plot_zigzag_panel(axis: Axes, trajectory: np.ndarray, prompt: str, is_bottom_panel: bool) -> None:
    repetitions = np.arange(1, len(trajectory) + 1)

    axis.axhline(0.5, color=MIDLINE_COLOR, linestyle=":", linewidth=1.6, zorder=0)
    axis.fill_between(repetitions, trajectory, 0, color=FILL_COLOR, linewidth=0, zorder=1)
    axis.plot(repetitions, trajectory, color=LINE_COLOR, linewidth=2.6, zorder=2)

    axis.set_xlim(1, len(trajectory))
    axis.set_ylim(-0.18, 1.18)
    axis.grid(False)

    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Disagreed", "Agreed"], fontsize=STANCE_FONTSIZE, fontweight="bold")
    axis.tick_params(axis="y", length=0)

    axis.set_title(wrap_prompt(prompt), fontsize=PROMPT_FONTSIZE, fontweight="bold", pad=14)

    if is_bottom_panel:
        axis.xaxis.set_major_locator(plt.MaxNLocator(nbins=6, integer=True))
        axis.tick_params(axis="x", labelsize=TICK_FONTSIZE)
        axis.set_xlabel("Repetitions", fontsize=REPETITION_LABEL_FONTSIZE, fontweight="bold")
    else:
        axis.set_xticks([])


def plot_zigzag_figure(df: pd.DataFrame, prompts: List[str]) -> Tuple[Figure, List[Tuple[str, np.ndarray]]]:
    ps.publication_style()

    classified = classify_prompts(df, prompts)

    print("\nPattern counts:")
    for key in PATTERN_KEYS:
        print(f"\t{key}: {len(classified[key])}")

    representatives = [pick_representative(classified[key], key) for key in PATTERN_KEYS]

    figure, axes = plt.subplots(len(representatives), 1, figsize=(8, 9.5))

    for index, (axis, (prompt, trajectory)) in enumerate(zip(axes, representatives)):
        is_bottom_panel = index == len(representatives) - 1
        plot_zigzag_panel(axis, trajectory, prompt, is_bottom_panel)

    figure.tight_layout(h_pad=3.0)
    return figure, representatives


def save_local(figure: Figure) -> str:
    path = os.path.join(FIGURE_DIR, f"{FIGURE_NAME}.png")
    os.makedirs(FIGURE_DIR, exist_ok=True)
    figure.savefig(path, dpi=LOCAL_DPI, bbox_inches="tight")
    print(f"\nSaved locally: {path}")
    return path


def representatives_dataframe(representatives: List[Tuple[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for pattern, (prompt, trajectory) in zip(PATTERN_KEYS, representatives):
        rows.append({
            "pattern": pattern,
            "prompt": prompt,
            "trajectory": trajectory.tolist(),
            "affirmation_rate": float(trajectory.mean()),
            "transitions": count_transitions(trajectory),
        })
    return pd.DataFrame(rows)


def main() -> None:
    prompts = load_prompts()
    df = load_verified(MODEL_ID)
    print(f"Loaded {len(df)} verified rows for {MODEL_ID}")

    figure, representatives = plot_zigzag_figure(df, prompts)

    if PREVIEW_MODE:
        plt.show()
    else:
        save_local(figure)
        save_figure_data(representatives_dataframe(representatives), FIGURE_NAME)

    plt.close(figure)


if __name__ == "__main__":
    main()
