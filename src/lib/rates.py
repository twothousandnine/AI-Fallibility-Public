import math
from typing import Tuple

import pandas as pd

Z_SCORE_95 = 1.959963984540054


def affirmation_rate_percent(df: pd.DataFrame) -> float:
    return float(df["is_affirm"].mean() * 100)


def wilson_confidence_interval_percent(affirmation_count: int, total_count: int) -> Tuple[float, float]:
    sample_proportion = affirmation_count / total_count
    z = Z_SCORE_95
    denominator = 1 + z**2 / total_count
    center = sample_proportion + z**2 / (2 * total_count)
    margin = z * math.sqrt(
        sample_proportion * (1 - sample_proportion) / total_count + z**2 / (4 * total_count**2)
    )
    lower_bound = (center - margin) / denominator
    upper_bound = (center + margin) / denominator
    return lower_bound * 100, upper_bound * 100


def confidence_interval_for_dataframe(df: pd.DataFrame) -> Tuple[float, float]:
    affirmation_count = int(df["is_affirm"].sum())
    return wilson_confidence_interval_percent(affirmation_count, len(df))
