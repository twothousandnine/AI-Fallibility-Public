import json
import os
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.lib.globals import FIGURE_DATA_DIR


FigureData = Union[pd.DataFrame, Dict[str, Any], List[Any], np.ndarray]


def save_figure_data(data: FigureData, filename_stem: str) -> str:
    os.makedirs(FIGURE_DATA_DIR, exist_ok=True)

    if isinstance(data, pd.DataFrame):
        return _save_dataframe(data, filename_stem)
    if isinstance(data, np.ndarray):
        return _save_json(data.tolist(), filename_stem)
    return _save_json(data, filename_stem)


def _save_dataframe(dataframe: pd.DataFrame, filename_stem: str) -> str:
    path = os.path.join(FIGURE_DATA_DIR, f"{filename_stem}.csv")
    dataframe.to_csv(path, index=False)
    print(f"Saved figure data: {path}")
    return path


def _save_json(data: Any, filename_stem: str) -> str:
    path = os.path.join(FIGURE_DATA_DIR, f"{filename_stem}.json")
    with open(path, "w") as file:
        json.dump(data, file, indent=2, default=_json_default)
    print(f"Saved figure data: {path}")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
