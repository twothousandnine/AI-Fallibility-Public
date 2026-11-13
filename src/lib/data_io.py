import os
from typing import Dict
from typing import List

import pandas as pd

from src.lib.globals import PROMPT_FILEPATH
from src.lib.globals import RAW_DIR
from src.lib.globals import VERIFIED_DIR
from src.lib.models import MODEL_IDS
from src.lib.models import normalize_model_identifier


def load_prompts() -> List[str]:
    with open(PROMPT_FILEPATH, "r") as file:
        return file.read().splitlines()


def raw_path(model_id: str) -> str:
    return os.path.join(RAW_DIR, f"{normalize_model_identifier(model_id)}.csv")


def verified_path(model_id: str) -> str:
    return os.path.join(VERIFIED_DIR, f"{normalize_model_identifier(model_id)}.csv")


def load_raw(model_id: str) -> pd.DataFrame:
    return pd.read_csv(raw_path(model_id))


def load_verified(model_id: str) -> pd.DataFrame:
    return pd.read_csv(verified_path(model_id))


def load_verified_by_model() -> Dict[str, pd.DataFrame]:
    return {model_id: load_verified(model_id) for model_id in MODEL_IDS}
