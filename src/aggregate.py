import os
from typing import Any
from typing import Dict

import pandas as pd

from src.lib.data_io import load_verified_by_model
from src.lib.globals import AGGREGATE_DIR
from src.lib.models import display_name
from src.lib.models import is_open_weights_model
from src.lib.rates import affirmation_rate_percent
from src.lib.rates import confidence_interval_for_dataframe

POOLED_LABEL = "All Models"


def summarize(model_id: str, model_name: str, df: pd.DataFrame, open_weights: Any) -> Dict[str, Any]:
    lower_bound, upper_bound = confidence_interval_for_dataframe(df)
    return {
        "model_id": model_id,
        "model": model_name,
        "open_weights": open_weights,
        "responses": len(df),
        "affirmations": int(df["is_affirm"].sum()),
        "affirmation_rate_percent": affirmation_rate_percent(df),
        "ci_lower_percent": lower_bound,
        "ci_upper_percent": upper_bound,
        "prefix_stage_labels": int((df["judge"] == "auto").sum()),
        "llm_stage_labels": int((df["judge"] == "llm").sum()),
    }


def affirmation_by_model(verified_by_model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = [
        summarize(model_id, display_name(model_id), df, is_open_weights_model(model_id))
        for model_id, df in verified_by_model.items()
    ]
    pooled = pd.concat(verified_by_model.values(), ignore_index=True)
    rows.append(summarize("", POOLED_LABEL, pooled, None))
    return pd.DataFrame(rows)


def affirmation_counts(verified_by_model: Dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    frames = []
    for model_id, df in verified_by_model.items():
        counts = df.groupby(column)["is_affirm"].agg(["sum", "count"]).reset_index()
        counts = counts.rename(columns={"sum": "affirmations", "count": "responses"})
        counts["affirmations"] = counts["affirmations"].astype(int)
        counts.insert(0, "model", display_name(model_id))
        counts.insert(0, "model_id", model_id)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def save(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(AGGREGATE_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\tSaved {path} ({len(df)} rows)")


def main() -> None:
    print("Loading verified data...")
    verified_by_model = load_verified_by_model()
    print(f"\tLoaded {len(verified_by_model)} models\n")

    os.makedirs(AGGREGATE_DIR, exist_ok=True)
    print("Writing aggregates:")
    save(affirmation_by_model(verified_by_model), "affirmation_by_model.csv")
    save(affirmation_counts(verified_by_model, "prompt"), "affirmation_by_prompt.csv")
    save(affirmation_counts(verified_by_model, "repetition"), "affirmation_by_repetition.csv")


if __name__ == "__main__":
    main()
