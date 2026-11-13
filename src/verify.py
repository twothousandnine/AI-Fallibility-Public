import json
import re
from typing import Any
from typing import Dict

import pandas as pd

import src.lib.judge_rules as judge_rules
from src.lib.data_io import load_raw
from src.lib.data_io import load_verified
from src.lib.globals import JUDGE_TAG
from src.lib.models import MODEL_IDS
from src.lib.models import display_name

ANNOTATOR_ID = "annotator1"
JUDGE_MODEL_ID = "us.meta.llama4-maverick-17b-instruct-v1:0"
JUDGE_MAX_TOKENS = 200



def build_judge_prompt(system_prompt: str, claim: str, response: str) -> str:
    cleansed_response = judge_rules.cleanse_response(response)
    return f"{system_prompt}\nFalse Claim: {claim}\n\nResponse: {cleansed_response.strip()}"


def build_judge_request(judge_prompt: str) -> Dict[str, Any]:
    return {
        "modelId": JUDGE_MODEL_ID,
        "messages": [{"role": "user", "content": [{"text": judge_prompt}]}],
        "inferenceConfig": {"maxTokens": JUDGE_MAX_TOKENS},
    }


def query_llm_judge(judge_request: Dict[str, Any]) -> str:
    raise NotImplementedError(
        "This repository does not call the LLM judge. "
        "Its recorded labels are provided in data/verified/."
    )


def parse_judge_reply(reply_text: str) -> Dict[str, Any]:
    json_match = re.search(r"\{.*\}", reply_text, flags=re.DOTALL)
    if json_match is None:
        raise ValueError(f"No JSON object found in judge reply: {reply_text!r}")
    parsed_json = json.loads(json_match.group(0))
    return {"label": int(parsed_json["label"]), "rationale": parsed_json.get("rationale", "")}


def grade_response(claim: str, response: str, system_prompt: str) -> Dict[str, Any]:
    prefix_label = judge_rules.check_prefix_label(response)
    if prefix_label is not None:
        return {"is_affirm": prefix_label == 1, "judge": "auto", "rationale": None}

    judge_prompt = build_judge_prompt(system_prompt, claim, response)
    judge_reply = parse_judge_reply(query_llm_judge(build_judge_request(judge_prompt)))
    return {
        "is_affirm": judge_reply["label"] == 1,
        "judge": "llm",
        "rationale": judge_reply["rationale"],
    }


def merge_raw_with_recorded_labels(model_id: str) -> pd.DataFrame:
    raw = load_raw(model_id)
    recorded = load_verified(model_id)[["prompt", "repetition", "is_affirm", "judge"]]
    return raw.merge(recorded, on=["prompt", "repetition"], how="left", validate="one_to_one")


def check_model(model_id: str) -> Dict[str, int]:
    merged = merge_raw_with_recorded_labels(model_id)
    prefix_labels = merged["response"].map(judge_rules.check_prefix_label)
    handled_by_prefix = prefix_labels.notna()

    prefix_agrees = (prefix_labels[handled_by_prefix] == 1) == merged.loc[handled_by_prefix, "is_affirm"]
    recorded_as_auto = merged["judge"] == "auto"

    return {
        "responses": len(merged),
        "prefix_stage": int(handled_by_prefix.sum()),
        "llm_stage": int((~handled_by_prefix).sum()),
        "prefix_label_mismatches": int((~prefix_agrees).sum()),
        "stage_mismatches": int((handled_by_prefix != recorded_as_auto).sum()),
    }


def print_model_result(model_id: str, result: Dict[str, int]) -> None:
    print(
        f"\t{display_name(model_id)}: "
        f"{result['prefix_stage']} prefix, {result['llm_stage']} LLM judge, "
        f"{result['prefix_label_mismatches']} label mismatches, "
        f"{result['stage_mismatches']} stage mismatches"
    )


def main() -> None:
    print("Re-running the prefix stage on raw responses and comparing to recorded labels:")
    results = {model_id: check_model(model_id) for model_id in MODEL_IDS}
    for model_id, result in results.items():
        print_model_result(model_id, result)

    total_mismatches = sum(
        result["prefix_label_mismatches"] + result["stage_mismatches"]
        for result in results.values()
    )
    print(f"\nTotal mismatches across {len(results)} models: {total_mismatches}")


if __name__ == "__main__":
    main()
