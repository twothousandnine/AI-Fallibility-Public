import os

LIB_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(LIB_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
VERIFIED_DIR = os.path.join(DATA_DIR, "verified")
AGGREGATE_DIR = os.path.join(DATA_DIR, "aggregate")
FIGURE_DATA_DIR = os.path.join(DATA_DIR, "figure_data")
FIGURE_DIR = os.path.join(ROOT_DIR, "figures")

PROMPT_FILEPATH = os.path.join(LIB_DIR, "all_prompts.txt")
STYLE_FILEPATH = os.path.join(LIB_DIR, "color.mplstyle")

NUMBER_OF_PROMPTS = 100
NUMBER_OF_REPETITIONS = 50

JUDGE_TAG = "judge_b4f24dea"
