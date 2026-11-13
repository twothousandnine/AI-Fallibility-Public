# AI Fallibility: Data and Analysis Code

This repository accompanies the manuscript on LLM affirmation of misinformation under repeated conversational pressure. It contains the raw chatbot responses, the graded (verified) responses, and the code used to grade, aggregate, and plot them.

Twenty chatbots each received 100 plainly false statements (`src/lib/all_prompts.txt`). Each statement was repeated 50 times within one conversation, so each model produced 5,000 responses. Every response was then labeled as affirming or rejecting the false statement.

This repository does not contact any model or cloud service. Response collection and LLM-judge calls were done beforehand; their outputs are included as data.

## Layout

```
data/
  raw/            one CSV per model: unprocessed chatbot responses
  verified/       one CSV per model: raw responses plus the grader's label
  aggregate/      summary tables written by src/aggregate.py
  figure_data/    the exact data behind each figure, written by the figure scripts
figures/          figure PNGs written by the figure scripts
src/
  verify.py       the grading procedure (prefix rules + LLM judge) and an offline check
  aggregate.py    builds the summary tables in data/aggregate/
  fig/            one script per manuscript figure
  lib/            shared code: paths, model list, grading rules, data loading, plot style
Makefile
pyproject.toml
```

Files in `data/raw/` and `data/verified/` are named by the model ID lowercased with non-alphanumeric characters removed. For example, `us.anthropic.claude-opus-4-7` becomes `usanthropicclaudeopus47.csv`. The full mapping from model ID to display name is in `src/lib/models.py`.

## Data formats

`data/raw/<model>.csv`

| column       | meaning                                   |
| ------------ | ----------------------------------------- |
| `prompt`     | The false statement sent to the model.    |
| `response`   | The model's reply.                        |
| `repetition` | Position in the conversation, 1 to 50.    |

`data/verified/<model>.csv` contains the raw columns plus:

| column      | meaning                                                                  |
| ----------- | ------------------------------------------------------------------------ |
| `sample_id` | Stable hash identifying the response.                                    |
| `is_affirm` | `True` if the response affirmed the false statement.                     |
| `judge`     | `auto` if labeled by the prefix rules, `llm` if labeled by the LLM judge. |
| `rationale` | One-sentence explanation from the LLM judge; empty for `auto` labels.   |

Responses were truncated during collection to 2,048 words or 8,192 characters, or to 400 characters when at least 75% of the characters were non-ASCII.

## How responses were graded

Grading has two stages, both defined in `src/lib/judge_rules.py` and `src/verify.py`.

1. **Prefix rules.** The response is cleaned (emoji, markdown symbols, and curly punctuation normalized) and checked against fixed lists of affirming openers ("You're right", "That's correct") and rejecting openers ("That's incorrect", "False"). A match sets the label directly.
2. **LLM judge.** Responses with no prefix match were sent to `us.meta.llama4-maverick-17b-instruct-v1:0` with the rubric in `load_system_prompt()`. The judge returned a label and a one-sentence rationale.

The verified files were produced under judge tag `judge_b4f24dea`. The tag is a hash of the annotator ID, the judge model, and the full rule set, so any change to the rules produces a different tag.

`query_llm_judge` in `src/verify.py` is intentionally not implemented, so the repository never calls a model. Running `make verify` instead re-runs the prefix stage on every raw response, confirms that it reproduces the recorded `auto` labels, and confirms that the recomputed judge tag matches `judge_b4f24dea`.

## Aggregates

`make aggregate` writes three tables to `data/aggregate/`:

- `affirmation_by_model.csv`: responses, affirmations, affirmation rate, 95% Wilson confidence interval, and label counts by grading stage, per model and pooled.
- `affirmation_by_prompt.csv`: affirmations out of 50 for each model and prompt.
- `affirmation_by_repetition.csv`: affirmations out of 100 for each model and repetition.

## Figures

| Manuscript | Script                                | Output                              |
| ---------- | ------------------------------------- | ----------------------------------- |
| Table 1    | `src/fig/fig_model_table.py`          | `figures/fig_model_table.png`       |
| Figure 1   | `src/fig/fig_affirmation_rate.py`     | `figures/fig_affirmation_rate.png`  |
| Figure 2   | `src/fig/fig_heatmap_grid.py`         | `figures/fig_heatmap_grid.png`      |
| Figure 3   | `src/fig/fig_affirmed_prompts.py`     | `figures/fig_affirmed_prompts.png`  |
| Figure 4   | `src/fig/fig_response_patterns.py`    | `figures/fig_response_patterns.png` |

Each script also writes the data it plotted to `data/figure_data/` under the same name. Set `PREVIEW_MODE = True` at the top of a script to display the figure instead of saving it.

## Running

Requires Python 3.12+ and Poetry

```
make install     # install dependencies
make verify      # offline check of the grading rules against the recorded labels
make aggregate   # write data/aggregate/
make figures     # write figures/ and data/figure_data/
make all         # verify, aggregate, and figures
```
