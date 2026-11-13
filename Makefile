RUN := poetry run python3 -m

.PHONY: install verify aggregate figures all

install:
	poetry install

verify:
	$(RUN) src.verify

aggregate:
	$(RUN) src.aggregate

figures:
	$(RUN) src.fig.fig_model_table
	$(RUN) src.fig.fig_affirmation_rate
	$(RUN) src.fig.fig_heatmap_grid
	$(RUN) src.fig.fig_affirmed_prompts
	$(RUN) src.fig.fig_response_patterns

all: verify aggregate figures
