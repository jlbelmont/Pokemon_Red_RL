PYTHON := python3
VENV_DIR := .venv
ACTIVATE := . $(VENV_DIR)/bin/activate

.PHONY: venv run run-headless simple clean

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip
	$(ACTIVATE) && pip install -r requirements.txt

run:
	$(ACTIVATE) && $(PYTHON) epsilon/pokemon_rl/minimal_epsilon_setup.py --config epsilon/pokemon_rl/training_config.json

run-headless:
	$(ACTIVATE) && $(PYTHON) epsilon/pokemon_rl/minimal_epsilon_setup.py --config epsilon/pokemon_rl/training_config.json --headless

simple:
	$(ACTIVATE) && $(PYTHON) epsilon/pokemon_rl/train.py

clean:
	rm -rf $(VENV_DIR) __pycache__ epsilon/pokemon_rl/__pycache__
