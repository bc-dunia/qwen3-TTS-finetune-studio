PYTHON ?= python3
VENV_DIR ?= .venv

.PHONY: setup ui check clean

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -U pip && pip install -r requirements.txt

ui:
	. $(VENV_DIR)/bin/activate && $(PYTHON) qwen_finetune_ui.py

check:
	$(PYTHON) -m py_compile qwen_finetune_ui.py qwen_finetune_cli.py finetune_studio/*.py

clean:
	find . -type f -name '*.pyc' -exec unlink {} \;
	find . -depth -type d -name '__pycache__' -exec rmdir {} \; 2>/dev/null || true

