# Qwen3-TTS Finetune Studio

This project is for **fine-tuning Qwen3-TTS** (single-speaker workflows), not instant one-click cloning.

If you want an instant voice clone workflow, use:
**https://github.com/bc-dunia/qwen3-TTS-studio**

Qwen3-TTS Finetune Studio provides a Gradio UI + CLI around the official Qwen finetuning scripts for:

- dataset curation and validation
- preflight safety checks
- prepare + train orchestration
- inference and generation review
- LLM-assisted iteration and Arena-based checkpoint comparison

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# launch web UI
python3 qwen_finetune_ui.py

# or CLI help
python3 qwen_finetune_cli.py --help
```

## Core Workflow

```text
raw audio + transcript + ref_audio
  -> validate / preflight
  -> prepare_data.py
  -> sft_12hz.py
  -> checkpoint review (ASR / speaker / style-speed)
  -> optional LLM optimization loop
  -> optional Arena ranking/promotion
```

## Main CLI Commands

- `validate`
- `preflight` / `precheck`
- `normalize`
- `prepare`
- `train`
- `pipeline`
- `infer`
- `review-generation`

## Documentation

- Feature matrix: [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md)
- Cloud deployment + autopilot/Arena: [`cloud/README.md`](cloud/README.md)
- Official vendored finetuning guide: [`third_party/Qwen3-TTS/finetuning/README.md`](third_party/Qwen3-TTS/finetuning/README.md)
