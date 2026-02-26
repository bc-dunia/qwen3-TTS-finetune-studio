# Qwen3-TTS Finetune Studio

A unified fine-tuning and operations toolkit for **Qwen3-TTS**, built on top of the official [`QwenLM/Qwen3-TTS` finetuning scripts](https://github.com/QwenLM/Qwen3-TTS). It provides a Gradio-based web UI and a CLI for the full lifecycle: data curation, quality validation, training orchestration, inference, and checkpoint export.

> **Note:** Only **single-speaker fine-tuning** is supported per the official specification.

---

## Table of Contents

1. [Features at a Glance](#1-features-at-a-glance)
2. [Architecture Overview](#2-architecture-overview)
3. [Code Structure](#3-code-structure)
4. [Data Contract](#4-data-contract)
5. [UI Tabs](#5-ui-tabs)
6. [CLI Reference](#6-cli-reference)
7. [Parameter Reference](#7-parameter-reference)
8. [Preflight Verdict Criteria](#8-preflight-verdict-criteria)
9. [Workspace Layout](#9-workspace-layout)
10. [Installation & Running](#10-installation--running)
11. [Operational Recommendations](#11-operational-recommendations)
12. [Current Limitations](#12-current-limitations)
13. [Scripts & Utilities](#13-scripts--utilities)
14. [E2E Smoke Test](#14-e2e-smoke-test)
15. [Additional Documentation](#15-additional-documentation)

---

## 1) Features at a Glance

| Area | Capabilities |
|---|---|
| **Data Upload & Curation** | Multi-audio upload, transcript import (CSV / JSON / JSONL), automatic path resolution, dataset preview & statistics |
| **Quality & Preparation** | Validation reports, preflight risk assessment (device / disk / model / signal quality), audio normalization (24 kHz mono, peak normalize), recommended hyperparameters by data volume |
| **Training Orchestration** | Official `prepare_data.py` and `sft_12hz.py` execution, live log streaming with epoch/step/loss parsing, safety gate (blocks training on `NO-GO`/`BLOCKED` preflight), full pipeline (prepare + train) |
| **Post-Training** | Single & batch inference, decoding parameter presets (Fast / Balanced / Similarity / Quality), default-on post-generation review (ASR / speaker cosine / speed), checkpoint export as ZIP |
| **Workspace Management** | Run registry, run summary inspection, workspace overview, environment check |
| **Automation** | Full CLI with subcommands (`validate`, `preflight`, `precheck`, `normalize`, `prepare`, `train`, `pipeline`, `infer`, `review-generation`) for scripting and CI integration |

For the complete feature matrix, see [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md).

---

## 2) Architecture Overview

```text
Raw data (audio / text / ref_audio)
  -> train_raw.jsonl
  -> validate + preflight/precheck + normalize (optional)
  -> prepare_data.py  (extract audio_codes)
  -> train_with_codes.jsonl
  -> sft_12hz.py  (supervised fine-tuning)
  -> checkpoint-epoch-*
  -> inference (single / batch) + export (ZIP)
```

**Layer breakdown:**

| Layer | Location |
|---|---|
| Web UI | `qwen_finetune_ui.py` |
| CLI | `qwen_finetune_cli.py` |
| Domain logic | `finetune_studio/*.py` |
| Official training code | `third_party/Qwen3-TTS/finetuning/*` |
| Runtime artifacts | `workspace/` |

---

## 3) Code Structure

### Entry Points

**`qwen_finetune_ui.py`** — Gradio web application
- Creates the Gradio app and binds all UI events
- 8 tabs: Dataset, Quality, Prepare, Train, Pipeline, Inference, Runs, Workspace
- Delegates to domain modules and manages UI state

**`qwen_finetune_cli.py`** — Command-line interface
- Subcommand routing for batch / server automation
- `speaker-name` is required for `train`, `pipeline`, and `infer` subcommands
- `review-generation` provides post-generation checks (ASR / speaker cosine / speaking-rate profile)

### Domain Modules (`finetune_studio/`)

| Module | Responsibility |
|---|---|
| **`paths.py`** | Project / workspace path management. Supports `QWEN_FT_WORKSPACE` env var. Provides dataset, run, and checkpoint discovery functions. |
| **`dataset_ops.py`** | Transcript parsing (CSV / JSON / JSONL), audio path resolution (absolute / relative / upload-name matching), `train_raw.jsonl` generation, import, preview, and statistics. |
| **`quality.py`** | `validate_dataset` — error/warning report. `run_preflight_review` — pre-training risk assessment: device availability (CUDA / MPS / CPU), model path validity (local / Hub), disk space estimation, signal quality sampling (SNR / clipping / silence / DC offset), text diversity and duplication warnings. `run_generation_review` — post-generation checks (ASR similarity, speaker cosine, speaking-rate fit). Report formatting and persistence. |
| **`audio_prep.py`** | Audio normalization: resample to 24 kHz mono, optional peak normalization. Produces a new `train_raw.jsonl` pointing to normalized files. |
| **`process_runner.py`** | Subprocess execution and cancellation for prepare / train. Concurrent-run key-based locking. |
| **`training_ops.py`** | `run_prepare_data` — runs official `prepare_data.py`. `run_training` — runs official `sft_12hz.py`. Parses progress (Epoch / Step / Loss). Records `run_config.json`, `train.log`, `run_summary.json`. |
| **`pipeline_ops.py`** | Orchestrates prepare + train in sequence. Yields stage-level progress events. |
| **`inference_ops.py`** | Checkpoint loading and caching. Single and batch speech generation. Batch metadata + ZIP creation. Model cache unloading. |
| **`run_registry.py`** | Updates and queries `run_summary.json`. Generates run table data. |
| **`export_ops.py`** | Checkpoint packaging (ZIP) with manifest, `run_config.json`, and `run_summary.json`. Optionally includes optimizer / scheduler / RNG state files. |
| **`ui_settings.py`** | Persists UI state to `workspace/ui_settings.json`. |

### Official Code (vendored)

| File | Purpose |
|---|---|
| `third_party/Qwen3-TTS/finetuning/prepare_data.py` | Extracts `audio_codes` from audio |
| `third_party/Qwen3-TTS/finetuning/sft_12hz.py` | Runs supervised fine-tuning (SFT) and produces checkpoints |
| `third_party/Qwen3-TTS/finetuning/dataset.py` | Dataset and collation implementation for training |

---

## 4) Data Contract

### Input JSONL

Each line must contain at minimum:

| Key | Description |
|---|---|
| `audio` | Path to the training utterance WAV file |
| `text` | Transcript text |
| `ref_audio` | Path to the speaker reference WAV file |

**Example:**

```jsonl
{"audio": "/data/utt0001.wav", "text": "Hello, how are you?", "ref_audio": "/data/ref.wav"}
{"audio": "/data/utt0002.wav", "text": "The weather is nice today.", "ref_audio": "/data/ref.wav"}
```

**Recommendations:**
- Use the **same** `ref_audio` file for all samples (single-speaker)
- Both `audio` and `ref_audio` should be **24 kHz mono** WAV
- Minimize short noisy or clipped segments

### After Prepare

- `train_with_codes.jsonl` — original fields plus an added `audio_codes` field

### Pre-Training Go / No-Go Checklist

**Mandatory** (any failure results in `NO-GO`):

- Zero dataset `ERROR`s
- Init model path is valid
- Selected device is available
- Disk free space >= estimated requirement

**Recommended** (failure results in `GO-WITH-CAUTION`):

- Total audio duration >= 10 minutes
- `ref_audio` is a single file across all samples
- Training audio is 24 kHz mono
- High text diversity (minimal duplicate transcripts)
- Low clipping / low-SNR ratio

Always run `preflight` or `precheck` before training to confirm the `GO` / `GO-WITH-CAUTION` / `NO-GO` verdict.

---

## 5) UI Tabs

### Tab 1 — Dataset
- Upload and compose datasets
- Import existing `train_raw.jsonl` files
- Preview data and view statistics

### Tab 2 — Quality & Normalize
- Quality validation report (JSON)
- Recommended hyperparameters auto-applied based on data volume
- Preflight Go / No-Go verdict (READY / CAUTION / BLOCKED + per-requirement Pass / Fail + resource estimates)
- Audio normalization (24 kHz mono + optional peak normalize)

### Tab 3 — Prepare Codes
- Execute `prepare_data.py`, stream logs, cancel mid-run
- Refreshes available output JSONL files

### Tab 4 — Train
- Execute `sft_12hz.py` with live progress and log streaming, cancel mid-run
- Automatic run and checkpoint indexing
- **Safety Gate:** blocks execution by default if preflight returned `NO-GO` / `BLOCKED` (can be overridden)

### Tab 5 — Full Pipeline
- Runs prepare + train in a single operation
- Stage-by-stage progress output
- **Safety Gate:** same behavior as the Train tab

### Tab 6 — Inference
- Single utterance and batch speech generation
- Decoding parameter controls: temperature, top-k, top-p, repetition penalty, max new tokens, subtalker parameters
- Quick Presets: **Fast** / **Balanced** / **Similarity** / **Quality** (auto-saved to `workspace/ui_settings.json`)
- Auto-fills speaker name from `run_summary.json` when a checkpoint is selected
- Post-generation review is enabled by default and saves `{generated_wav_stem}_review.json`
- When checkpoint metadata exists, review defaults are auto-filled (`ref_audio`, `train_raw.jsonl`)
- Checkpoint path accepts workspace checkpoints, local paths, or HuggingFace repo IDs
- Batch output downloadable as ZIP

### Tab 7 — Runs & Export
- Run registry table
- Run summary viewer
- Checkpoint package ZIP export

### Tab 8 — Workspace
- Artifact overview (datasets, runs, exports)
- Environment check (modules, scripts)

---

## 6) CLI Reference

```bash
# Validate dataset
python3 qwen_finetune_cli.py validate \
  --raw-jsonl /path/to/train_raw.jsonl \
  --output-report /tmp/validate.json

# Preflight risk assessment
python3 qwen_finetune_cli.py preflight \
  --raw-jsonl /path/to/train_raw.jsonl \
  --init-model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --prepare-device auto \
  --batch-size 2 --num-epochs 8 \
  --output-report /tmp/preflight.json

# Precheck (alias for preflight)
python3 qwen_finetune_cli.py precheck \
  --raw-jsonl /path/to/train_raw.jsonl \
  --init-model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --prepare-device auto \
  --batch-size 2 --num-epochs 8 \
  --output-report /tmp/precheck.json

# Normalize audio
python3 qwen_finetune_cli.py normalize \
  --raw-jsonl /path/to/train_raw.jsonl \
  --name my_norm \
  --target-sr 24000 --peak-normalize

# Prepare audio codes
python3 qwen_finetune_cli.py prepare \
  --input-jsonl /path/to/train_raw.jsonl \
  --device auto \
  --tokenizer-model-path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --output-filename train_with_codes.jsonl \
  --batch-infer-num 32

# Train
python3 qwen_finetune_cli.py train \
  --train-jsonl /path/to/train_with_codes.jsonl \
  --speaker-name my_speaker \
  --run-name run_a \
  --batch-size 2 --learning-rate 2e-5 --num-epochs 3 \
  --attn-implementation auto --max-steps 0

# Full pipeline (prepare + train)
python3 qwen_finetune_cli.py pipeline \
  --raw-jsonl /path/to/train_raw.jsonl \
  --prepare-device auto \
  --speaker-name my_speaker \
  --run-name run_pipeline \
  --batch-size 2 --learning-rate 2e-5 --num-epochs 3 \
  --prepare-batch-infer-num 32 \
  --attn-implementation auto

# Inference
python3 qwen_finetune_cli.py infer \
  --checkpoint /path/to/checkpoint-epoch-2 \
  --speaker-name my_speaker \
  --language auto \
  --instruct "calm style" \
  --text "Hello, how are you?" \
  --seed 42 \
  --device auto \
  --review-reference-audio /path/to/ref.wav \
  --review-profile-raw-jsonl /path/to/train_raw.jsonl \
  --review-output-report /tmp/infer_review.json

# Disable automatic post-generation review (if needed)
python3 qwen_finetune_cli.py infer \
  --checkpoint /path/to/checkpoint-epoch-2 \
  --speaker-name my_speaker \
  --text "Hello, how are you?" \
  --no-review-after-generation

# Post-generation review (ASR + speaker cosine + speed profile)
python3 qwen_finetune_cli.py review-generation \
  --generated-wav /path/to/output.wav \
  --target-text "안녕하세요. 오늘 시장 이야기를 간단하게 말씀드리겠습니다." \
  --reference-audio /path/to/ref.wav \
  --profile-raw-jsonl /path/to/train_raw.jsonl \
  --base-speaker-model /path/to/Qwen3-TTS-12Hz-0.6B-Base \
  --output-report /tmp/review_generation.json
```

### Exit Codes

| Command | Code | Meaning |
|---|---|---|
| `validate` | `2` | Dataset contains errors |
| `preflight` | `2` | Verdict is `NO-GO` / `BLOCKED` |
| `precheck` | `2` | Verdict is `NO-GO` / `BLOCKED` |
| `review-generation` | `1` | Quality checks returned `WARN` |
| `review-generation` | `2` | Quality checks returned `FAIL` |
| `infer` (default review ON) | `1` | Post-generation checks returned `WARN` |
| `infer` (default review ON) | `2` | Post-generation checks returned `FAIL` |
| Any command | `1` | General execution failure |

---

## 7) Parameter Reference

### Prepare Parameters

| Parameter | Description |
|---|---|
| `device` | Compute device (`auto` / `cuda:0` / `mps` / `cpu`) |
| `tokenizer_model_path` | HuggingFace repo ID or local path to tokenizer |
| `input_jsonl` / `output_jsonl` | Input and output JSONL paths |
| `batch_infer_num` | Batch size for audio code extraction |

### Training Parameters

| Parameter | Description |
|---|---|
| `init_model_path` | HuggingFace repo ID or local directory (downloads snapshot at training time if needed) |
| `train_jsonl` | Path to prepared JSONL with audio codes |
| `run_name` | Name for the training run |
| `batch_size` | Training batch size |
| `learning_rate` | Learning rate |
| `num_epochs` | Number of training epochs |
| `speaker_name` | **Required.** Speaker identifier |

<details>
<summary><b>Advanced Training Parameters</b></summary>

| Parameter | Description |
|---|---|
| `speaker_id` | Numeric speaker ID |
| `gradient_accumulation_steps` | Gradient accumulation steps |
| `mixed_precision` | Mixed precision mode |
| `torch_dtype` | Torch data type |
| `attn_implementation` | Attention implementation (`auto` / `flash_attention_2` / `sdpa` / `eager`) |
| `weight_decay` | Weight decay |
| `max_grad_norm` | Maximum gradient norm for clipping |
| `subtalker_loss_weight` | Loss weight for the subtalker |
| `log_every_n_steps` | Logging frequency (steps) |
| `save_every_n_epochs` | Checkpoint save frequency (epochs) |
| `max_steps` | Maximum training steps (useful for smoke tests) |
| `seed` | Random seed |

</details>

### Inference Parameters

| Parameter | Description |
|---|---|
| `device` | Compute device |
| `speaker_name` | Speaker identifier |
| `language` | Language code or `auto` |
| `instruct` | Style / tone control text |
| `seed` | Random seed for deterministic/reproducible inference |
| `temperature` | Sampling temperature |
| `top_k` | Top-k sampling |
| `top_p` | Top-p (nucleus) sampling |
| `repetition_penalty` | Repetition penalty |
| `max_new_tokens` | Maximum new tokens to generate |
| `subtalker_temperature` | Subtalker sampling temperature |
| `subtalker_top_k` | Subtalker top-k |
| `subtalker_top_p` | Subtalker top-p |
| `review_after_generation` | Run post-generation review after inference (default ON in UI/CLI infer) |
| `review_reference_audio` | Reference audio path for speaker cosine check |
| `review_profile_raw_jsonl` | Raw JSONL path for speaking-rate profile check |
| `review_base_speaker_model` | Base model used to extract speaker embeddings for cosine |
| `review_whisper_model` | Whisper model name for ASR similarity check |

### Modifications to Official Scripts

This project exposes additional arguments from the official scripts in a **backward-compatible** manner:

- **`prepare_data.py`**: `--batch_infer_num`
- **`sft_12hz.py`**: `--speaker_id`, `--gradient_accumulation_steps`, `--mixed_precision`, `--torch_dtype`, `--attn_implementation`, `--weight_decay`, `--max_grad_norm`, `--subtalker_loss_weight`, `--log_every_n_steps`, `--save_every_n_epochs`, `--max_steps`, `--seed`

### Fixed / Constrained Aspects

- Official implementation supports **single-speaker** only (this studio follows the same constraint)
- The training loop and model architecture are fixed in the official code (optimizer type, input embedding structure, etc.)
- Checkpoints are saved as `checkpoint-epoch-*` directories

---

## 8) Preflight Verdict Criteria

| Verdict | Meaning |
|---|---|
| **READY** | No blockers, no or minor cautions |
| **CAUTION** | Training can proceed but risks exist |
| **BLOCKED** | Must-fix issues before training |

**Checks performed:**

- Dataset errors present
- Device availability (CUDA / MPS / CPU)
- Model path validity (local or HuggingFace Hub)
- Disk free space estimation
- Signal quality (noise / clipping / silence / DC offset)
- Text diversity and duplication
- Batch / epoch overfitting risk heuristics

---

## 9) Workspace Layout

```text
workspace/
├── datasets/
│   └── <dataset_name>/
│       ├── train_raw.jsonl            # Raw training manifest
│       ├── source_transcript.*        # Original uploaded transcript
│       ├── source_train_raw.jsonl     # Original imported JSONL
│       ├── quality_report_*.json      # Validation reports
│       ├── preflight_report_*.json    # Preflight reports
│       ├── prepare_data.log           # Prepare stage log
│       ├── train_with_codes*.jsonl    # Prepared JSONL with audio codes
│       └── normalize_meta.json        # Normalization metadata
├── runs/
│   └── <run_name>/
│       ├── run_config.json            # Training configuration
│       ├── run_summary.json           # Run results summary
│       ├── train.log                  # Training log
│       └── checkpoint-epoch-*/        # Model checkpoints
└── exports/
    ├── single/single_*.wav            # Single inference outputs
    ├── batch_*/wav/*.wav              # Batch inference outputs
    ├── batch_*.zip                    # Batch output archives
    └── checkpoint_package_*/<ckpt>.zip  # Checkpoint packages
```

---

## 10) Installation & Running

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is strongly recommended for training

### Quick Start

```bash
cd /path/to/qwen3-tts-finetune-studio

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Launch the web UI
python3 qwen_finetune_ui.py
```

### Using Make

```bash
make setup    # Create venv + install dependencies
make ui       # Launch the Gradio UI
make check    # Validate Python syntax for all source files
make clean    # Remove .pyc and __pycache__
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QWEN_FT_WORKSPACE` | `./workspace` | Root directory for all artifacts |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio server bind address |
| `GRADIO_SERVER_PORT` | `7861` | Gradio server port |

---

## 11) Operational Recommendations

- **Use a CUDA environment** whenever possible for training
- **Reference audio** should be a single high-quality clip of 3-10 seconds
- **Re-run validate / preflight** after normalization to confirm improvements
- **Monitor for overfitting** on small datasets by listening to intermediate checkpoints
- **Keep 30%+ free disk space** beyond the estimated requirement
- For best results, ensure training audio is clean with minimal background noise

---

## 12) Current Limitations

- Multi-speaker fine-tuning is **not supported** (official spec limitation)
- Training the 1.7B base model **requires a GPU** — CPU and MPS environments may fail or be extremely slow due to memory and compute constraints
- Objective evaluation metrics (CER / WER / MCD) are **not yet implemented**
- TensorBoard visualization panel is **not yet implemented**
- Early stopping and automatic best-checkpoint selection are **not yet implemented**

---

## 13) Scripts & Utilities

The `scripts/` directory contains standalone utilities for voice quality evaluation and experimentation:

| Script | Purpose |
|---|---|
| `e2e_smoke.py` | End-to-end pipeline smoke test (synthetic data → train → infer) |
| `build_and_test_voice.py` | Full voice-building automation: train + grid-search decoding params + quality evaluation |
| `boost_korean_similarity.py` | Hyperparameter sweep focused on maximizing Korean ASR similarity |
| `boost_style_speed.py` | Hyperparameter sweep focused on speaking style and speed matching |
| `review_65min_v1.py` | Multi-checkpoint quality review (speaker cosine, ASR accuracy, signal quality) |
| `test_icl_voice_clone.py` | In-Context Learning (ICL) voice cloning test — no fine-tuning required |
| `test_icl_vs_finetune.py` | Side-by-side comparison of ICL vs fine-tuned voice quality |
| `test_better_ref.py` | Reference audio selection testing with sentence-boundary candidates |

Most scripts require `faster-whisper` and a GPU for practical use. They output results to `workspace/exports/`.

---

## 14) E2E Smoke Test

An end-to-end smoke test script is provided to quickly verify that the entire pipeline works correctly.

```bash
cd /path/to/qwen3-tts-finetune-studio
.venv/bin/python scripts/e2e_smoke.py
```

**What it does:**
- Generates a small synthetic (sine wave) audio dataset
- Runs the full pipeline: dataset creation -> validate -> preflight -> prepare -> train (`max_steps=1`) -> inference
- Default learning rate is `0.0` (stability check, not quality evaluation)
- Cleans up generated artifacts by default; set `KEEP_E2E_ARTIFACTS=1` to preserve them

### E2E Environment Variables

**Model paths (local-first):**

| Variable | Description |
|---|---|
| `QWEN3_TTS_MODELS_ROOT` | Local model root directory (e.g., `.../qwen3-tts-studio/qwen3-TTS-studio`) |
| `E2E_INIT_MODEL_PATH` | Init model local path or HuggingFace repo ID (default: 0.6B Base) |
| `E2E_TOKENIZER_MODEL_PATH` | Tokenizer local path or HuggingFace repo ID |
| `E2E_ALLOW_HF_DOWNLOAD=1` | Allow HuggingFace download when local path is not found |

**Execution control:**

| Variable | Default | Description |
|---|---|---|
| `E2E_PREPARE_DEVICE` | `auto` | Device for prepare stage (`auto` / `cuda:0` / `mps` / `cpu`) |
| `E2E_INFER_DEVICE` | CUDA if available, else `cpu` | Device for inference stage |
| `E2E_LR` | `0.0` | Learning rate for the smoke test |
| `KEEP_E2E_ARTIFACTS` | `0` | Set to `1` to skip cleanup |

---

## 15) Additional Documentation

- **Feature Matrix:** [`FEATURE_MATRIX.md`](FEATURE_MATRIX.md)
- **Official Fine-tuning Guide (vendored):** [`third_party/Qwen3-TTS/finetuning/README.md`](third_party/Qwen3-TTS/finetuning/README.md)
