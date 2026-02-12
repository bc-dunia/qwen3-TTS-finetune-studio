# Product Feature Matrix

## Official Qwen3-TTS Finetuning Capability (Current)

- Single-speaker fine-tuning only
- Training input: `audio`, `text`, `ref_audio` JSONL
- Data preparation step required (`audio_codes` extraction)
- SFT checkpoint output: `checkpoint-epoch-*`
- Inference with tuned speaker name

## Implemented in This Studio

1. Data Upload & Curation
- Multi-audio upload
- Transcript import (`csv/jsonl/json`)
- Global reference audio or row-level `ref_audio`
- `train_raw.jsonl` generation
- Dataset preview and duration/text statistics

2. Data Quality & Preparation
- Quality report (`errors/warnings/sample-rate/ref consistency`)
- Preflight Go/No-Go check
- Requirements checklist (required/recommended pass/fail)
- Resource estimate (steps/runtime range/disk)
- HuggingFace model access check (cached or reachable)
- Preflight risk review (device/disk/model-path/signal health/duplicate text)
- Hyperparameter recommendation by data volume
- Audio normalization (`24k mono`, optional peak normalize)
- UI execution of official `prepare_data.py`
- Log streaming in UI
- Prepared dataset selection for training

3. Training Orchestration
- UI execution of official `sft_12hz.py`
- Advanced training options (attn_implementation / dtype / grad accum / clipping / save cadence / max_steps)
- Run-level config recording (`run_config.json`)
- Run summary registry (`run_summary.json`)
- Training logfile (`train.log`)
- Live log streaming and epoch/step/loss parsing
- Stop training action
- Automatic checkpoint discovery

4. Post-training Usage
- Single text synthesis
- Batch synthesis (one line per utterance)
- Language + instruct control for tuned voice inference
- Inference quick presets (Fast/Balanced/Quality) + auto-save to `workspace/ui_settings.json`
- Speaker preset/recent picker (for preset voices + run speaker names)
- ZIP export with `metadata.jsonl`
- Model cache reload/unload

5. Workspace Management
- Datasets/runs/checkpoints overview
- Environment dependency check
- Refresh path index from UI
- Run registry table + summary view
- Checkpoint package export (ZIP)

6. Automation Interface
- CLI subcommands: validate / preflight / precheck / normalize / prepare / train / pipeline / infer

## Extension Points

- Multi-speaker fine-tuning UI mode (when officially supported)
- Hyperparameter presets by dataset size
- Validation split + objective metrics (MCD, CER/WER proxy)
- Prompt templates for style/control evaluation
- TensorBoard charts and learning-curve panel
- Auto early-stopping and best-checkpoint selection
