from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

from .training_ops import run_prepare_data, run_training


def run_full_pipeline(
    *,
    raw_jsonl_path: str,
    prepare_device: str,
    tokenizer_model_path: str,
    prepare_output_filename: str,
    prepare_batch_infer_num: int = 32,
    init_model_path: str,
    run_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    speaker_name: str,
    speaker_id: int = 3000,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "bf16",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    subtalker_loss_weight: float = 0.3,
    attn_implementation: str = "auto",
    torch_dtype: str = "bfloat16",
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 1,
    max_steps: int = 0,
    random_seed: int = 42,
) -> Generator[dict[str, Any], None, None]:
    prepared_jsonl = ""
    prepare_done = False
    prepare_success = False

    for event in run_prepare_data(
        device=prepare_device,
        tokenizer_model_path=tokenizer_model_path,
        input_jsonl=raw_jsonl_path,
        output_filename=prepare_output_filename,
        batch_infer_num=prepare_batch_infer_num,
    ):
        prepared_jsonl = event.get("output_jsonl", prepared_jsonl)
        if event.get("done"):
            prepare_done = True
            prepare_success = bool(event.get("success"))
        event = dict(event)
        event["stage"] = "prepare"
        yield event

    if prepare_done and not prepare_success:
        yield {
            "stage": "pipeline",
            "done": True,
            "success": False,
            "status": "Pipeline stopped: prepare stage failed.",
        }
        return

    if not prepared_jsonl or not Path(prepared_jsonl).exists():
        yield {
            "stage": "pipeline",
            "done": True,
            "success": False,
            "status": "Pipeline failed: prepare stage did not produce output JSONL.",
        }
        return

    for event in run_training(
        init_model_path=init_model_path,
        train_jsonl=prepared_jsonl,
        run_name=run_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        speaker_name=speaker_name,
        speaker_id=speaker_id,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        subtalker_loss_weight=subtalker_loss_weight,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        log_every_n_steps=log_every_n_steps,
        save_every_n_epochs=save_every_n_epochs,
        max_steps=max_steps,
        random_seed=random_seed,
    ):
        event = dict(event)
        event["stage"] = "train"
        yield event
