from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Generator

from .paths import (
    THIRD_PARTY_FINETUNE_DIR,
    checkpoint_epoch,
    ensure_unique_dir,
    is_loadable_checkpoint_dir,
    list_checkpoint_paths,
    list_coded_jsonl_paths,
    list_run_paths,
    run_dir,
    sanitize_name,
    timestamp_name,
)
from .process_runner import (
    AlreadyRunningError,
    clear_process,
    is_running,
    start_process,
    stop_process,
)
from .run_registry import update_run_summary

PREPARE_KEY = "prepare_data"
TRAIN_KEY = "training"

_TRAIN_PROGRESS_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Step\s+(\d+)\s+\|\s+Loss:\s+([0-9]*\.?[0-9]+)"
)

_LOG_POLL_INTERVAL_SEC = 0.1
_LOG_YIELD_INTERVAL_SEC = 0.5


def _start_stdout_pump(proc: Any) -> Queue[str | None]:
    q: Queue[str | None] = Queue()

    def _pump() -> None:
        try:
            stream = getattr(proc, "stdout", None)
            if stream is None:
                return
            for line in stream:
                q.put(line)
        finally:
            q.put(None)

    t = threading.Thread(target=_pump, daemon=True)
    t.start()
    return q


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _process_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _resolve_prepare_device(device: str) -> str:
    selected = (device or "").strip()
    if selected and selected.lower() != "auto":
        return selected
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_training_mixed_precision(mixed_precision: str) -> str:
    v = (mixed_precision or "").strip().lower()
    device = _resolve_prepare_device("auto")
    if v and v != "auto":
        # Accelerate fp16 AMP can fail on MPS/CPU with "Attempting to unscale FP16 gradients".
        # Keep CUDA behavior, but force a safe fallback elsewhere.
        if not device.startswith("cuda") and v in {"fp16", "bf16"}:
            return "no"
        return v
    if device.startswith("cuda"):
        return "bf16"
    return "no"


def _resolve_training_torch_dtype(torch_dtype: str) -> str:
    v = (torch_dtype or "").strip().lower()
    if v and v != "auto":
        return torch_dtype
    device = _resolve_prepare_device("auto")
    if device.startswith("cuda"):
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def default_run_name() -> str:
    return timestamp_name("run")


def stop_prepare() -> str:
    return stop_process(PREPARE_KEY)


def stop_training() -> str:
    return stop_process(TRAIN_KEY)


def is_prepare_running() -> bool:
    return is_running(PREPARE_KEY)


def is_training_running() -> bool:
    return is_running(TRAIN_KEY)


def expected_raw_jsonl_for_train_jsonl(train_jsonl: str | Path) -> str | None:
    train_path = Path(train_jsonl).expanduser().resolve()
    raw_candidate = train_path.parent / "train_raw.jsonl"
    if raw_candidate.exists():
        return str(raw_candidate.resolve())
    return None


def run_prepare_data(
    *,
    device: str,
    tokenizer_model_path: str,
    input_jsonl: str,
    output_filename: str,
    batch_infer_num: int = 32,
) -> Generator[dict[str, Any], None, None]:
    if is_running(PREPARE_KEY):
        raise AlreadyRunningError("Prepare is already running. Stop it before starting a new prepare run.")

    input_path = Path(input_jsonl).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if int(batch_infer_num) <= 0:
        raise ValueError("batch_infer_num must be >= 1.")

    output_name = output_filename.strip() or "train_with_codes.jsonl"
    if not output_name.endswith(".jsonl"):
        output_name += ".jsonl"
    output_path = (input_path.parent / output_name).resolve()
    prepare_log_path = input_path.parent / "prepare_data.log"

    resolved_device = _resolve_prepare_device(device)
    command = [
        sys.executable,
        str((THIRD_PARTY_FINETUNE_DIR / "prepare_data.py").resolve()),
        "--device",
        resolved_device,
        "--tokenizer_model_path",
        tokenizer_model_path.strip(),
        "--input_jsonl",
        str(input_path),
        "--output_jsonl",
        str(output_path),
        "--batch_infer_num",
        str(int(batch_infer_num)),
    ]

    started_at = time.time()
    proc = start_process(
        PREPARE_KEY,
        command,
        cwd=THIRD_PARTY_FINETUNE_DIR,
        env=_process_env(),
    )
    out_q = _start_stdout_pump(proc)
    logs: list[str] = []
    prepare_log_path.write_text("", encoding="utf-8")
    yield {
        "status": "Preparing audio codes...",
        "logs": "",
        "output_jsonl": str(output_path),
        "prepare_log_path": str(prepare_log_path),
        "done": False,
        "success": False,
    }

    try:
        last_yield = 0.0
        while True:
            got_line = False
            while True:
                try:
                    item = out_q.get_nowait()
                except Empty:
                    break
                if item is None:
                    break
                normalized = str(item).rstrip("\n")
                if not normalized.strip():
                    continue
                got_line = True
                logs.append(normalized)
                with prepare_log_path.open("a", encoding="utf-8") as f:
                    f.write(normalized + "\n")

            now = time.time()
            if got_line or (now - last_yield) >= _LOG_YIELD_INTERVAL_SEC:
                last_yield = now
                yield {
                    "status": "Preparing audio codes...",
                    "logs": "\n".join(logs[-800:]),
                    "output_jsonl": str(output_path),
                    "prepare_log_path": str(prepare_log_path),
                    "done": False,
                    "success": False,
                }

            if proc.poll() is not None:
                # Drain remaining buffered lines, if any.
                while True:
                    try:
                        item = out_q.get_nowait()
                    except Empty:
                        break
                    if item is None:
                        break
                    normalized = str(item).rstrip("\n")
                    if not normalized.strip():
                        continue
                    logs.append(normalized)
                    with prepare_log_path.open("a", encoding="utf-8") as f:
                        f.write(normalized + "\n")
                break

            time.sleep(_LOG_POLL_INTERVAL_SEC)

        return_code = proc.wait()
        elapsed = time.time() - started_at
        success = return_code == 0
        status = (
            f"Prepare completed in {elapsed:.1f}s. Output: {output_path}"
            if success
            else f"Prepare failed (exit={return_code}). Check logs."
        )
        yield {
            "status": status,
            "logs": "\n".join(logs[-1000:]),
            "output_jsonl": str(output_path),
            "prepare_log_path": str(prepare_log_path),
            "done": True,
            "success": success,
            "coded_jsonl_choices": list_coded_jsonl_paths(),
        }
    finally:
        clear_process(PREPARE_KEY, proc)


def run_training(
    *,
    init_model_path: str,
    train_jsonl: str,
    run_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    speaker_name: str,
    speaker_id: int = 3000,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "auto",
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    subtalker_loss_weight: float = 0.3,
    attn_implementation: str = "auto",
    torch_dtype: str = "auto",
    log_every_n_steps: int = 10,
    save_every_n_epochs: int = 1,
    max_steps: int = 0,
    random_seed: int = 42,
) -> Generator[dict[str, Any], None, None]:
    if is_running(TRAIN_KEY):
        raise AlreadyRunningError("Training is already running. Stop it before starting a new run.")

    train_path = Path(train_jsonl).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Prepared training JSONL not found: {train_path}")
    final_speaker_name = speaker_name.strip()
    if not final_speaker_name:
        raise ValueError("speaker_name is required and cannot be empty.")
    if int(gradient_accumulation_steps) <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1.")
    if float(max_grad_norm) <= 0:
        raise ValueError("max_grad_norm must be > 0.")
    if int(log_every_n_steps) <= 0:
        raise ValueError("log_every_n_steps must be >= 1.")
    if int(save_every_n_epochs) <= 0:
        raise ValueError("save_every_n_epochs must be >= 1.")
    if int(speaker_id) <= 0:
        raise ValueError("speaker_id must be >= 1.")
    if int(max_steps) < 0:
        raise ValueError("max_steps must be >= 0.")
    if float(subtalker_loss_weight) < 0:
        raise ValueError("subtalker_loss_weight must be >= 0.")

    final_run_name = sanitize_name(run_name, default_run_name())
    output_dir = ensure_unique_dir(run_dir(final_run_name))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = output_dir / "train.log"
    train_log_path.write_text("", encoding="utf-8")

    resolved_mixed_precision = _resolve_training_mixed_precision(mixed_precision)
    resolved_torch_dtype = _resolve_training_torch_dtype(torch_dtype)

    config_path = output_dir / "run_config.json"
    run_config = {
        "created_at": datetime.now().isoformat(),
        "init_model_path": init_model_path,
        "train_jsonl": str(train_path),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "speaker_name": final_speaker_name,
        "speaker_id": int(speaker_id),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "mixed_precision": resolved_mixed_precision,
        "mixed_precision_input": mixed_precision,
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "subtalker_loss_weight": float(subtalker_loss_weight),
        "attn_implementation": attn_implementation,
        "torch_dtype": resolved_torch_dtype,
        "torch_dtype_input": torch_dtype,
        "log_every_n_steps": int(log_every_n_steps),
        "save_every_n_epochs": int(save_every_n_epochs),
        "max_steps": int(max_steps),
        "random_seed": int(random_seed),
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    update_run_summary(
        output_dir,
        {
            "run_name": final_run_name,
            "status": "running",
            "created_at": run_config["created_at"],
            "init_model_path": init_model_path,
            "train_jsonl": str(train_path),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "num_epochs": int(num_epochs),
            "speaker_name": final_speaker_name,
            "speaker_id": int(speaker_id),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "mixed_precision": resolved_mixed_precision,
            "mixed_precision_input": mixed_precision,
            "weight_decay": float(weight_decay),
            "max_grad_norm": float(max_grad_norm),
            "subtalker_loss_weight": float(subtalker_loss_weight),
            "attn_implementation": attn_implementation,
            "torch_dtype": resolved_torch_dtype,
            "torch_dtype_input": torch_dtype,
            "log_every_n_steps": int(log_every_n_steps),
            "save_every_n_epochs": int(save_every_n_epochs),
            "max_steps": int(max_steps),
            "random_seed": int(random_seed),
            "run_dir": str(output_dir.resolve()),
            "train_log_path": str(train_log_path.resolve()),
        },
    )

    command = [
        sys.executable,
        str((THIRD_PARTY_FINETUNE_DIR / "sft_12hz.py").resolve()),
        "--init_model_path",
        init_model_path.strip(),
        "--output_model_path",
        str(output_dir),
        "--train_jsonl",
        str(train_path),
        "--batch_size",
        str(int(batch_size)),
        "--lr",
        str(float(learning_rate)),
        "--num_epochs",
        str(int(num_epochs)),
        "--speaker_name",
        final_speaker_name,
        "--speaker_id",
        str(int(speaker_id)),
        "--gradient_accumulation_steps",
        str(int(gradient_accumulation_steps)),
        "--mixed_precision",
        resolved_mixed_precision,
        "--weight_decay",
        str(float(weight_decay)),
        "--max_grad_norm",
        str(float(max_grad_norm)),
        "--subtalker_loss_weight",
        str(float(subtalker_loss_weight)),
        "--attn_implementation",
        attn_implementation,
        "--torch_dtype",
        resolved_torch_dtype,
        "--log_every_n_steps",
        str(int(log_every_n_steps)),
        "--save_every_n_epochs",
        str(int(save_every_n_epochs)),
        "--max_steps",
        str(int(max_steps)),
        "--seed",
        str(int(random_seed)),
    ]

    samples = max(1, _line_count(train_path))
    steps_per_epoch = max(1, math.ceil(samples / max(1, int(batch_size))))

    update_run_summary(
        output_dir,
        {
            "samples": samples,
            "steps_per_epoch_estimate": steps_per_epoch,
        },
    )

    started_at = time.time()
    try:
        proc = start_process(
            TRAIN_KEY,
            command,
            cwd=THIRD_PARTY_FINETUNE_DIR,
            env=_process_env(),
        )
    except AlreadyRunningError:
        raise AlreadyRunningError("Training is already running. Stop it before starting a new run.")
    except Exception as e:
        update_run_summary(
            output_dir,
            {
                "status": "failed",
                "error": str(e),
                "process_pid": None,
                "stale_process": False,
            },
        )
        raise

    update_run_summary(
        output_dir,
        {
            "process_pid": int(proc.pid),
            "process_script": str((THIRD_PARTY_FINETUNE_DIR / "sft_12hz.py").resolve()),
            "stale_process": False,
        },
    )

    out_q = _start_stdout_pump(proc)
    logs: list[str] = []
    current_epoch = 0
    current_step = 0
    current_loss: str | None = None
    progress = "0.0%"

    yield {
        "status": f"Training started: {output_dir}",
        "progress": f"{progress} | epoch 0/{num_epochs} | step 0/{steps_per_epoch}",
        "logs": "",
        "run_dir": str(output_dir),
        "train_log_path": str(train_log_path.resolve()),
        "done": False,
        "success": False,
    }

    try:
        last_yield = 0.0
        while True:
            got_line = False
            while True:
                try:
                    item = out_q.get_nowait()
                except Empty:
                    break
                if item is None:
                    break
                line = str(item).rstrip("\n")
                if not line.strip():
                    continue
                got_line = True
                logs.append(line)
                with train_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
                match = _TRAIN_PROGRESS_RE.search(line)
                if match:
                    current_epoch = int(match.group(1)) + 1
                    current_step = int(match.group(2)) + 1
                    current_loss = match.group(3)
                    done_steps = ((current_epoch - 1) * steps_per_epoch) + min(
                        current_step, steps_per_epoch
                    )
                    total_steps = max(1, num_epochs * steps_per_epoch)
                    ratio = min(1.0, done_steps / total_steps)
                    progress = f"{ratio * 100:.1f}%"
                    update_run_summary(
                        output_dir,
                        {
                            "status": "running",
                            "epochs_done": current_epoch,
                            "last_step": current_step,
                            "last_loss": current_loss,
                            "progress": progress,
                        },
                    )

            now = time.time()
            if got_line or (now - last_yield) >= _LOG_YIELD_INTERVAL_SEC:
                last_yield = now
                detail = f"{progress} | epoch {current_epoch}/{num_epochs} | step {current_step}/{steps_per_epoch}"
                if current_loss is not None:
                    detail += f" | loss {current_loss}"

                yield {
                    "status": "Training in progress...",
                    "progress": detail,
                    "logs": "\n".join(logs[-1200:]),
                    "run_dir": str(output_dir),
                    "train_log_path": str(train_log_path.resolve()),
                    "done": False,
                    "success": False,
                }

            if proc.poll() is not None:
                # Drain remaining buffered lines, if any.
                while True:
                    try:
                        item = out_q.get_nowait()
                    except Empty:
                        break
                    if item is None:
                        break
                    line = str(item).rstrip("\n")
                    if not line.strip():
                        continue
                    logs.append(line)
                    with train_log_path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")
                break

            time.sleep(_LOG_POLL_INTERVAL_SEC)

        return_code = proc.wait()
        elapsed = time.time() - started_at
        success = return_code == 0

        checkpoints = [
            p.resolve()
            for p in output_dir.glob("checkpoint-epoch-*")
            if p.is_dir()
        ]
        checkpoints = sorted(checkpoints, key=lambda p: (checkpoint_epoch(p), p.name))
        loadable_checkpoints = [str(p) for p in checkpoints if is_loadable_checkpoint_dir(p)]
        all_checkpoints = [str(p) for p in checkpoints]
        last_checkpoint = loadable_checkpoints[-1] if loadable_checkpoints else ""

        stopped = return_code in {-15, 143, -9, 137}
        if success and not loadable_checkpoints:
            status = (
                "Training process finished, but no loadable checkpoint was found "
                "(missing `model.safetensors` / `pytorch_model*.bin`)."
            )
            final_status = "failed"
            success = False
        elif success:
            status = (
                f"Training completed in {elapsed / 60:.1f} min. "
                f"{len(loadable_checkpoints)} loadable checkpoints saved."
            )
            final_status = "completed"
        elif stopped:
            status = f"Training stopped by user (exit={return_code})."
            final_status = "stopped"
        else:
            status = f"Training failed (exit={return_code}). Check logs."
            final_status = "failed"

        update_run_summary(
            output_dir,
            {
                "status": final_status,
                "elapsed_sec": round(elapsed, 2),
                "epochs_done": current_epoch,
                "last_step": current_step,
                "last_loss": current_loss,
                "progress": "100.0%" if success else progress,
                "checkpoints": len(loadable_checkpoints),
                "checkpoints_all_dirs": len(all_checkpoints),
                "last_checkpoint": last_checkpoint,
                "all_checkpoints": all_checkpoints,
                "loadable_checkpoints": loadable_checkpoints,
                "exit_code": return_code,
                "process_pid": None,
                "process_script": str((THIRD_PARTY_FINETUNE_DIR / "sft_12hz.py").resolve()),
                "stale_process": False,
            },
        )

        yield {
            "status": status,
            "progress": "100.0%" if success else progress,
            "logs": "\n".join(logs[-1500:]),
            "run_dir": str(output_dir),
            "train_log_path": str(train_log_path.resolve()),
            "done": True,
            "success": success,
            "stopped": stopped,
            "checkpoints": loadable_checkpoints,
            "all_checkpoints": all_checkpoints,
            "last_checkpoint": last_checkpoint,
            "run_choices": list_run_paths(),
            "checkpoint_choices": list_checkpoint_paths(),
        }
    finally:
        clear_process(TRAIN_KEY, proc)
