from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.getenv("QWEN_FT_WORKSPACE", str(PROJECT_ROOT / "workspace"))
).expanduser()

DATASETS_DIR = WORKSPACE_ROOT / "datasets"
RUNS_DIR = WORKSPACE_ROOT / "runs"
EXPORTS_DIR = WORKSPACE_ROOT / "exports"

THIRD_PARTY_FINETUNE_DIR = (
    PROJECT_ROOT / "third_party" / "Qwen3-TTS" / "finetuning"
).resolve()

_CHECKPOINT_EPOCH_RE = re.compile(r"checkpoint-epoch-(\d+)$")


def ensure_workspace_dirs() -> None:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


def sanitize_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (name or "").strip())
    cleaned = cleaned.strip("-_")
    return cleaned or fallback


def timestamp_name(prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}"


def ensure_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base_dir.parent / f"{base_dir.name}_{suffix}"
    counter = 1
    while candidate.exists():
        candidate = base_dir.parent / f"{base_dir.name}_{suffix}_{counter}"
        counter += 1
    return candidate


def dataset_dir(dataset_name: str) -> Path:
    ensure_workspace_dirs()
    return DATASETS_DIR / sanitize_name(dataset_name, "dataset")


def run_dir(run_name: str) -> Path:
    ensure_workspace_dirs()
    return RUNS_DIR / sanitize_name(run_name, "run")


def list_raw_jsonl_paths() -> list[str]:
    ensure_workspace_dirs()
    paths = sorted(DATASETS_DIR.glob("*/train_raw.jsonl"))
    return [str(p.resolve()) for p in paths]


def _jsonl_has_audio_codes(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                return isinstance(obj, dict) and "audio_codes" in obj
    except Exception:
        return False
    return False


def list_coded_jsonl_paths() -> list[str]:
    ensure_workspace_dirs()
    candidates: list[Path] = []
    for dataset in DATASETS_DIR.iterdir():
        if not dataset.is_dir():
            continue
        candidates.extend([p for p in dataset.glob("*.jsonl") if p.is_file()])

    coded = [p for p in candidates if _jsonl_has_audio_codes(p)]
    coded = sorted(coded, key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in coded]


def list_run_paths() -> list[str]:
    ensure_workspace_dirs()
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir()]
    runs = sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.resolve()) for p in runs]


def checkpoint_epoch(path: str | Path) -> int:
    p = Path(path)
    m = _CHECKPOINT_EPOCH_RE.search(p.name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _is_valid_safetensors_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as f:
            _ = f.keys()
        return True
    except Exception:
        return False


def is_loadable_checkpoint_dir(path: str | Path) -> bool:
    p = Path(path)
    if not p.exists() or not p.is_dir():
        return False

    safe_file = p / "model.safetensors"
    if safe_file.exists() and _is_valid_safetensors_file(safe_file):
        return True

    if (p / "pytorch_model.bin").exists():
        return True
    if (p / "model.safetensors.index.json").exists():
        return True
    if (p / "pytorch_model.bin.index.json").exists():
        return True

    # Sharded checkpoints can be saved without the single-file names above.
    safe_shards = [x for x in p.glob("model-*.safetensors") if x.is_file()]
    if safe_shards and all(_is_valid_safetensors_file(x) for x in safe_shards):
        return True
    if any(p.glob("pytorch_model-*.bin")):
        return True
    return False


def sort_checkpoint_paths(paths: list[Path]) -> list[Path]:
    def _key(p: Path) -> tuple[float, int, str]:
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (mtime, checkpoint_epoch(p), p.name)

    return sorted(paths, key=_key, reverse=True)


def list_checkpoint_paths() -> list[str]:
    ensure_workspace_dirs()
    checkpoints = [p for p in RUNS_DIR.glob("**/checkpoint-epoch-*") if p.is_dir()]
    checkpoints = sort_checkpoint_paths(checkpoints)
    # Only expose loadable checkpoints by default to avoid selecting directories
    # that cannot be loaded for inference/export.
    checkpoints = [p for p in checkpoints if is_loadable_checkpoint_dir(p)]
    return [str(p.resolve()) for p in checkpoints]
