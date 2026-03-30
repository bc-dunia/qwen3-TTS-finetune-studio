from __future__ import annotations

import csv
import logging
import json
import shutil
from pathlib import Path
from typing import Any

import soundfile as sf

from .paths import DATASETS_DIR, dataset_dir, ensure_unique_dir, sanitize_name

logger = logging.getLogger(__name__)


def _to_uploaded_path(item: Any) -> Path:
    if item is None:
        raise ValueError("uploaded item is None")
    if isinstance(item, str):
        return Path(item)
    if isinstance(item, dict) and "name" in item:
        return Path(str(item["name"]))
    if hasattr(item, "name"):
        return Path(str(item.name))
    raise TypeError(f"Unsupported upload type: {type(item)}")


def _read_transcript_rows(transcript_path: Path) -> list[dict[str, Any]]:
    suffix = transcript_path.suffix.lower()
    if suffix == ".csv":
        with transcript_path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows
    if suffix == ".jsonl":
        return load_raw_jsonl(transcript_path)
    if suffix == ".json":
        with transcript_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON transcript must be a list of objects.")
        return data
    raise ValueError("Transcript file must be one of .csv, .jsonl, .json")


def _resolve_audio_path(
    value: str,
    transcript_dir: Path,
    uploaded_audio_by_name: dict[str, Path],
) -> Path:
    if not value:
        raise ValueError("Empty audio path in transcript.")

    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()

    rel_candidate = (transcript_dir / value).resolve()
    if rel_candidate.exists():
        return rel_candidate

    by_name = uploaded_audio_by_name.get(candidate.name)
    if by_name and by_name.exists():
        return by_name.resolve()

    raise FileNotFoundError(
        f"Cannot resolve audio path '{value}'. "
        "Use absolute paths, transcript-relative paths, or upload matching audio files."
    )


def _copy_uploaded_files(uploaded_files: list[Any], target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    by_name: dict[str, Path] = {}
    for item in uploaded_files:
        src = _to_uploaded_path(item).resolve()
        if not src.exists():
            continue

        filename = sanitize_name(src.stem, "audio") + src.suffix.lower()
        dest = target_dir / filename
        index = 1
        while dest.exists():
            dest = target_dir / f"{sanitize_name(src.stem, 'audio')}_{index}{src.suffix.lower()}"
            index += 1
        shutil.copy2(src, dest)
        by_name[src.name] = dest
        by_name[dest.name] = dest
    return by_name


def _copy_media_into_dataset(
    source_path: Path,
    target_dir: Path,
    copied_by_source: dict[Path, Path],
    fallback_stem: str,
) -> Path:
    src = source_path.resolve()
    try:
        target_root = target_dir.resolve()
    except Exception:
        target_root = target_dir

    if src.parent == target_root or target_root in src.parents:
        copied_by_source[src] = src
        return src

    if src in copied_by_source and copied_by_source[src].exists():
        return copied_by_source[src]

    ext = src.suffix.lower() or ".wav"
    stem = sanitize_name(src.stem, fallback_stem)
    dest = target_dir / f"{stem}{ext}"
    idx = 1
    while dest.exists():
        dest = target_dir / f"{stem}_{idx}{ext}"
        idx += 1

    shutil.copy2(src, dest)
    resolved = dest.resolve()
    copied_by_source[src] = resolved
    return resolved


def _audio_duration_seconds(path: Path) -> float:
    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        return 0.0


def build_dataset_from_uploads(
    dataset_name: str,
    uploaded_audios: list[Any] | None,
    transcript_file: Any | None,
    reference_audio_file: Any | None,
) -> tuple[Path, Path]:
    if transcript_file is None:
        raise ValueError("Transcript file is required.")

    raw_dataset_name = sanitize_name(dataset_name, "dataset")
    target_dataset_dir = ensure_unique_dir(dataset_dir(raw_dataset_name))
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = target_dataset_dir / "audio"
    ref_dir = target_dataset_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)

    uploaded = uploaded_audios or []
    uploaded_audio_by_name = _copy_uploaded_files(uploaded, audio_dir)
    copied_audio_by_source: dict[Path, Path] = {}
    copied_ref_by_source: dict[Path, Path] = {}

    transcript_path = _to_uploaded_path(transcript_file).resolve()
    shutil.copy2(transcript_path, target_dataset_dir / f"source_transcript{transcript_path.suffix.lower()}")
    rows = _read_transcript_rows(transcript_path)
    if not rows:
        raise ValueError("Transcript file is empty.")

    global_ref_path: Path | None = None
    if reference_audio_file is not None:
        src_ref = _to_uploaded_path(reference_audio_file).resolve()
        global_ref_path = _copy_media_into_dataset(
            source_path=src_ref,
            target_dir=ref_dir,
            copied_by_source=copied_ref_by_source,
            fallback_stem="reference",
        )

    result_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if "audio" not in row or "text" not in row:
            raise ValueError(
                f"Transcript row {i + 1} is missing required keys. "
                "Each row must include 'audio' and 'text'."
            )

        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"Transcript row {i + 1} has empty 'text'.")

        audio_src = _resolve_audio_path(
            str(row.get("audio", "")).strip(),
            transcript_path.parent,
            uploaded_audio_by_name,
        )
        audio_path = _copy_media_into_dataset(
            source_path=audio_src,
            target_dir=audio_dir,
            copied_by_source=copied_audio_by_source,
            fallback_stem="audio",
        )

        row_ref_raw = str(row.get("ref_audio", "")).strip()
        if row_ref_raw:
            ref_src = _resolve_audio_path(
                row_ref_raw,
                transcript_path.parent,
                uploaded_audio_by_name,
            )
            ref_audio_path = _copy_media_into_dataset(
                source_path=ref_src,
                target_dir=ref_dir,
                copied_by_source=copied_ref_by_source,
                fallback_stem="reference",
            )
        elif global_ref_path is not None:
            ref_audio_path = global_ref_path
        else:
            raise ValueError(
                f"Transcript row {i + 1} has no 'ref_audio' and no global reference audio was uploaded."
            )

        out = {
            "audio": str(audio_path.resolve()),
            "text": text,
            "ref_audio": str(ref_audio_path.resolve()),
        }

        if row.get("language"):
            out["language"] = str(row["language"]).strip()

        result_rows.append(out)

    raw_jsonl = target_dataset_dir / "train_raw.jsonl"
    with raw_jsonl.open("w", encoding="utf-8") as f:
        for row in result_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return target_dataset_dir.resolve(), raw_jsonl.resolve()


def import_existing_raw_jsonl(dataset_name: str, raw_jsonl_file: Any) -> tuple[Path, Path]:
    if raw_jsonl_file is None:
        raise ValueError("Raw JSONL file is required.")

    src = _to_uploaded_path(raw_jsonl_file).resolve()
    if src.suffix.lower() != ".jsonl":
        raise ValueError("Input file must be .jsonl")

    target_name = sanitize_name(dataset_name, src.stem)
    target_dataset_dir = ensure_unique_dir(dataset_dir(target_name))
    target_dataset_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original file for traceability, but write a normalized train_raw.jsonl that
    # resolves any relative audio/ref_audio paths against the source JSONL directory.
    shutil.copy2(src, target_dataset_dir / f"source_train_raw{src.suffix.lower()}")

    base_dir = src.parent
    rows = load_raw_jsonl(src)
    if not rows:
        raise ValueError("Raw JSONL is empty.")

    def _is_url_like(value: str) -> bool:
        v = value.strip().lower()
        return "://" in v or v.startswith("data:")

    def _resolve_media_path(value: str) -> str:
        v = value.strip()
        if not v or _is_url_like(v):
            return v
        p = Path(v).expanduser()
        if p.is_absolute():
            return str(p.resolve())
        return str((base_dir / p).resolve())

    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        if "audio" not in row or "text" not in row or "ref_audio" not in row:
            raise ValueError(
                f"Raw JSONL row {i} is missing required keys. "
                "Each row must include 'audio', 'text', and 'ref_audio'."
            )
        out = dict(row)
        out["audio"] = _resolve_media_path(str(row.get("audio", "")))
        out["ref_audio"] = _resolve_media_path(str(row.get("ref_audio", "")))
        if row.get("language"):
            out["language"] = str(row["language"]).strip()
        out["text"] = str(row.get("text", "")).strip()
        normalized.append(out)

    dest = target_dataset_dir / "train_raw.jsonl"
    with dest.open("w", encoding="utf-8") as f:
        for row in normalized:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return target_dataset_dir.resolve(), dest.resolve()


def load_raw_jsonl(path: str | Path, *, strict: bool = True) -> list[dict[str, Any]]:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"JSONL not found: {p}")
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Malformed JSONL line {line_num} in {p}: {exc}"
                if strict:
                    raise ValueError(msg) from exc
                logger.warning(msg)
                continue

            if not isinstance(parsed, dict):
                msg = f"JSONL line {line_num} in {p} must be a JSON object."
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
                continue
            rows.append(parsed)
    return rows


def dataset_stats(path: str | Path) -> dict[str, Any]:
    rows = load_raw_jsonl(path)
    if not rows:
        return {
            "samples": 0,
            "total_duration_sec": 0.0,
            "avg_duration_sec": 0.0,
            "avg_text_len": 0.0,
            "missing_files": 0,
        }

    durations: list[float] = []
    text_lens: list[int] = []
    missing = 0
    for row in rows:
        audio_path = Path(str(row.get("audio", "")))
        if not audio_path.exists():
            missing += 1
            durations.append(0.0)
        else:
            durations.append(_audio_duration_seconds(audio_path))
        text_lens.append(len(str(row.get("text", ""))))

    total_duration = sum(durations)
    avg_duration = total_duration / len(durations)
    avg_text_len = sum(text_lens) / len(text_lens)

    return {
        "samples": len(rows),
        "total_duration_sec": round(total_duration, 2),
        "avg_duration_sec": round(avg_duration, 2),
        "avg_text_len": round(avg_text_len, 2),
        "min_duration_sec": round(min(durations), 2),
        "max_duration_sec": round(max(durations), 2),
        "missing_files": missing,
    }


def preview_table(path: str | Path, limit: int = 20) -> list[list[Any]]:
    rows = load_raw_jsonl(path)
    preview: list[list[Any]] = []
    for row in rows[:limit]:
        audio_path = Path(str(row.get("audio", "")))
        duration = _audio_duration_seconds(audio_path) if audio_path.exists() else 0.0
        preview.append(
            [
                str(audio_path),
                str(row.get("text", "")),
                str(row.get("ref_audio", "")),
                round(duration, 2),
            ]
        )
    return preview


def list_dataset_names() -> list[str]:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in DATASETS_DIR.iterdir() if p.is_dir()]
    names = sorted(names)
    return names
