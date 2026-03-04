from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .dataset_ops import load_raw_jsonl
from .paths import dataset_dir, ensure_unique_dir, sanitize_name

SAMPLE_RATE_HZ = 24000

def _normalize_peak(wav: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(wav))) if wav.size else 0.0
    if max_abs <= 0:
        return wav
    scale = peak / max_abs
    return wav * scale


def _load_resample_mono(path: Path, target_sr: int) -> np.ndarray:
    wav, _sr = librosa.load(str(path), sr=target_sr, mono=True)
    return wav.astype(np.float32)


def normalize_dataset_audio(
    raw_jsonl_path: str,
    normalized_dataset_name: str,
    *,
    target_sr: int = SAMPLE_RATE_HZ,
    peak_normalize: bool = True,
) -> tuple[str, str]:
    rows = load_raw_jsonl(raw_jsonl_path)
    if not rows:
        raise ValueError("Input dataset is empty.")

    dataset_name = sanitize_name(normalized_dataset_name, "normalized_dataset")
    out_dataset_dir = ensure_unique_dir(dataset_dir(dataset_name))
    out_audio_dir = out_dataset_dir / "audio"
    out_ref_dir = out_dataset_dir / "reference"
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_ref_dir.mkdir(parents=True, exist_ok=True)

    ref_map: dict[str, str] = {}
    output_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=1):
        src_audio = Path(str(row.get("audio", ""))).resolve()
        if not src_audio.exists():
            raise FileNotFoundError(f"Row {idx} audio missing: {src_audio}")

        src_ref = Path(str(row.get("ref_audio", ""))).resolve()
        if not src_ref.exists():
            raise FileNotFoundError(f"Row {idx} ref_audio missing: {src_ref}")

        audio_out = out_audio_dir / f"{idx:05d}.wav"
        wav = _load_resample_mono(src_audio, target_sr=target_sr)
        if peak_normalize:
            wav = _normalize_peak(wav)
        sf.write(str(audio_out), wav, target_sr)

        ref_key = str(src_ref)
        if ref_key not in ref_map:
            ref_out = out_ref_dir / f"ref_{len(ref_map)+1:03d}.wav"
            ref_wav = _load_resample_mono(src_ref, target_sr=target_sr)
            if peak_normalize:
                ref_wav = _normalize_peak(ref_wav)
            sf.write(str(ref_out), ref_wav, target_sr)
            ref_map[ref_key] = str(ref_out.resolve())

        out_row: dict[str, Any] = {
            "audio": str(audio_out.resolve()),
            "text": str(row.get("text", "")).strip(),
            "ref_audio": ref_map[ref_key],
        }
        if row.get("language"):
            out_row["language"] = row["language"]
        output_rows.append(out_row)

    raw_out = out_dataset_dir / "train_raw.jsonl"
    with raw_out.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "source_raw_jsonl": str(Path(raw_jsonl_path).resolve()),
        "target_sample_rate": target_sr,
        "peak_normalize": peak_normalize,
        "samples": len(output_rows),
        "unique_ref_audio": len(ref_map),
    }
    with (out_dataset_dir / "normalize_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return str(out_dataset_dir.resolve()), str(raw_out.resolve())

