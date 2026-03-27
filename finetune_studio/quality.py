from __future__ import annotations

import json
import importlib.util
import math
import re
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .dataset_ops import load_raw_jsonl


@dataclass
class DatasetIssue:
    severity: str
    code: str
    message: str
    row_index: int | None = None

SAMPLE_RATE_HZ = 24000

def _safe_audio_info(path: Path) -> tuple[int | None, float]:
    try:
        info = sf.info(str(path))
        return int(info.samplerate), float(info.duration)
    except Exception:
        return None, 0.0


def _safe_signal_metrics(path: Path, max_seconds: float = 20.0) -> dict[str, float]:
    """Read a prefix of audio and compute lightweight signal-health metrics."""
    try:
        info = sf.info(str(path))
        sr = int(info.samplerate)
        if sr <= 0:
            return {}

        max_frames = int(sr * max_seconds)
        frames = min(int(info.frames), max_frames) if info.frames > 0 else max_frames
        if frames <= 0:
            return {}

        audio, _ = sf.read(str(path), frames=frames, dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            return {}

        peak = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        dc_offset = float(abs(np.mean(audio)))
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.999))
        silent_ratio = float(np.mean(np.abs(audio) < 1e-4))

        frame_size = max(1, int(sr * 0.02))
        n_frames = len(audio) // frame_size
        if n_frames >= 4:
            frame_rms = np.array(
                [
                    float(
                        np.sqrt(
                            np.mean(audio[i * frame_size : (i + 1) * frame_size] ** 2)
                        )
                    )
                    for i in range(n_frames)
                ]
            )
            noise_floor = float(np.percentile(frame_rms, 10))
            if noise_floor > 1e-8 and rms > 1e-8:
                snr = float(max(0.0, min(60.0, 20 * np.log10(rms / noise_floor))))
            else:
                snr = 40.0
        else:
            snr = 20.0

        return {
            "peak": peak,
            "rms": rms,
            "dc_offset": dc_offset,
            "clipping_ratio": clipping_ratio,
            "silent_ratio": silent_ratio,
            "snr": snr,
        }
    except Exception:
        return {}


def _sample_rows(rows: list[dict[str, Any]], max_items: int = 200) -> list[dict[str, Any]]:
    if len(rows) <= max_items:
        return rows
    step = len(rows) / max_items
    picked = [rows[min(len(rows) - 1, int(i * step))] for i in range(max_items)]
    return picked


def _check_device(requested_device: str) -> dict[str, Any]:
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        selected = requested_device.strip().lower() if requested_device else "auto"
        if selected == "auto":
            if cuda_ok:
                selected = "cuda:0"
            elif mps_ok:
                selected = "mps"
            else:
                selected = "cpu"

        ok = True
        reason = ""
        if selected.startswith("cuda") and not cuda_ok:
            ok = False
            reason = "CUDA requested but not available."
        if selected == "mps" and not mps_ok:
            ok = False
            reason = "MPS requested but not available."

        vram_gb = None
        if cuda_ok:
            props = torch.cuda.get_device_properties(0)
            vram_gb = round(float(props.total_memory) / (1024**3), 2)

        return {
            "ok": ok,
            "selected": selected,
            "cuda_available": cuda_ok,
            "mps_available": mps_ok,
            "cuda_vram_gb": vram_gb,
            "reason": reason,
        }
    except Exception as e:
        return {
            "ok": False,
            "selected": requested_device or "auto",
            "cuda_available": False,
            "mps_available": False,
            "cuda_vram_gb": None,
            "reason": f"torch check failed: {e}",
        }


def _check_disk(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(str(path))
    free_gb = round(usage.free / (1024**3), 2)
    total_gb = round(usage.total / (1024**3), 2)
    used_gb = round((usage.total - usage.free) / (1024**3), 2)
    return {
        "path": str(path),
        "free_gb": free_gb,
        "used_gb": used_gb,
        "total_gb": total_gb,
    }


def _check_model_path(init_model_path: str) -> dict[str, Any]:
    p = Path(init_model_path).expanduser()
    is_local_style = init_model_path.startswith(("/", "./", "../", "~"))
    exists = p.exists()
    if exists:
        return {"ok": True, "mode": "local", "path": str(p.resolve()), "exists": True}
    if is_local_style:
        return {"ok": False, "mode": "local", "path": str(p), "exists": False}
    return {
        "ok": True,
        "mode": "hub_id",
        "path": init_model_path,
        "exists": None,
    }


def _check_python_module(module_name: str) -> dict[str, Any]:
    try:
        ok = importlib.util.find_spec(module_name) is not None
        return {"ok": bool(ok), "module": module_name}
    except Exception as e:
        return {"ok": False, "module": module_name, "reason": str(e)}


def _check_binary(binary_name: str) -> dict[str, Any]:
    path = shutil.which(binary_name)
    return {"ok": bool(path), "binary": binary_name, "path": path}


def _check_hf_model_access(repo_id: str) -> dict[str, Any]:
    """Check whether a HuggingFace model repo id is reachable or at least cached locally."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        return {"ok": False, "repo_id": repo_id, "reason": f"huggingface_hub not available: {e}"}

    # First, check local cache without network.
    try:
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename="config.json",
            local_files_only=True,
            token=False,
        )
        return {"ok": True, "repo_id": repo_id, "cached": True, "config_path": config_path}
    except Exception:
        pass

    # Then, check remote reachability.
    try:
        api = HfApi()
        try:
            info = api.model_info(repo_id)
            used_token = True
        except Exception:
            info = api.model_info(repo_id, token=False)
            used_token = False
        return {
            "ok": True,
            "repo_id": repo_id,
            "cached": False,
            "sha": getattr(info, "sha", None),
            "private": getattr(info, "private", None),
            "used_token": used_token,
        }
    except Exception as e:
        return {"ok": False, "repo_id": repo_id, "reason": str(e)}


def _estimate_required_disk_gb(
    total_audio_sec: float,
    model_size_hint: str,
    num_epochs: int,
) -> float:
    # Rough estimate: checkpoints + logs + temp. We keep conservative headroom.
    if "1.7b" in model_size_hint.lower():
        base_ckpt_gb = 4.5
    elif "0.6b" in model_size_hint.lower():
        base_ckpt_gb = 2.6
    else:
        base_ckpt_gb = 4.0

    checkpoint_count = max(1, num_epochs)
    checkpoints_gb = base_ckpt_gb * checkpoint_count

    data_gb = (total_audio_sec * SAMPLE_RATE_HZ * 2) / (1024**3)  # mono 16-bit WAV rough
    # data_gb is usually small; include preprocessing/export overhead.
    overhead_gb = max(2.0, data_gb * 2.0)
    return round(checkpoints_gb + overhead_gb, 2)


def _requirement_check(
    *,
    code: str,
    level: str,
    requirement: str,
    passed: bool,
    current: str,
    target: str,
    impact: str,
    action: str,
) -> dict[str, str]:
    status = "pass" if passed else ("fail" if level == "required" else "warn")
    return {
        "code": code,
        "level": level,
        "status": status,
        "requirement": requirement,
        "current": current,
        "target": target,
        "impact": impact,
        "action": action,
    }


def _estimate_runtime_range_hours(total_steps: int, selected_device: str) -> tuple[float, float]:
    if total_steps <= 0:
        return 0.0, 0.0
    d = (selected_device or "").strip().lower()
    step_sec = 3.0
    if d.startswith("cuda"):
        step_sec = 0.45
    elif d == "mps":
        step_sec = 0.95
    base_hours = (float(total_steps) * step_sec) / 3600.0
    min_hours = max(0.01, round(base_hours * 0.7, 2))
    max_hours = max(0.01, round(base_hours * 1.8, 2))
    return min_hours, max_hours


def run_preflight_review(
    raw_jsonl_path: str | Path,
    *,
    init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    prepare_device: str = "auto",
    batch_size: int = 2,
    num_epochs: int = 3,
) -> dict[str, Any]:
    dataset_report = validate_dataset(raw_jsonl_path)
    rows = load_raw_jsonl(raw_jsonl_path)
    sampled = _sample_rows(rows, max_items=200)

    analyzed = 0
    clip_count = 0
    high_silence_count = 0
    low_snr_count = 0
    high_dc_count = 0
    snr_values: list[float] = []
    clip_values: list[float] = []
    silence_values: list[float] = []

    text_counts: dict[str, int] = {}
    for row in rows:
        t = str(row.get("text", "")).strip()
        if t:
            text_counts[t] = text_counts.get(t, 0) + 1

    for row in sampled:
        audio_path = Path(str(row.get("audio", ""))).expanduser()
        if not audio_path.exists():
            continue
        metrics = _safe_signal_metrics(audio_path)
        if not metrics:
            continue
        analyzed += 1
        snr_values.append(metrics.get("snr", 0.0))
        clip_values.append(metrics.get("clipping_ratio", 0.0))
        silence_values.append(metrics.get("silent_ratio", 0.0))

        if metrics.get("clipping_ratio", 0.0) > 0.01:
            clip_count += 1
        if metrics.get("silent_ratio", 0.0) > 0.6:
            high_silence_count += 1
        if metrics.get("snr", 20.0) < 12.0:
            low_snr_count += 1
        if metrics.get("dc_offset", 0.0) > 0.02:
            high_dc_count += 1

    unique_text_ratio = (
        round(len(text_counts) / len(rows), 4) if rows else 0.0
    )

    device_check = _check_device(prepare_device)
    disk_check = _check_disk(Path(raw_jsonl_path).resolve().parent)
    model_check = _check_model_path(init_model_path)
    hf_model_check = (
        _check_hf_model_access(init_model_path)
        if str(model_check.get("mode", "")) == "hub_id"
        else {"ok": True, "repo_id": init_model_path, "skipped": True}
    )
    flash_attn_check = _check_python_module("flash_attn")
    sox_check = _check_binary("sox")

    summary = dataset_report.get("summary", {})
    issue_codes = {
        str(item.get("code", "")).strip().upper()
        for item in dataset_report.get("issues", [])
        if isinstance(item, dict)
    }
    total_audio_sec = float(summary.get("total_duration_sec", 0.0))
    total_audio_min = float(summary.get("total_duration_min", 0.0))
    samples = int(summary.get("samples", 0))
    unique_ref_count = int(summary.get("unique_ref_audio_count", 0))
    required_disk_gb = _estimate_required_disk_gb(
        total_audio_sec=total_audio_sec,
        model_size_hint=init_model_path,
        num_epochs=int(max(1, num_epochs)),
    )

    blockers: list[str] = []
    cautions: list[str] = []
    recommendations: list[str] = []

    if summary.get("errors", 0) > 0:
        blockers.append("Dataset has blocking errors. Resolve all ERROR items before training.")

    if not device_check.get("ok", False):
        blockers.append(f"Requested device is unavailable: {device_check.get('reason', 'device check failed')}")

    if not model_check.get("ok", False):
        blockers.append(f"Init model path is invalid: {model_check.get('path', '')}")
    elif not hf_model_check.get("ok", True):
        blockers.append(
            "Init model repo is not accessible (not cached and cannot reach hub). "
            f"repo_id={hf_model_check.get('repo_id', '')} reason={hf_model_check.get('reason', '')}"
        )

    if not flash_attn_check.get("ok", False):
        cautions.append(
            "flash_attn is not installed. If you use `attn_implementation=flash_attention_2`, "
            "training can fail. Consider installing flash-attn or choose `sdpa/eager`."
        )

    if disk_check.get("free_gb", 0.0) < required_disk_gb:
        blockers.append(
            f"Insufficient disk: free {disk_check.get('free_gb')}GB < estimated required {required_disk_gb}GB."
        )
    elif disk_check.get("free_gb", 0.0) < required_disk_gb * 1.3:
        cautions.append(
            f"Low disk headroom: free {disk_check.get('free_gb')}GB, estimated need {required_disk_gb}GB."
        )

    if summary.get("total_duration_min", 0.0) < 3.0:
        cautions.append("Very small dataset (<3 min). Expect unstable identity and prosody.")
    elif summary.get("total_duration_min", 0.0) < 10.0:
        cautions.append("Small dataset (<10 min). Overfitting risk is high; monitor checkpoints closely.")

    if summary.get("unique_ref_audio_count", 0) > 1:
        cautions.append("Multiple reference audios detected. Prefer one consistent high-quality ref_audio.")

    if not sox_check.get("ok", False):
        cautions.append("SoX binary not found. Some audio ops may warn or fail depending on your environment.")

    if analyzed > 0:
        clip_ratio = clip_count / analyzed
        low_snr_ratio = low_snr_count / analyzed
        high_silence_ratio = high_silence_count / analyzed
        if clip_ratio >= 0.1:
            cautions.append(f"Clipping risk: {clip_count}/{analyzed} sampled files show clipping >1%.")
        if low_snr_ratio >= 0.15:
            cautions.append(f"Noise risk: {low_snr_count}/{analyzed} sampled files have low SNR (<12dB).")
        if high_silence_ratio >= 0.2:
            cautions.append(f"Silence-heavy data: {high_silence_count}/{analyzed} sampled files have >60% silence.")
        if high_dc_count > 0:
            cautions.append(f"DC offset detected in {high_dc_count}/{analyzed} sampled files.")

    if unique_text_ratio < 0.85:
        cautions.append(
            f"Transcript diversity is low (unique ratio {unique_text_ratio}). Duplicate text may hurt generalization."
        )

    # Training-parameter caution heuristics
    if samples > 0 and batch_size > samples:
        cautions.append(f"Batch size ({batch_size}) is larger than sample count ({samples}).")
    if num_epochs > 20 and summary.get("total_duration_min", 0.0) < 30:
        cautions.append("High epoch count for small data can overfit. Consider reducing epochs.")

    recommendations.append("Run Normalize step if any NON_24K or REF_NON_24K issues exist.")
    recommendations.append("Use a single clean 3-10s reference audio shared across samples.")
    recommendations.append("Start from recommended LR/epochs, then compare checkpoints by listening tests.")
    recommendations.append("Keep at least 30% disk headroom beyond estimated checkpoint footprint.")

    steps_per_epoch = max(1, math.ceil(samples / max(1, int(batch_size)))) if samples > 0 else 0
    total_steps = steps_per_epoch * int(max(1, num_epochs)) if samples > 0 else 0
    runtime_min_h, runtime_max_h = _estimate_runtime_range_hours(
        total_steps=total_steps,
        selected_device=str(device_check.get("selected", "")),
    )

    clip_ratio = (clip_count / analyzed) if analyzed > 0 else 0.0
    low_snr_ratio = (low_snr_count / analyzed) if analyzed > 0 else 0.0

    requirements = [
        _requirement_check(
            code="REQ_SCHEMA",
            level="required",
            requirement="No blocking dataset errors (missing files/empty text/ref 24k).",
            passed=int(summary.get("errors", 0)) == 0,
            current=f"errors={summary.get('errors', 0)}",
            target="errors=0",
            impact="Training can fail immediately or produce unusable checkpoints.",
            action="Run Quality Validation and fix all ERROR items first.",
        ),
        _requirement_check(
            code="REQ_MODEL_PATH",
            level="required",
            requirement="Init model path must be valid (local path or HuggingFace model id).",
            passed=bool(model_check.get("ok", False)),
            current=str(model_check.get("path", "unknown")),
            target="resolvable model path",
            impact="Model load failure at training start.",
            action="Fix `init_model_path` before prepare/train.",
        ),
        _requirement_check(
            code="REQ_MODEL_ACCESS",
            level="required",
            requirement="Init model must be accessible (local exists or hub reachable/cached).",
            passed=bool(model_check.get("ok", False)) and bool(hf_model_check.get("ok", True)),
            current=(
                str(model_check.get("path", "unknown"))
                if str(model_check.get("mode", "")) == "local"
                else f"hub:{init_model_path} cached={hf_model_check.get('cached', False)}"
            ),
            target="accessible model source",
            impact="Training will fail after spending time on setup if the model cannot be downloaded.",
            action="Ensure network access to HuggingFace, login with a token for private/gated repos, or use a local model path.",
        ),
        _requirement_check(
            code="REQ_DEVICE",
            level="required",
            requirement="Requested prepare device must be available.",
            passed=bool(device_check.get("ok", False)),
            current=str(device_check.get("selected", "unknown")),
            target="available cuda/mps/cpu device",
            impact="Prepare stage fails before audio code extraction.",
            action="Set device to `auto` or an actually available device.",
        ),
        _requirement_check(
            code="REQ_DISK",
            level="required",
            requirement="Free disk must cover estimated checkpoint/output footprint.",
            passed=float(disk_check.get("free_gb", 0.0)) >= float(required_disk_gb),
            current=f"free={disk_check.get('free_gb', 0.0)}GB",
            target=f">={required_disk_gb}GB",
            impact="Training can stop mid-run due to disk exhaustion.",
            action="Free disk or lower epochs/model size before training.",
        ),
        _requirement_check(
            code="REC_DURATION",
            level="recommended",
            requirement="Prefer at least 10 minutes of clean target speech.",
            passed=total_audio_min >= 10.0,
            current=f"{total_audio_min:.2f} min",
            target=">= 10.00 min",
            impact="Too little data increases overfitting and unstable speaker identity.",
            action="Add more clean speech data or reduce learning rate/epochs.",
        ),
        _requirement_check(
            code="REC_SINGLE_REF",
            level="recommended",
            requirement="Use one consistent reference audio across samples.",
            passed=unique_ref_count <= 1,
            current=f"unique_ref_audio={unique_ref_count}",
            target="1",
            impact="Multiple refs often reduce timbre consistency.",
            action="Unify `ref_audio` to a single clean reference clip.",
        ),
        _requirement_check(
            code="REC_AUDIO_24K",
            level="recommended",
            requirement="All train audio should be 24kHz mono.",
            passed="NON_24K_SAMPLE_RATE" not in issue_codes,
            current=f"sample_rates={summary.get('sample_rates', {})}",
            target=f"{{{SAMPLE_RATE_HZ}: N}}",
            impact="Resampling at runtime may lower consistency and quality.",
            action="Run Normalize step before prepare/train.",
        ),
        _requirement_check(
            code="REC_TEXT_DIVERSITY",
            level="recommended",
            requirement="Transcript diversity should be high.",
            passed=unique_text_ratio >= 0.9,
            current=f"unique_text_ratio={unique_text_ratio}",
            target=">= 0.9",
            impact="Low diversity hurts generalization to unseen sentences.",
            action="Remove duplicate text and expand script variety.",
        ),
        _requirement_check(
            code="REC_SIGNAL",
            level="recommended",
            requirement="Signal quality should avoid heavy clipping/noise.",
            passed=(analyzed == 0) or (clip_ratio < 0.1 and low_snr_ratio < 0.15),
            current=(
                f"clip_ratio={clip_ratio:.3f}, low_snr_ratio={low_snr_ratio:.3f}"
                if analyzed > 0
                else "not analyzed"
            ),
            target="clip_ratio<0.1 and low_snr_ratio<0.15",
            impact="Noisy/clipped data degrades naturalness and intelligibility.",
            action="Denoise/re-record bad clips and rerun validation.",
        ),
    ]

    hard_failures = [item for item in requirements if item.get("status") == "fail"]
    soft_warnings = [item for item in requirements if item.get("status") == "warn"]

    status = "ready"
    if hard_failures or blockers:
        status = "blocked"
    elif soft_warnings or cautions:
        status = "caution"

    decision = "go"
    if status == "blocked":
        decision = "no-go"
    elif status == "caution":
        decision = "go-with-caution"

    next_actions: list[str] = []
    for item in hard_failures:
        next_actions.append(f"[required] {item.get('action', '')}")
    if not hard_failures:
        for item in soft_warnings[:3]:
            next_actions.append(f"[recommended] {item.get('action', '')}")
    if not next_actions:
        next_actions.append("No blocking requirement failed. Proceed to prepare/train.")

    signal = {
        "analyzed_files": analyzed,
        "sampled_files": len(sampled),
        "clip_count": clip_count,
        "high_silence_count": high_silence_count,
        "low_snr_count": low_snr_count,
        "high_dc_count": high_dc_count,
        "avg_snr_db": round(float(np.mean(snr_values)), 2) if snr_values else None,
        "avg_clip_ratio": round(float(np.mean(clip_values)), 4) if clip_values else None,
        "avg_silence_ratio": round(float(np.mean(silence_values)), 4) if silence_values else None,
    }

    score = 100
    score -= min(60, len(blockers) * 25)
    score -= min(35, len(cautions) * 5)
    score = max(0, int(score))

    return {
        "status": status,
        "decision": decision,
        "score": score,
        "dataset": dataset_report,
        "signal": signal,
        "requirements": requirements,
        "resource_estimate": {
            "samples": samples,
            "batch_size": int(batch_size),
            "num_epochs": int(num_epochs),
            "steps_per_epoch_estimate": int(steps_per_epoch),
            "total_steps_estimate": int(total_steps),
            "selected_device": str(device_check.get("selected", "unknown")),
            "estimated_runtime_range_hours": [runtime_min_h, runtime_max_h],
            "estimated_required_disk_gb": float(required_disk_gb),
            "free_disk_gb": float(disk_check.get("free_gb", 0.0)),
        },
        "environment": {
            "device": device_check,
            "disk": disk_check,
            "model_path": model_check,
            "tools": {
                "hf_model": hf_model_check,
                "flash_attn": flash_attn_check,
                "sox": sox_check,
            },
            "estimated_required_disk_gb": required_disk_gb,
            "inputs": {
                "raw_jsonl_path": str(Path(raw_jsonl_path).resolve()),
                "prepare_device": prepare_device,
                "init_model_path": init_model_path,
                "batch_size": int(batch_size),
                "num_epochs": int(num_epochs),
            },
        },
        "cautions": cautions,
        "blockers": blockers,
        "recommendations": recommendations,
        "next_actions": next_actions,
    }


def load_preflight_report(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Preflight report not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Preflight report must be a JSON object.")
    return data


def validate_preflight_gate(
    *,
    preflight_report_path: str | None,
    expected_raw_jsonl_path: str | Path | None,
    require_preflight: bool,
) -> tuple[bool, str]:
    if not require_preflight:
        return True, ""

    if not preflight_report_path:
        return (
            False,
            "Preflight check is required. Run `2) Quality & Normalize` -> `Run Preflight Go/No-Go Check` first.",
        )

    try:
        report = load_preflight_report(preflight_report_path)
    except Exception as e:
        return False, f"Failed to read preflight report: {e}"

    decision = str(report.get("decision", "")).strip().lower()
    if decision == "no-go":
        return False, "Preflight decision is NO-GO. Resolve required failures before training."

    status = str(report.get("status", "")).strip().lower()
    if status == "blocked":
        return False, "Preflight status is BLOCKED. Resolve blockers before starting training."
    if status not in {"ready", "caution"}:
        return False, f"Invalid preflight status: `{report.get('status', 'unknown')}`."

    if expected_raw_jsonl_path:
        expected = Path(expected_raw_jsonl_path).expanduser().resolve()
        reported_raw = (
            report.get("environment", {})
            .get("inputs", {})
            .get("raw_jsonl_path")
        )
        if reported_raw:
            reported = Path(str(reported_raw)).expanduser().resolve()
            if reported != expected:
                return (
                    False,
                    "Preflight report dataset mismatch. "
                    f"expected `{expected}`, report `{reported}`. "
                    "Run preflight again with the selected dataset.",
                )

    return True, ""


def _recommend_hparams(total_minutes: float) -> dict[str, Any]:
    # Conservative defaults tuned for stability first.
    if total_minutes < 10:
        return {
            "profile": "tiny-dataset",
            "batch_size": 2,
            "learning_rate": 2e-5,
            "num_epochs": 20,
            "notes": "Data is very small. Prefer more epochs, stronger regularization via careful monitoring.",
        }
    if total_minutes < 60:
        return {
            "profile": "small-dataset",
            "batch_size": 4,
            "learning_rate": 1e-5,
            "num_epochs": 12,
            "notes": "Typical small speaker adaptation. Monitor early for overfit.",
        }
    if total_minutes < 240:
        return {
            "profile": "medium-dataset",
            "batch_size": 8,
            "learning_rate": 8e-6,
            "num_epochs": 8,
            "notes": "Medium data volume. Reduce epochs to keep timbre consistency.",
        }
    return {
        "profile": "large-dataset",
        "batch_size": 12,
        "learning_rate": 6e-6,
        "num_epochs": 6,
        "notes": "Large set. Favor lower LR and fewer epochs.",
    }


def validate_dataset(raw_jsonl_path: str | Path) -> dict[str, Any]:
    rows = load_raw_jsonl(raw_jsonl_path)
    issues: list[DatasetIssue] = []

    if not rows:
        issues.append(
            DatasetIssue(
                severity="error",
                code="EMPTY_DATASET",
                message="Dataset is empty.",
            )
        )
        return {
            "summary": {
                "samples": 0,
                "ok": False,
                "errors": 1,
                "warnings": 0,
            },
            "issues": [issue.__dict__ for issue in issues],
            "recommendation": _recommend_hparams(0.0),
        }

    sample_rates: dict[int, int] = {}
    ref_audio_counts: dict[str, int] = {}
    total_duration = 0.0
    min_duration = float("inf")
    max_duration = 0.0
    text_len_sum = 0
    missing_audio = 0
    missing_ref = 0
    duplicate_audio_paths = set()
    seen_audio_paths: set[str] = set()

    for idx, row in enumerate(rows, start=1):
        text = str(row.get("text", "")).strip()
        if not text:
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="EMPTY_TEXT",
                    row_index=idx,
                    message="Text is empty.",
                )
            )
        elif len(text) < 2:
            issues.append(
                DatasetIssue(
                    severity="warning",
                    code="SHORT_TEXT",
                    row_index=idx,
                    message=f"Text length is very short ({len(text)}).",
                )
            )
        elif len(text) > 220:
            issues.append(
                DatasetIssue(
                    severity="warning",
                    code="LONG_TEXT",
                    row_index=idx,
                    message=f"Text length is long ({len(text)}). Consider splitting for stability.",
                )
            )
        text_len_sum += len(text)

        audio_path = Path(str(row.get("audio", ""))).expanduser()
        if not audio_path.exists():
            missing_audio += 1
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="MISSING_AUDIO",
                    row_index=idx,
                    message=f"Audio file not found: {audio_path}",
                )
            )
            duration = 0.0
            sr = None
        else:
            sr, duration = _safe_audio_info(audio_path)

            if sr is None:
                issues.append(
                    DatasetIssue(
                        severity="error",
                        code="UNREADABLE_AUDIO",
                        row_index=idx,
                        message=f"Audio file exists but could not be read: {audio_path}",
                    )
                )
        ref_path = Path(str(row.get("ref_audio", ""))).expanduser()
        if not ref_path.exists():
            missing_ref += 1
            issues.append(
                DatasetIssue(
                    severity="error",
                    code="MISSING_REF_AUDIO",
                    row_index=idx,
                    message=f"Reference audio not found: {ref_path}",
                )
            )
        else:
            ref_audio_counts[str(ref_path.resolve())] = (
                ref_audio_counts.get(str(ref_path.resolve()), 0) + 1
            )
            ref_sr, ref_dur = _safe_audio_info(ref_path)
            if ref_sr is None:
                issues.append(
                    DatasetIssue(
                        severity="error",
                        code="UNREADABLE_REF_AUDIO",
                        row_index=idx,
                        message=f"Reference audio file exists but could not be read: {ref_path}",
                    )
                )
            if ref_sr is not None and ref_sr != SAMPLE_RATE_HZ:
                # Training dataset (official) asserts ref_audio is 24kHz when extracting mels.
                issues.append(
                    DatasetIssue(
                        severity="error",
                        code="REF_NON_24K_SAMPLE_RATE",
                        row_index=idx,
                        message=f"ref_audio sample rate is {ref_sr}Hz (required {SAMPLE_RATE_HZ}Hz). Use Normalize step.",
                    )
                )
            if ref_dur > 0 and ref_dur < 2.0:
                issues.append(
                    DatasetIssue(
                        severity="warning",
                        code="SHORT_REF_AUDIO",
                        row_index=idx,
                        message=f"ref_audio duration is short ({ref_dur:.2f}s). Consider a cleaner 3-10s reference.",
                    )
                )

        if sr is not None:
            sample_rates[sr] = sample_rates.get(sr, 0) + 1
            if sr != SAMPLE_RATE_HZ:
                issues.append(
                    DatasetIssue(
                        severity="warning",
                        code="NON_24K_SAMPLE_RATE",
                        row_index=idx,
                        message=f"Sample rate is {sr}Hz (recommended {SAMPLE_RATE_HZ}Hz).",
                    )
                )

        if duration > 0:
            total_duration += duration
            min_duration = min(min_duration, duration)
            max_duration = max(max_duration, duration)
            if duration < 1.0:
                issues.append(
                    DatasetIssue(
                        severity="warning",
                        code="SHORT_AUDIO",
                        row_index=idx,
                        message=f"Audio duration is short ({duration:.2f}s).",
                    )
                )
            if duration > 25.0:
                issues.append(
                    DatasetIssue(
                        severity="warning",
                        code="LONG_AUDIO",
                        row_index=idx,
                        message=f"Audio duration is long ({duration:.2f}s).",
                    )
                )

        audio_key = str(audio_path.resolve()) if audio_path.exists() else str(audio_path)
        if audio_key in seen_audio_paths:
            duplicate_audio_paths.add(audio_key)
        seen_audio_paths.add(audio_key)

    if len(ref_audio_counts) > 1:
        issues.append(
            DatasetIssue(
                severity="warning",
                code="MULTIPLE_REFERENCE_AUDIO",
                message=(
                    f"Multiple reference audios detected ({len(ref_audio_counts)}). "
                    "Single consistent ref_audio is usually more stable."
                ),
            )
        )

    for dup in sorted(duplicate_audio_paths):
        issues.append(
            DatasetIssue(
                severity="warning",
                code="DUPLICATE_AUDIO_ENTRY",
                message=f"Duplicate audio entry detected: {dup}",
            )
        )

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    samples = len(rows)

    avg_duration = total_duration / samples if samples else 0.0
    avg_text_len = text_len_sum / samples if samples else 0.0
    total_minutes = total_duration / 60.0

    recommendation = _recommend_hparams(total_minutes)
    recommendation["speaker_consistency_ref_count"] = len(ref_audio_counts)
    recommendation["estimated_steps_per_epoch"] = max(
        1, samples // max(1, int(recommendation["batch_size"]))
    )

    summary = {
        "samples": samples,
        "ok": errors == 0,
        "errors": errors,
        "warnings": warnings,
        "total_duration_sec": round(total_duration, 2),
        "total_duration_min": round(total_minutes, 2),
        "avg_duration_sec": round(avg_duration, 2),
        "min_duration_sec": round(min_duration if min_duration != float("inf") else 0, 2),
        "max_duration_sec": round(max_duration, 2),
        "avg_text_len": round(avg_text_len, 2),
        "missing_audio": missing_audio,
        "missing_ref_audio": missing_ref,
        "sample_rates": sample_rates,
        "unique_ref_audio_count": len(ref_audio_counts),
    }

    return {
        "summary": summary,
        "issues": [issue.__dict__ for issue in issues],
        "recommendation": recommendation,
    }


def format_quality_report(report: dict[str, Any], max_issues: int = 40) -> str:
    summary = report.get("summary", {})
    issues = report.get("issues", [])
    rec = report.get("recommendation", {})

    lines = [
        f"- samples: `{summary.get('samples', 0)}`",
        f"- valid: `{summary.get('ok', False)}`",
        f"- errors: `{summary.get('errors', 0)}`",
        f"- warnings: `{summary.get('warnings', 0)}`",
        f"- total duration(min): `{summary.get('total_duration_min', 0)}`",
        f"- avg duration(sec): `{summary.get('avg_duration_sec', 0)}`",
        f"- avg text length: `{summary.get('avg_text_len', 0)}`",
        f"- sample rates: `{summary.get('sample_rates', {})}`",
        f"- unique ref_audio count: `{summary.get('unique_ref_audio_count', 0)}`",
        "",
        "### Recommended Training Plan",
        f"- profile: `{rec.get('profile', 'n/a')}`",
        f"- batch_size: `{rec.get('batch_size', 'n/a')}`",
        f"- learning_rate: `{rec.get('learning_rate', 'n/a')}`",
        f"- num_epochs: `{rec.get('num_epochs', 'n/a')}`",
        f"- notes: {rec.get('notes', '')}",
        "",
        "### Issues",
    ]

    if not issues:
        lines.append("- none")
    else:
        for issue in issues[:max_issues]:
            row = issue.get("row_index")
            row_info = f" (row {row})" if row else ""
            lines.append(
                f"- [{issue.get('severity', 'info')}] {issue.get('code', 'UNKNOWN')}{row_info}: {issue.get('message', '')}"
            )
        if len(issues) > max_issues:
            lines.append(f"- ... and {len(issues) - max_issues} more")

    return "\n".join(lines)


def format_preflight_report(report: dict[str, Any], max_lines: int = 280) -> str:
    status = str(report.get("status", "unknown")).upper()
    decision = str(report.get("decision", "unknown")).upper()
    score = report.get("score", "n/a")
    dataset_summary = report.get("dataset", {}).get("summary", {})
    signal = report.get("signal", {})
    env = report.get("environment", {})
    requirements = report.get("requirements", [])
    resource_estimate = report.get("resource_estimate", {})
    blockers = report.get("blockers", [])
    cautions = report.get("cautions", [])
    recommendations = report.get("recommendations", [])
    next_actions = report.get("next_actions", [])

    lines = [
        f"## Preflight Status: `{status}` (score `{score}`)",
        f"## Go/No-Go Decision: `{decision}`",
        "",
        "### Dataset Snapshot",
        f"- samples: `{dataset_summary.get('samples', 0)}`",
        f"- errors: `{dataset_summary.get('errors', 0)}`",
        f"- warnings: `{dataset_summary.get('warnings', 0)}`",
        f"- total duration(min): `{dataset_summary.get('total_duration_min', 0)}`",
        f"- unique ref_audio count: `{dataset_summary.get('unique_ref_audio_count', 0)}`",
        "",
        "### Signal Health (sampled)",
        f"- analyzed files: `{signal.get('analyzed_files', 0)}` / sampled `{signal.get('sampled_files', 0)}`",
        f"- avg SNR(dB): `{signal.get('avg_snr_db', 'n/a')}`",
        f"- avg clipping ratio: `{signal.get('avg_clip_ratio', 'n/a')}`",
        f"- avg silence ratio: `{signal.get('avg_silence_ratio', 'n/a')}`",
        "",
        "### Environment",
        f"- device selected: `{env.get('device', {}).get('selected', 'n/a')}`",
        f"- device ok: `{env.get('device', {}).get('ok', False)}`",
        f"- cuda available: `{env.get('device', {}).get('cuda_available', False)}`",
        f"- mps available: `{env.get('device', {}).get('mps_available', False)}`",
        f"- cuda vram(GB): `{env.get('device', {}).get('cuda_vram_gb', 'n/a')}`",
        f"- hf model accessible: `{env.get('tools', {}).get('hf_model', {}).get('ok', False)}`",
        f"- hf model cached: `{env.get('tools', {}).get('hf_model', {}).get('cached', False)}`",
        f"- flash_attn installed: `{env.get('tools', {}).get('flash_attn', {}).get('ok', False)}`",
        f"- sox available: `{env.get('tools', {}).get('sox', {}).get('ok', False)}`",
        f"- model path mode: `{env.get('model_path', {}).get('mode', 'n/a')}`",
        f"- model path ok: `{env.get('model_path', {}).get('ok', False)}`",
        f"- preflight raw_jsonl: `{env.get('inputs', {}).get('raw_jsonl_path', 'n/a')}`",
        f"- disk free(GB): `{env.get('disk', {}).get('free_gb', 'n/a')}`",
        f"- estimated required disk(GB): `{env.get('estimated_required_disk_gb', 'n/a')}`",
        "",
        "### Resource Estimate (rough)",
        f"- planned batch_size: `{resource_estimate.get('batch_size', 'n/a')}`",
        f"- planned num_epochs: `{resource_estimate.get('num_epochs', 'n/a')}`",
        f"- steps per epoch (est): `{resource_estimate.get('steps_per_epoch_estimate', 'n/a')}`",
        f"- total steps (est): `{resource_estimate.get('total_steps_estimate', 'n/a')}`",
        f"- runtime range (hours): `{resource_estimate.get('estimated_runtime_range_hours', ['n/a', 'n/a'])}`",
        f"- disk free / required (GB): `{resource_estimate.get('free_disk_gb', 'n/a')} / {resource_estimate.get('estimated_required_disk_gb', 'n/a')}`",
        "",
        "### Source Data Requirements (Pass/Fail)",
    ]

    if not requirements:
        lines.append("- none")
    else:
        for item in requirements:
            status_tag = str(item.get("status", "unknown")).upper()
            level = str(item.get("level", "recommended")).upper()
            lines.append(
                f"- [{status_tag}][{level}] {item.get('code', 'REQ')}: "
                f"{item.get('requirement', '')} | current `{item.get('current', '')}` | target `{item.get('target', '')}`"
            )

    lines.append("")
    lines.append("### Blockers")
    if not blockers:
        lines.append("- none")
    else:
        lines.extend([f"- {b}" for b in blockers])

    lines.append("")
    lines.append("### Cautions")
    if not cautions:
        lines.append("- none")
    else:
        lines.extend([f"- {c}" for c in cautions])

    lines.append("")
    lines.append("### Next Actions (before training)")
    if not next_actions:
        lines.append("- none")
    else:
        lines.extend([f"- {a}" for a in next_actions])

    lines.append("")
    lines.append("### Recommendations")
    lines.extend([f"- {r}" for r in recommendations] if recommendations else ["- none"])

    if len(lines) > max_lines:
        lines = lines[:max_lines] + ["- ... truncated ..."]
    return "\n".join(lines)


def _save_json_report(report: dict[str, Any], output_path: str | Path) -> str:
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(out)


def save_quality_report(report: dict[str, Any], output_path: str | Path) -> str:
    return _save_json_report(report, output_path)


def save_preflight_report(report: dict[str, Any], output_path: str | Path) -> str:
    return _save_json_report(report, output_path)


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\uAC00-\uD7A3]+", "", (text or "").lower())


def _levenshtein_ratio(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    n = len(b)
    prev = list(range(n + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * n
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[n]
    return 1.0 - (dist / max(len(a), len(b), 1))


def _safe_transcribe_whisper(audio_path: Path, model_name: str = "base") -> tuple[str | None, str | None]:
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        return None, f"faster-whisper unavailable: {e}"

    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(audio_path),
            language="ko",
            beam_size=3,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return text, None
    except Exception as e:
        return None, f"whisper transcription failed: {e}"


def _safe_profile_target_duration(
    profile_raw_jsonl: str | Path, target_text: str, max_items: int = 200
) -> tuple[float | None, dict[str, Any]]:
    try:
        rows = load_raw_jsonl(profile_raw_jsonl)
    except Exception as e:
        return None, {"error": f"failed to load profile jsonl: {e}"}

    sampled = _sample_rows(rows, max_items=max_items)
    cps_values: list[float] = []
    for row in sampled:
        text = str(row.get("text", "")).strip()
        audio = Path(str(row.get("audio", ""))).expanduser()
        if not text or not audio.exists():
            continue
        _, dur = _safe_audio_info(audio)
        n_chars = len(_normalize_text_for_compare(text))
        if dur <= 0.4 or n_chars < 4:
            continue
        cps_values.append(float(n_chars / dur))

    if not cps_values:
        return None, {"error": "no valid rows for speaking-rate profile"}

    cps_med = float(statistics.median(cps_values))
    target_chars = len(_normalize_text_for_compare(target_text))
    target_duration = float(target_chars / max(cps_med, 1e-6))
    return target_duration, {
        "sampled_rows": len(sampled),
        "valid_rows": len(cps_values),
        "median_chars_per_sec": cps_med,
        "target_chars": target_chars,
    }


def _safe_speaker_cosine(
    generated_audio_path: Path,
    reference_audio_path: Path,
    base_speaker_model: str,
) -> tuple[float | None, str | None]:
    try:
        import librosa
        import torch
        from qwen_tts import Qwen3TTSModel
    except Exception as e:
        return None, f"speaker cosine dependencies unavailable: {e}"

    def _resolve_local_base_model_path(raw: str) -> str:
        val = (raw or "").strip()
        if not val:
            return val
        p = Path(val).expanduser()
        if p.exists():
            return str(p.resolve())
        # Prefer locally cached 0.6B base snapshot in offline environments.
        if val == "Qwen/Qwen3-TTS-12Hz-0.6B-Base":
            try:
                from huggingface_hub.constants import HF_HUB_CACHE
                hub_cache = Path(HF_HUB_CACHE)
            except Exception:
                hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
            root = (
                hub_cache
                / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
                / "snapshots"
            )
            if root.exists():
                snaps = sorted([d for d in root.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
                if snaps:
                    return str(snaps[0].resolve())
        return val

    try:
        resolved_model = _resolve_local_base_model_path(base_speaker_model)
        model = Qwen3TTSModel.from_pretrained(resolved_model, device_map="cpu", dtype=torch.float32)
        gen_wav, _ = librosa.load(str(generated_audio_path), sr=SAMPLE_RATE_HZ, mono=True)
        ref_wav, _ = librosa.load(str(reference_audio_path), sr=SAMPLE_RATE_HZ, mono=True)
        gen_emb = model.model.extract_speaker_embedding(gen_wav.astype(np.float32), SAMPLE_RATE_HZ).float().cpu().numpy()
        ref_emb = model.model.extract_speaker_embedding(ref_wav.astype(np.float32), SAMPLE_RATE_HZ).float().cpu().numpy()
        den = (float(np.linalg.norm(gen_emb)) * float(np.linalg.norm(ref_emb))) + 1e-12
        val = float(np.dot(gen_emb, ref_emb) / den)
        return val, None
    except Exception as e:
        return None, f"speaker cosine failed: {e}"


def _metric_check(
    name: str,
    value: float | None,
    target: str,
    pass_lo: float,
    warn_lo: float,
    pass_hi: float | None = None,
    warn_hi: float | None = None,
) -> dict[str, Any]:
    """Return a pass/warn/fail/unknown check dict for a single metric."""
    if value is None:
        return {"name": name, "status": "unknown", "value": None, "target": target}
    phi = float("inf") if pass_hi is None else pass_hi
    whi = float("inf") if warn_hi is None else warn_hi
    if pass_lo <= value <= phi:
        return {"name": name, "status": "pass", "value": value, "target": target}
    if warn_lo <= value <= whi:
        return {"name": name, "status": "warn", "value": value, "target": target}
    return {"name": name, "status": "fail", "value": value, "target": target}


def run_generation_review(
    *,
    generated_audio_path: str | Path,
    target_text: str,
    reference_audio_path: str | Path = "",
    profile_raw_jsonl: str | Path = "",
    base_speaker_model: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    whisper_model: str = "base",
) -> dict[str, Any]:
    audio_path = Path(generated_audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Generated audio not found: {audio_path}")
    if not str(target_text).strip():
        raise ValueError("target_text is required.")

    sr, duration = _safe_audio_info(audio_path)

    asr_text, asr_err = _safe_transcribe_whisper(audio_path, model_name=whisper_model)
    if asr_text is None:
        asr_sim = None
    else:
        asr_sim = float(
            _levenshtein_ratio(
                _normalize_text_for_compare(target_text),
                _normalize_text_for_compare(asr_text),
            )
        )

    speaker_cos = None
    speaker_err = None
    ref_path = Path(str(reference_audio_path)).expanduser() if str(reference_audio_path).strip() else None
    if ref_path and ref_path.exists():
        speaker_cos, speaker_err = _safe_speaker_cosine(
            generated_audio_path=audio_path,
            reference_audio_path=ref_path,
            base_speaker_model=base_speaker_model,
        )

    target_duration = None
    profile_info: dict[str, Any] = {}
    if str(profile_raw_jsonl).strip():
        target_duration, profile_info = _safe_profile_target_duration(
            profile_raw_jsonl=profile_raw_jsonl,
            target_text=target_text,
        )
    speed_ratio = float(duration / target_duration) if target_duration and target_duration > 0 else None

    checks: list[dict[str, Any]] = []
    checks.append(_metric_check("asr_similarity", asr_sim, ">=0.98", 0.98, 0.90))
    checks.append(_metric_check("speaker_cosine", speaker_cos, ">=0.982", 0.982, 0.970))
    checks.append(_metric_check(
        "speed_ratio", speed_ratio, "0.90~1.15",
        pass_lo=0.90, warn_lo=0.80, pass_hi=1.15, warn_hi=1.30,
    ))

    has_fail = any(c["status"] == "fail" for c in checks)
    has_warn = any(c["status"] == "warn" for c in checks)
    if has_fail:
        decision = "fail"
    elif has_warn:
        decision = "warn"
    else:
        decision = "pass"

    recommendations: list[str] = []
    if speed_ratio is not None and speed_ratio > 1.15:
        recommendations.append(
            "Speech is slower than profile target. Prefer punctuation-light text and run seed sweep; if needed, apply mild tempo-up postprocess."
        )
    if speed_ratio is not None and speed_ratio < 0.90:
        recommendations.append(
            "Speech is faster than profile target. Add short pauses via punctuation or reduce tempo in postprocess."
        )
    if asr_sim is not None and asr_sim < 0.98:
        recommendations.append(
            "ASR similarity is low. Re-run with lower sampling randomness and verify punctuation/token limits."
        )
    if speaker_cos is not None and speaker_cos < 0.982:
        recommendations.append(
            "Speaker cosine is below target. Re-run speaker-embedding selection and multi-seed sampling."
        )
    if not recommendations:
        recommendations.append("Quality is within target range for current checks.")

    return {
        "decision": decision,
        "input": {
            "generated_audio_path": str(audio_path),
            "reference_audio_path": str(ref_path) if ref_path else "",
            "profile_raw_jsonl": str(profile_raw_jsonl) if str(profile_raw_jsonl).strip() else "",
            "base_speaker_model": base_speaker_model,
            "whisper_model": whisper_model,
        },
        "metrics": {
            "sample_rate": sr,
            "duration_sec": duration,
            "target_duration_sec": target_duration,
            "speed_ratio": speed_ratio,
            "speaker_cosine": speaker_cos,
            "asr_similarity": asr_sim,
            "asr_text": asr_text,
        },
        "checks": checks,
        "recommendations": recommendations,
        "diagnostics": {
            "profile_info": profile_info,
            "asr_error": asr_err,
            "speaker_error": speaker_err,
        },
    }


def format_generation_review(report: dict[str, Any]) -> str:
    decision = str(report.get("decision", "unknown")).upper()
    metrics = report.get("metrics", {})
    checks = report.get("checks", [])
    recs = report.get("recommendations", [])

    lines = [
        f"## Generation Review: `{decision}`",
        "",
        "### Metrics",
        f"- duration(sec): `{metrics.get('duration_sec', 'n/a')}`",
        f"- target duration(sec): `{metrics.get('target_duration_sec', 'n/a')}`",
        f"- speed ratio: `{metrics.get('speed_ratio', 'n/a')}`",
        f"- speaker cosine: `{metrics.get('speaker_cosine', 'n/a')}`",
        f"- asr similarity: `{metrics.get('asr_similarity', 'n/a')}`",
        f"- asr text: {metrics.get('asr_text', '')}",
        "",
        "### Checks",
    ]
    if not checks:
        lines.append("- none")
    else:
        for item in checks:
            lines.append(
                f"- [{str(item.get('status', 'unknown')).upper()}] "
                f"{item.get('name', 'check')}: `{item.get('value', 'n/a')}` "
                f"(target `{item.get('target', 'n/a')}`)"
            )

    lines.append("")
    lines.append("### Recommendations")
    if not recs:
        lines.append("- none")
    else:
        lines.extend([f"- {r}" for r in recs])
    return "\n".join(lines)


def save_generation_review(report: dict[str, Any], output_path: str | Path) -> str:
    return _save_json_report(report, output_path)
