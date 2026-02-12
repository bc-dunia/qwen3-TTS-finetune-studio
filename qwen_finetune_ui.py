#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import traceback
from datetime import datetime
import importlib.util
from pathlib import Path
from typing import Any

import gradio as gr

from finetune_studio.audio_prep import normalize_dataset_audio
from finetune_studio.dataset_ops import (
    build_dataset_from_uploads,
    dataset_stats,
    import_existing_raw_jsonl,
    preview_table,
)
from finetune_studio.export_ops import package_checkpoint
from finetune_studio.inference_ops import synthesize_batch, synthesize_single, unload_model
from finetune_studio.pipeline_ops import run_full_pipeline
from finetune_studio.paths import (
    EXPORTS_DIR,
    THIRD_PARTY_FINETUNE_DIR,
    WORKSPACE_ROOT,
    list_checkpoint_paths,
    list_coded_jsonl_paths,
    list_raw_jsonl_paths,
    list_run_paths,
)
from finetune_studio.quality import (
    format_preflight_report,
    format_quality_report,
    run_preflight_review,
    save_preflight_report,
    save_quality_report,
    validate_preflight_gate,
    validate_dataset,
)
from finetune_studio.run_registry import read_run_summary, run_summaries_table
from finetune_studio.process_runner import AlreadyRunningError
from finetune_studio.ui_settings import load_ui_settings, save_ui_settings
from finetune_studio.training_ops import (
    default_run_name,
    expected_raw_jsonl_for_train_jsonl,
    is_prepare_running,
    is_training_running,
    run_prepare_data,
    run_training,
    stop_prepare,
    stop_training,
)

APP_TITLE = "Qwen3-TTS Finetune Studio"

# Match Qwen3-TTS Studio's inference defaults and quick presets.
INFER_DEFAULT_PARAMS: dict[str, Any] = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
}

INFER_PARAM_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.7,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.9,
    },
    "balanced": dict(INFER_DEFAULT_PARAMS),
    "quality": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 4096,
        "subtalker_temperature": 1.0,
        "subtalker_top_k": 80,
        "subtalker_top_p": 1.0,
    },
}

INFER_PARAM_TOOLTIPS: dict[str, str] = {
    "temperature": "Lower = consistent pronunciation, Higher = varied intonation. Natural speech: 0.7-0.9, Precise reading: 0.3-0.5",
    "top_k": "Number of candidates for next token. Lower = stable, Higher = diverse. Recommended: 30-50",
    "top_p": "Probability-based token selection range. 1.0 = full range, lower = more certain. Recommended: 0.9-1.0",
    "repetition_penalty": "Prevents sound/word repetition. 1.0 = no penalty, higher = less repetition. Recommended: 1.0-1.1",
    "max_new_tokens": "Upper limit on generation length. If you see truncated speech, increase this.",
    "subtalker_temperature": "Voice rhythm/accent control. Default recommended, adjust if needed",
    "subtalker_top_k": "Intonation diversity control. Default recommended",
    "subtalker_top_p": "Intonation selection range. Default recommended",
}

SPEAKER_PRESETS: list[str] = [
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
]

CUSTOM_CSS = """
:root {
    --gray-50: #f8f9fa;
    --gray-100: #f1f3f5;
    --gray-200: #e9ecef;
    --gray-300: #dee2e6;
    --gray-400: #ced4da;
    --gray-500: #6c757d;
    --gray-600: #5a6268;
    --gray-700: #495057;
    --gray-800: #343a40;
    --gray-900: #212529;
    --white: #ffffff;
    --radius: 6px;
    --ok: #2d8a00;
    --warn: #e85d04;
    --err: #dc3545;
}

footer { display: none !important; }
.settings-btn, [class*="settings"] { display: none !important; }
.built-with { display: none !important; }

.gradio-container {
    max-width: 1460px !important;
    margin: 0 auto;
    background: var(--gray-50) !important;
}

/* Header (match qwen3-tts-studio tone) */
.main-header {
    text-align: center;
    padding: 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--gray-200);
}
.main-title {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--gray-800);
    margin: 0;
    letter-spacing: -0.01em;
}
.sub-title {
    color: var(--gray-600);
    font-size: 0.9rem;
    margin: 0.35rem 0 0;
}

/* Section headers */
.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--gray-200);
}

/* Panel wrapper */
.params-panel {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 1rem;
}

.gradio-container .tabs {
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    background: var(--white);
    padding: 0.4rem;
}

.gradio-container .tabitem {
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    background: var(--white);
    padding: 0.8rem 0.7rem !important;
}

.gradio-container .tab-nav button {
    border-radius: var(--radius) !important;
    border: 1px solid transparent !important;
}

.gradio-container .tab-nav button.selected {
    background: var(--gray-100) !important;
    border-color: var(--gray-300) !important;
    color: var(--gray-900) !important;
}

.gradio-container .gr-button-primary {
    background: var(--gray-800) !important;
    border-color: var(--gray-800) !important;
}

.gradio-container .gr-button-primary:hover {
    background: var(--gray-900) !important;
    border-color: var(--gray-900) !important;
}

.gradio-container .gr-button-stop {
    background: var(--err) !important;
    border-color: var(--err) !important;
}

.gradio-container .gr-button-secondary {
    background: var(--gray-100) !important;
    border-color: var(--gray-200) !important;
    color: var(--gray-800) !important;
}

.status-ok {
    color: var(--ok);
    font-weight: 700;
}
.status-warn {
    color: var(--warn);
    font-weight: 700;
}
.status-error {
    color: var(--err);
    font-weight: 700;
}

/* Save indicator + presets (ported from qwen3-tts-studio) */
.save-indicator {
    display: inline-block;
    font-size: 0.75rem;
    color: var(--gray-600);
    opacity: 0;
    transition: opacity 0.3s ease;
    padding: 0.25rem 0.5rem;
    background: var(--gray-100);
    border-radius: var(--radius);
}

.save-indicator.show {
    opacity: 1;
    animation: fadeInOut 3s ease;
}

@keyframes fadeInOut {
    0% { opacity: 0; }
    10% { opacity: 1; }
    80% { opacity: 1; }
    100% { opacity: 0; }
}

.preset-btn-group {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.preset-section {
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 0.75rem;
    margin-bottom: 0.75rem;
}

.preset-btn-lg {
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    background: var(--white) !important;
    border: 1px solid var(--gray-300) !important;
    color: var(--gray-700) !important;
}

.preset-btn-lg:hover {
    background: var(--gray-100) !important;
    border-color: var(--gray-400) !important;
}

.generate-btn {
    min-height: 44px !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius) !important;
    background: var(--gray-800) !important;
    border: none !important;
    color: var(--white) !important;
}

.generate-btn:hover {
    background: var(--gray-900) !important;
}

.gradio-accordion {
    border: 1px solid var(--gray-200) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden;
}

.gradio-accordion > .label-wrap {
    background: var(--gray-50) !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--gray-200) !important;
}

.gradio-accordion > .wrap {
    padding: 0.75rem !important;
    background: var(--white) !important;
}

.compact-params-panel .wrap {
    gap: 0.25rem !important;
}
.compact-params-panel input[type="range"] {
    height: 4px !important;
}
.compact-params-panel label span {
    font-size: 0.8rem !important;
    color: var(--gray-700) !important;
}
.compact-params-panel .info {
    font-size: 0.7rem !important;
    line-height: 1.3 !important;
    color: var(--gray-500) !important;
    margin-top: 2px !important;
}
"""

DATASET_PREVIEW_HEADERS = ["audio", "text", "ref_audio", "duration_sec"]
RUN_TABLE_HEADERS = [
    "status",
    "run_name",
    "speaker",
    "train_jsonl",
    "last_loss",
    "epochs_done",
    "checkpoints",
    "created_at",
    "run_dir",
]


def _first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def _format_stats_markdown(stats: dict[str, Any]) -> str:
    if not stats:
        return "No dataset statistics."

    return (
        f"- samples: `{stats.get('samples', 0)}`\n"
        f"- total duration (sec): `{stats.get('total_duration_sec', 0)}`\n"
        f"- avg duration (sec): `{stats.get('avg_duration_sec', 0)}`\n"
        f"- min duration (sec): `{stats.get('min_duration_sec', 0)}`\n"
        f"- max duration (sec): `{stats.get('max_duration_sec', 0)}`\n"
        f"- avg text length: `{stats.get('avg_text_len', 0)}`\n"
        f"- missing files: `{stats.get('missing_files', 0)}`"
    )


def _infer_save_indicator_html(text: str) -> str:
    safe = (text or "Settings saved").replace("<", "&lt;").replace(">", "&gt;")
    return f"<span class='save-indicator show'>{safe}</span>"


def _load_inference_cfg() -> dict[str, Any]:
    settings = load_ui_settings()
    infer = settings.get("inference", {})
    return infer if isinstance(infer, dict) else {}


def _save_inference_cfg(patch: dict[str, Any]) -> dict[str, Any]:
    infer = _load_inference_cfg()
    infer.update(patch)
    save_ui_settings({"inference": infer})
    return infer


def list_speaker_choices() -> list[str]:
    # Presets first (stable order), then speakers observed in run summaries.
    speakers: list[str] = []
    seen: set[str] = set()
    for s in SPEAKER_PRESETS:
        if s not in seen:
            speakers.append(s)
            seen.add(s)

    try:
        for run_path in list_run_paths():
            summary = read_run_summary(run_path)
            name = str(summary.get("speaker_name", "")).strip()
            if name and name not in seen:
                speakers.append(name)
                seen.add(name)
    except Exception:
        pass

    return speakers


def ui_infer_select_speaker_choice(speaker_value: str) -> tuple[str, str]:
    speaker = (speaker_value or "").strip()
    if speaker:
        save_ui_settings({"speaker_name": speaker})
        _save_inference_cfg({"speaker_name": speaker})
    return speaker, _infer_save_indicator_html("Speaker selected")


def ui_infer_on_params_change(
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    subtalker_temperature: float,
    subtalker_top_k: int,
    subtalker_top_p: float,
) -> str:
    params = {
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "repetition_penalty": float(repetition_penalty),
        "max_new_tokens": int(max_new_tokens),
        "subtalker_temperature": float(subtalker_temperature),
        "subtalker_top_k": int(subtalker_top_k),
        "subtalker_top_p": float(subtalker_top_p),
    }
    _save_inference_cfg({"params": params})
    return _infer_save_indicator_html("Settings saved")


def ui_infer_apply_preset(
    preset_key: str,
) -> tuple[float, int, float, float, int, float, int, float, str]:
    key = (preset_key or "").strip().lower()
    preset = INFER_PARAM_PRESETS.get(key, INFER_DEFAULT_PARAMS)
    params = {k: preset.get(k, INFER_DEFAULT_PARAMS[k]) for k in INFER_DEFAULT_PARAMS}
    _save_inference_cfg({"params": params})
    label = key.capitalize() if key else "Preset"
    return (
        float(params["temperature"]),
        int(params["top_k"]),
        float(params["top_p"]),
        float(params["repetition_penalty"]),
        int(params["max_new_tokens"]),
        float(params["subtalker_temperature"]),
        int(params["subtalker_top_k"]),
        float(params["subtalker_top_p"]),
        _infer_save_indicator_html(f"Applied {label} preset"),
    )


def ui_infer_on_checkpoint_change(
    checkpoint_path: str,
    current_speaker_name: str,
) -> tuple[str, str]:
    final_ckpt = (checkpoint_path or "").strip()
    speaker = (current_speaker_name or "").strip()
    if final_ckpt:
        try:
            run_dir = Path(final_ckpt).resolve().parent
            summary = read_run_summary(run_dir)
            if isinstance(summary, dict) and str(summary.get("speaker_name", "")).strip():
                speaker = str(summary["speaker_name"]).strip()
        except Exception:
            pass

    infer_patch: dict[str, Any] = {"checkpoint_path": final_ckpt}
    if speaker:
        infer_patch["speaker_name"] = speaker
        save_ui_settings({"speaker_name": speaker})
    _save_inference_cfg(infer_patch)
    return speaker, _infer_save_indicator_html("Checkpoint selected")


def _refresh_raw_updates(selected: str | None = None) -> gr.Dropdown:
    raws = list_raw_jsonl_paths()
    value = selected if selected in raws else _first_or_none(raws)
    return gr.update(choices=raws, value=value)


def _refresh_coded_updates(selected: str | None = None) -> gr.Dropdown:
    coded = list_coded_jsonl_paths()
    value = selected if selected in coded else _first_or_none(coded)
    return gr.update(choices=coded, value=value)


def _refresh_run_updates(selected: str | None = None) -> gr.Dropdown:
    runs = list_run_paths()
    value = selected if selected in runs else _first_or_none(runs)
    return gr.update(choices=runs, value=value)


def _refresh_checkpoint_updates(selected: str | None = None) -> gr.Dropdown:
    checkpoints = list_checkpoint_paths()
    value = selected if selected in checkpoints else _first_or_none(checkpoints)
    return gr.update(choices=checkpoints, value=value)


def ui_refresh_everything() -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    return (
        _refresh_raw_updates(),
        _refresh_coded_updates(),
        _refresh_run_updates(),
        _refresh_checkpoint_updates(),
    )


def ui_refresh_all_dropdowns(
    inspect_raw_selected: str | None = None,
    quality_raw_selected: str | None = None,
    normalize_raw_selected: str | None = None,
    prepare_raw_selected: str | None = None,
    prepared_jsonl_selected: str | None = None,
    train_jsonl_selected: str | None = None,
    pipeline_raw_selected: str | None = None,
    run_selected: str | None = None,
    run_manage_selected: str | None = None,
    checkpoint_train_selected: str | None = None,
    checkpoint_infer_selected: str | None = None,
    export_ckpt_selected: str | None = None,
) -> tuple[
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
]:
    return (
        _refresh_raw_updates(inspect_raw_selected),        # inspect_raw_path
        _refresh_raw_updates(quality_raw_selected),        # quality_raw_jsonl
        _refresh_raw_updates(normalize_raw_selected),      # normalize_raw_jsonl
        _refresh_raw_updates(prepare_raw_selected),        # prepare_raw_jsonl
        _refresh_coded_updates(prepared_jsonl_selected),   # prepared_jsonl_dropdown
        _refresh_coded_updates(train_jsonl_selected),      # train_jsonl_dropdown
        _refresh_raw_updates(pipeline_raw_selected),       # pipeline_raw_jsonl
        _refresh_run_updates(run_selected),                # run_dropdown
        _refresh_run_updates(run_manage_selected),         # run_dropdown_manage
        _refresh_checkpoint_updates(checkpoint_train_selected), # checkpoint_dropdown_train
        _refresh_checkpoint_updates(checkpoint_infer_selected), # checkpoint_dropdown_infer
        _refresh_checkpoint_updates(export_ckpt_selected), # export_ckpt_dropdown
    )


def ui_build_dataset(
    dataset_name: str,
    uploaded_audios: list[Any] | None,
    transcript_file: Any | None,
    reference_audio_file: Any | None,
) -> tuple[
    str,
    str,
    list[list[Any]],
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
]:
    try:
        dataset_path, raw_jsonl = build_dataset_from_uploads(
            dataset_name=dataset_name,
            uploaded_audios=uploaded_audios,
            transcript_file=transcript_file,
            reference_audio_file=reference_audio_file,
        )
        stats = dataset_stats(raw_jsonl)
        preview = preview_table(raw_jsonl, limit=20)
        status = (
            f"<span class='status-ok'>Dataset ready.</span> "
            f"`{dataset_path}` / `{raw_jsonl}`"
        )
        return (
            status,
            _format_stats_markdown(stats),
            preview,
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
        )
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return (
            f"<span class='status-error'>Dataset build failed:</span> `{e}`\n\n`{tb}`",
            "",
            [],
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
        )


def ui_import_raw_jsonl(
    dataset_name: str,
    raw_jsonl_file: Any | None,
) -> tuple[
    str,
    str,
    list[list[Any]],
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
    gr.Dropdown,
]:
    try:
        dataset_path, raw_jsonl = import_existing_raw_jsonl(
            dataset_name=dataset_name,
            raw_jsonl_file=raw_jsonl_file,
        )
        stats = dataset_stats(raw_jsonl)
        preview = preview_table(raw_jsonl, limit=20)
        status = (
            f"<span class='status-ok'>Raw JSONL imported.</span> "
            f"`{dataset_path}` / `{raw_jsonl}`"
        )
        return (
            status,
            _format_stats_markdown(stats),
            preview,
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
            _refresh_raw_updates(str(raw_jsonl)),
        )
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        return (
            f"<span class='status-error'>Import failed:</span> `{e}`\n\n`{tb}`",
            "",
            [],
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
        )


def ui_inspect_raw(raw_jsonl_path: str) -> tuple[str, list[list[Any]]]:
    if not raw_jsonl_path:
        return "No raw JSONL selected.", []
    try:
        stats = dataset_stats(raw_jsonl_path)
        preview = preview_table(raw_jsonl_path, limit=20)
        return _format_stats_markdown(stats), preview
    except Exception as e:
        return f"Failed to inspect dataset: `{e}`", []


def ui_validate_dataset_for_plan(
    raw_jsonl_path: str,
) -> tuple[str, str | None, int, float, int, int, float, int, int, int]:
    if not raw_jsonl_path:
        return ("Select raw JSONL first.", None, 2, 2e-5, 8, 2, 2e-5, 8, 2, 8)

    try:
        report = validate_dataset(raw_jsonl_path)
        report_md = format_quality_report(report)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(raw_jsonl_path).resolve().parent / f"quality_report_{ts}.json"
        report_file = save_quality_report(report, report_path)

        rec = report.get("recommendation", {})
        bs = int(rec.get("batch_size", 2))
        lr = float(rec.get("learning_rate", 2e-5))
        epochs = int(rec.get("num_epochs", 8))
        return report_md, report_file, bs, lr, epochs, bs, lr, epochs, bs, epochs
    except Exception as e:
        return f"Validation failed: `{e}`", None, 2, 2e-5, 8, 2, 2e-5, 8, 2, 8


def ui_run_preflight_review(
    raw_jsonl_path: str,
    init_model_path: str,
    prepare_device: str,
    batch_size: int,
    num_epochs: int,
) -> tuple[str, str | None, str | None, str | None]:
    if not raw_jsonl_path:
        return "Select raw JSONL first.", None, None, None

    try:
        report = run_preflight_review(
            raw_jsonl_path=raw_jsonl_path,
            init_model_path=init_model_path,
            prepare_device=prepare_device,
            batch_size=int(batch_size),
            num_epochs=int(num_epochs),
        )
        report_md = format_preflight_report(report)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = (
            Path(raw_jsonl_path).resolve().parent / f"preflight_report_{ts}.json"
        )
        report_file = save_preflight_report(report, report_path)
        # Also fan out the same report file to other tabs' "preflight report" inputs.
        return report_md, report_file, report_file, report_file
    except Exception as e:
        return f"Preflight review failed: `{e}`", None, None, None


def ui_normalize_dataset(
    raw_jsonl_path: str,
    normalized_dataset_name: str,
    target_sr: int,
    peak_normalize: bool,
) -> tuple[str, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    if not raw_jsonl_path:
        return (
            "Select raw JSONL first.",
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
        )
    try:
        out_dir, out_raw = normalize_dataset_audio(
            raw_jsonl_path=raw_jsonl_path,
            normalized_dataset_name=normalized_dataset_name,
            target_sr=int(target_sr),
            peak_normalize=bool(peak_normalize),
        )
        status = f"Normalized dataset created: `{out_dir}`"
        return (
            status,
            _refresh_raw_updates(out_raw),
            _refresh_raw_updates(out_raw),
            _refresh_raw_updates(out_raw),
            _refresh_raw_updates(out_raw),
            _refresh_raw_updates(out_raw),
        )
    except Exception as e:
        return (
            f"Normalize failed: `{e}`",
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
            _refresh_raw_updates(),
        )


def ui_run_prepare(
    raw_jsonl_path: str,
    device: str,
    tokenizer_model_path: str,
    output_filename: str,
    batch_infer_num: int,
):
    if not raw_jsonl_path:
        yield (
            "Raw JSONL path is required.",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
        )
        return

    if is_prepare_running():
        yield (
            "Prepare is already running. Use `Stop Prepare` if you want to cancel it.",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
        )
        return

    try:
        for event in run_prepare_data(
            device=device,
            tokenizer_model_path=tokenizer_model_path,
            input_jsonl=raw_jsonl_path,
            output_filename=output_filename,
            batch_infer_num=int(batch_infer_num),
        ):
            output_path = event.get("output_jsonl")
            coded_choices = event.get("coded_jsonl_choices", list_coded_jsonl_paths())
            selected = (
                output_path
                if output_path and output_path in coded_choices
                else _first_or_none(coded_choices)
            )
            yield (
                event.get("status", ""),
                event.get("logs", ""),
                gr.update(choices=coded_choices, value=selected),
                gr.update(choices=coded_choices, value=selected),
            )
    except Exception as e:
        yield (
            f"Prepare failed: `{e}`",
            traceback.format_exc(),
            _refresh_coded_updates(),
            _refresh_coded_updates(),
        )


def ui_stop_prepare() -> str:
    return stop_prepare()


def ui_make_new_run_name() -> str:
    return default_run_name()


def ui_run_training(
    init_model_path: str,
    train_jsonl_path: str,
    run_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    speaker_name: str,
    require_preflight: bool,
    preflight_report_path: Any | None,
    speaker_id: int,
    gradient_accumulation_steps: int,
    mixed_precision: str,
    weight_decay: float,
    max_grad_norm: float,
    subtalker_loss_weight: float,
    attn_implementation: str,
    torch_dtype: str,
    log_every_n_steps: int,
    save_every_n_epochs: int,
    max_steps: int,
    seed: int,
):
    if not train_jsonl_path:
        yield (
            "Prepared JSONL path is required.",
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    if is_training_running():
        yield (
            "Training is already running. Use `Stop Training` before starting a new run.",
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    final_speaker = speaker_name.strip()
    if not final_speaker:
        yield (
            "Speaker Name is required.",
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    # Persist commonly reused fields across tabs (matches qwen3-tts-studio auto-save UX).
    save_ui_settings({"speaker_name": final_speaker})

    # Preflight gate (optional but recommended)
    expected_raw = expected_raw_jsonl_for_train_jsonl(train_jsonl_path)
    preflight_path = ""
    if preflight_report_path:
        if isinstance(preflight_report_path, str):
            preflight_path = preflight_report_path
        elif isinstance(preflight_report_path, dict) and "name" in preflight_report_path:
            preflight_path = str(preflight_report_path["name"])
        elif hasattr(preflight_report_path, "name"):
            preflight_path = str(preflight_report_path.name)

    ok, reason = validate_preflight_gate(
        preflight_report_path=preflight_path or None,
        expected_raw_jsonl_path=expected_raw,
        require_preflight=bool(require_preflight),
    )
    if not ok:
        yield (
            reason,
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    final_attn = (attn_implementation or "").strip()
    if final_attn.lower() == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        yield (
            "flash_attn is not installed. Choose `attn_implementation=auto/sdpa/eager` or install flash-attn.",
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    try:
        for event in run_training(
            init_model_path=init_model_path,
            train_jsonl=train_jsonl_path,
            run_name=run_name,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            speaker_name=final_speaker,
            speaker_id=int(speaker_id),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            mixed_precision=str(mixed_precision),
            weight_decay=float(weight_decay),
            max_grad_norm=float(max_grad_norm),
            subtalker_loss_weight=float(subtalker_loss_weight),
            attn_implementation=final_attn,
            torch_dtype=str(torch_dtype),
            log_every_n_steps=int(log_every_n_steps),
            save_every_n_epochs=int(save_every_n_epochs),
            max_steps=int(max_steps),
            random_seed=int(seed),
        ):
            run_dir = event.get("run_dir")
            run_choices = event.get("run_choices", list_run_paths())
            checkpoint_choices = event.get("checkpoint_choices", list_checkpoint_paths())
            selected_run = run_dir if run_dir in run_choices else _first_or_none(run_choices)
            selected_ckpt = event.get("last_checkpoint")
            if selected_ckpt not in checkpoint_choices:
                selected_ckpt = _first_or_none(checkpoint_choices)

            yield (
                event.get("status", ""),
                event.get("progress", ""),
                event.get("logs", ""),
                gr.update(choices=run_choices, value=selected_run),
                gr.update(choices=run_choices, value=selected_run),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
            )
    except AlreadyRunningError as e:
        yield (
            str(e),
            "",
            "",
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
    except Exception as e:
        yield (
            f"Training failed: `{e}`",
            "",
            traceback.format_exc(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )


def ui_stop_training() -> str:
    return stop_training()


def ui_run_pipeline(
    raw_jsonl_path: str,
    prepare_device: str,
    tokenizer_model_path: str,
    prepare_output_filename: str,
    prepare_batch_infer_num: int,
    init_model_path: str,
    run_name: str,
    batch_size: int,
    learning_rate: float,
    num_epochs: int,
    speaker_name: str,
    require_preflight: bool,
    preflight_report_path: Any | None,
    speaker_id: int,
    gradient_accumulation_steps: int,
    mixed_precision: str,
    weight_decay: float,
    max_grad_norm: float,
    subtalker_loss_weight: float,
    attn_implementation: str,
    torch_dtype: str,
    log_every_n_steps: int,
    save_every_n_epochs: int,
    max_steps: int,
    seed: int,
):
    if not raw_jsonl_path:
        yield (
            "Raw JSONL path is required.",
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    if is_prepare_running() or is_training_running():
        running = []
        if is_prepare_running():
            running.append("prepare")
        if is_training_running():
            running.append("training")
        yield (
            f"Pipeline cannot start because these stages are already running: {', '.join(running)}. Stop them first.",
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    final_speaker = speaker_name.strip()
    if not final_speaker:
        yield (
            "Speaker Name is required.",
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    # Persist commonly reused fields across tabs (matches qwen3-tts-studio auto-save UX).
    save_ui_settings({"speaker_name": final_speaker})

    preflight_path = ""
    if preflight_report_path:
        if isinstance(preflight_report_path, str):
            preflight_path = preflight_report_path
        elif isinstance(preflight_report_path, dict) and "name" in preflight_report_path:
            preflight_path = str(preflight_report_path["name"])
        elif hasattr(preflight_report_path, "name"):
            preflight_path = str(preflight_report_path.name)

    ok, reason = validate_preflight_gate(
        preflight_report_path=preflight_path or None,
        expected_raw_jsonl_path=raw_jsonl_path,
        require_preflight=bool(require_preflight),
    )
    if not ok:
        yield (
            reason,
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    final_attn = (attn_implementation or "").strip()
    if final_attn.lower() == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        yield (
            "flash_attn is not installed. Choose `attn_implementation=auto/sdpa/eager` or install flash-attn.",
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
        return

    try:
        for event in run_full_pipeline(
            raw_jsonl_path=raw_jsonl_path,
            prepare_device=prepare_device,
            tokenizer_model_path=tokenizer_model_path,
            prepare_output_filename=prepare_output_filename,
            prepare_batch_infer_num=int(prepare_batch_infer_num),
            init_model_path=init_model_path,
            run_name=run_name,
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            num_epochs=int(num_epochs),
            speaker_name=final_speaker,
            speaker_id=int(speaker_id),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            mixed_precision=str(mixed_precision),
            weight_decay=float(weight_decay),
            max_grad_norm=float(max_grad_norm),
            subtalker_loss_weight=float(subtalker_loss_weight),
            attn_implementation=final_attn,
            torch_dtype=str(torch_dtype),
            log_every_n_steps=int(log_every_n_steps),
            save_every_n_epochs=int(save_every_n_epochs),
            max_steps=int(max_steps),
            random_seed=int(seed),
        ):
            stage = event.get("stage", "")
            status = event.get("status", "")
            if stage:
                status = f"[{stage}] {status}"

            coded_choices = event.get("coded_jsonl_choices", list_coded_jsonl_paths())
            run_choices = event.get("run_choices", list_run_paths())
            checkpoint_choices = event.get("checkpoint_choices", list_checkpoint_paths())

            selected_coded = event.get("output_jsonl")
            if selected_coded not in coded_choices:
                selected_coded = _first_or_none(coded_choices)

            selected_run = event.get("run_dir")
            if selected_run not in run_choices:
                selected_run = _first_or_none(run_choices)

            selected_ckpt = event.get("last_checkpoint")
            if selected_ckpt not in checkpoint_choices:
                selected_ckpt = _first_or_none(checkpoint_choices)

            yield (
                status,
                event.get("progress", ""),
                event.get("logs", ""),
                gr.update(choices=coded_choices, value=selected_coded),
                gr.update(choices=coded_choices, value=selected_coded),
                gr.update(choices=run_choices, value=selected_run),
                gr.update(choices=run_choices, value=selected_run),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
                gr.update(choices=checkpoint_choices, value=selected_ckpt),
            )
    except AlreadyRunningError as e:
        yield (
            str(e),
            "",
            "",
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )
    except Exception as e:
        yield (
            f"Pipeline failed: `{e}`",
            "",
            traceback.format_exc(),
            _refresh_coded_updates(),
            _refresh_coded_updates(),
            _refresh_run_updates(),
            _refresh_run_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
            _refresh_checkpoint_updates(),
        )


def ui_stop_pipeline() -> str:
    prepare_was_running = is_prepare_running()
    training_was_running = is_training_running()
    prepare_status = stop_prepare()
    train_status = stop_training()
    return (
        "Pipeline stop request:\n"
        f"- prepare: running={prepare_was_running} -> {prepare_status}\n"
        f"- training: running={training_was_running} -> {train_status}"
    )


def ui_load_run_checkpoints(run_path: str) -> tuple[gr.Dropdown, gr.Dropdown, gr.Dropdown]:
    if not run_path:
        update = _refresh_checkpoint_updates()
        return update, update, update

    run_dir = Path(run_path)
    checkpoints = [p for p in run_dir.glob("checkpoint-epoch-*") if p.is_dir()]
    checkpoints = sorted(checkpoints, key=lambda p: p.stat().st_mtime, reverse=True)
    values = [str(p.resolve()) for p in checkpoints]
    value = _first_or_none(values)
    update = gr.update(choices=values, value=value)
    return update, update, update


def _generation_params(
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    subtalker_temperature: float,
    subtalker_top_k: int,
    subtalker_top_p: float,
) -> dict[str, Any]:
    return {
        "temperature": temperature,
        "top_k": int(top_k),
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": int(max_new_tokens),
        "subtalker_temperature": subtalker_temperature,
        "subtalker_top_k": int(subtalker_top_k),
        "subtalker_top_p": subtalker_top_p,
    }


def ui_generate_single(
    checkpoint_path: str,
    device: str,
    speaker_name: str,
    language: str,
    instruct: str,
    text: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    subtalker_temperature: float,
    subtalker_top_k: int,
    subtalker_top_p: float,
) -> tuple[str, str | None]:
    params = _generation_params(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        subtalker_temperature=subtalker_temperature,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
    )
    final_speaker = (speaker_name or "").strip()
    if final_speaker:
        save_ui_settings({"speaker_name": final_speaker})
    _save_inference_cfg(
        {
            "checkpoint_path": (checkpoint_path or "").strip(),
            "device": (device or "").strip() or "auto",
            "speaker_name": final_speaker,
            "language": (language or "").strip() or "auto",
            "instruct": (instruct or "").strip(),
            "params": params,
        }
    )
    try:
        wav_path, status = synthesize_single(
            checkpoint_path=checkpoint_path,
            device=device,
            speaker_name=speaker_name,
            text=text,
            params=params,
            language=language,
            instruct=instruct,
        )
        return status, wav_path
    except Exception as e:
        return f"Single generation failed: `{e}`", None


def ui_generate_batch(
    checkpoint_path: str,
    device: str,
    speaker_name: str,
    language: str,
    instruct: str,
    batch_text: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    subtalker_temperature: float,
    subtalker_top_k: int,
    subtalker_top_p: float,
) -> tuple[str, str | None, str | None]:
    params = _generation_params(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        subtalker_temperature=subtalker_temperature,
        subtalker_top_k=subtalker_top_k,
        subtalker_top_p=subtalker_top_p,
    )
    final_speaker = (speaker_name or "").strip()
    if final_speaker:
        save_ui_settings({"speaker_name": final_speaker})
    _save_inference_cfg(
        {
            "checkpoint_path": (checkpoint_path or "").strip(),
            "device": (device or "").strip() or "auto",
            "speaker_name": final_speaker,
            "language": (language or "").strip() or "auto",
            "instruct": (instruct or "").strip(),
            "params": params,
        }
    )
    try:
        first_wav, zip_path, status = synthesize_batch(
            checkpoint_path=checkpoint_path,
            device=device,
            speaker_name=speaker_name,
            multiline_text=batch_text,
            params=params,
            language=language,
            instruct=instruct,
        )
        return status, first_wav, zip_path
    except Exception as e:
        return f"Batch generation failed: `{e}`", None, None


def ui_unload_infer_model() -> str:
    return unload_model()


def ui_refresh_run_table() -> list[list[Any]]:
    return run_summaries_table(limit=200)


def ui_show_run_summary(run_path: str) -> str:
    if not run_path:
        return "Select a run path first."
    try:
        summary = read_run_summary(run_path)
        if not summary:
            config_path = Path(run_path) / "run_config.json"
            if config_path.exists():
                return config_path.read_text(encoding="utf-8")
            return "No run summary found yet."
        lines = []
        for key in [
            "status",
            "run_name",
            "speaker_name",
            "train_jsonl",
            "init_model_path",
            "batch_size",
            "learning_rate",
            "num_epochs",
            "samples",
            "epochs_done",
            "last_step",
            "last_loss",
            "progress",
            "checkpoints",
            "last_checkpoint",
            "elapsed_sec",
            "created_at",
            "updated_at",
        ]:
            if key in summary:
                lines.append(f"- {key}: `{summary[key]}`")
        return "\n".join(lines) if lines else "Run summary is empty."
    except Exception as e:
        return f"Failed to load run summary: `{e}`"


def ui_export_checkpoint(
    checkpoint_path: str,
    include_optimizer_files: bool,
) -> tuple[str, str | None]:
    if not checkpoint_path:
        return "Select checkpoint first.", None
    try:
        zip_path, status = package_checkpoint(
            checkpoint_path=checkpoint_path,
            include_optimizer_files=include_optimizer_files,
        )
        return status, zip_path
    except Exception as e:
        return f"Export failed: `{e}`", None


def ui_workspace_overview() -> str:
    raws = list_raw_jsonl_paths()
    coded = list_coded_jsonl_paths()
    runs = list_run_paths()
    checkpoints = list_checkpoint_paths()
    exports = sorted(EXPORTS_DIR.glob("**/*.zip")) if EXPORTS_DIR.exists() else []
    return (
        f"- workspace: `{WORKSPACE_ROOT.resolve()}`\n"
        f"- finetuning scripts: `{THIRD_PARTY_FINETUNE_DIR}`\n"
        f"- raw datasets: `{len(raws)}`\n"
        f"- prepared datasets: `{len(coded)}`\n"
        f"- runs: `{len(runs)}`\n"
        f"- checkpoints: `{len(checkpoints)}`\n"
        f"- exports(zip): `{len(exports)}`"
    )


def ui_environment_check() -> str:
    checks: list[tuple[str, bool, str]] = []
    for module_name in [
        "gradio",
        "soundfile",
        "librosa",
        "numpy",
        "torch",
        "accelerate",
        "transformers",
        "huggingface_hub",
        "safetensors",
        "qwen_tts",
    ]:
        ok = importlib.util.find_spec(module_name) is not None
        checks.append((f"python module `{module_name}`", ok, "required"))

    for module_name in [
        "flash_attn",
        "tensorboard",
    ]:
        ok = importlib.util.find_spec(module_name) is not None
        checks.append((f"python module `{module_name}`", ok, "recommended"))

    checks.append(("binary `sox`", bool(shutil.which("sox")), "recommended"))

    checks.append(
        (
            f"official finetuning scripts `{THIRD_PARTY_FINETUNE_DIR}`",
            THIRD_PARTY_FINETUNE_DIR.exists(),
            "required",
        )
    )

    summary_lines = []
    ok_count = 0
    for name, ok, level in checks:
        status = "OK" if ok else "MISSING"
        if ok:
            ok_count += 1
        summary_lines.append(f"- {name}: `{status}` ({level})")

    summary_lines.append(f"\n- result: `{ok_count}/{len(checks)} checks passed`")
    return "\n".join(summary_lines)


def build_app() -> gr.Blocks:
    ui_settings = load_ui_settings()
    default_speaker_name = str(ui_settings.get("speaker_name", "") or "").strip()
    infer_cfg = ui_settings.get("inference", {}) if isinstance(ui_settings.get("inference", {}), dict) else {}
    infer_params = infer_cfg.get("params", {}) if isinstance(infer_cfg.get("params", {}), dict) else {}
    infer_device_default = str(infer_cfg.get("device", "auto") or "auto")
    infer_language_default = str(infer_cfg.get("language", "auto") or "auto")
    infer_instruct_default = str(infer_cfg.get("instruct", "") or "")
    infer_ckpt_default = str(infer_cfg.get("checkpoint_path", "") or "")

    with gr.Blocks(title=APP_TITLE, css=CUSTOM_CSS) as demo:
        gr.HTML(
            """
            <div class="main-header">
              <h1 class="main-title">Qwen3-TTS Finetune Studio</h1>
              <p class="sub-title">Dataset → Quality/Preflight → Prepare → Train → Inference</p>
            </div>
            """
        )

        with gr.Row():
            refresh_all_button = gr.Button("Refresh All Paths", variant="secondary")

        with gr.Tabs():
            with gr.Tab("1) Dataset"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Dataset Builder</div>')
                        dataset_name = gr.Textbox(
                            label="Dataset Name",
                            value="my_dataset",
                            placeholder="e.g. my_speaker_dataset",
                        )
                        uploaded_audios = gr.Files(
                            label="Upload training audio files (optional if transcript has valid absolute paths)",
                            file_types=["audio"],
                        )
                        transcript_file = gr.File(
                            label="Transcript (.csv/.jsonl/.json) with required keys: audio, text",
                            file_types=[".csv", ".jsonl", ".json"],
                        )
                        reference_audio_file = gr.File(
                            label="Global reference audio (recommended single reference)",
                            file_types=["audio"],
                        )
                        build_dataset_button = gr.Button("Build train_raw.jsonl", variant="primary")

                        gr.HTML('<div class="section-header" style="margin-top:1rem;">Import Existing</div>')
                        import_raw_jsonl_file = gr.File(
                            label="Raw JSONL file",
                            file_types=[".jsonl"],
                        )
                        import_raw_button = gr.Button("Import Raw JSONL")

                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Preview</div>')
                        dataset_status = gr.Markdown()
                        dataset_stats_view = gr.Markdown()
                        dataset_preview = gr.Dataframe(
                            label="Dataset Preview (first 20 rows)",
                            headers=DATASET_PREVIEW_HEADERS,
                            datatype=["str", "str", "str", "number"],
                            interactive=False,
                            row_count=20,
                            col_count=(4, "fixed"),
                        )
                        inspect_raw_path = gr.Dropdown(
                            label="Inspect existing raw JSONL",
                            choices=list_raw_jsonl_paths(),
                            value=_first_or_none(list_raw_jsonl_paths()),
                        )
                        inspect_button = gr.Button("Inspect Raw JSONL")

            with gr.Tab("2) Quality & Normalize"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Quality Validation</div>')
                        quality_raw_jsonl = gr.Dropdown(
                            label="Raw JSONL for quality validation",
                            choices=list_raw_jsonl_paths(),
                            value=_first_or_none(list_raw_jsonl_paths()),
                        )
                        validate_button = gr.Button("Run Quality Validation", variant="primary")
                        quality_report_file = gr.File(label="Quality Report JSON")
                        gr.HTML('<div class="section-header" style="margin-top:1rem;">Preflight Go/No-Go</div>')
                        gr.Markdown(
                            "- REQUIRED: zero blocking dataset errors, valid model path, available device, enough disk\n"
                            "- RECOMMENDED: >=10 min clean speech, single reference audio, high text diversity, low noise/clipping"
                        )
                        preflight_init_model_path = gr.Textbox(
                            label="Init model path for review",
                            value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        )
                        preflight_device = gr.Dropdown(
                            label="Target prepare device",
                            choices=["cuda:0", "mps", "cpu", "auto"],
                            value="auto",
                        )
                        with gr.Row():
                            preflight_batch_size = gr.Number(
                                label="Planned batch_size",
                                value=2,
                                precision=0,
                            )
                            preflight_num_epochs = gr.Number(
                                label="Planned num_epochs",
                                value=3,
                                precision=0,
                            )
                        preflight_button = gr.Button("Run Preflight Go/No-Go Check", variant="secondary")
                        preflight_report_file = gr.File(label="Preflight Report JSON")
                        gr.HTML('<div class="section-header" style="margin-top:1rem;">Normalize Audio</div>')
                        gr.Markdown(
                            "- Resample to 24kHz mono\n"
                            "- Optional peak normalization"
                        )
                        normalize_raw_jsonl = gr.Dropdown(
                            label="Raw JSONL for normalization",
                            choices=list_raw_jsonl_paths(),
                            value=_first_or_none(list_raw_jsonl_paths()),
                        )
                        normalized_dataset_name = gr.Textbox(
                            label="Normalized dataset name",
                            value="normalized_dataset",
                        )
                        normalize_target_sr = gr.Dropdown(
                            label="Target sample rate",
                            choices=[24000, 22050, 16000],
                            value=24000,
                        )
                        normalize_peak = gr.Checkbox(
                            label="Peak normalize audio",
                            value=True,
                        )
                        normalize_button = gr.Button("Normalize Dataset Audio")
                        normalize_status = gr.Markdown()
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Reports</div>')
                        quality_report_view = gr.Markdown()
                        preflight_report_view = gr.Markdown()

            with gr.Tab("3) Prepare Codes"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Prepare Codes</div>')
                        prepare_raw_jsonl = gr.Dropdown(
                            label="Input Raw JSONL",
                            choices=list_raw_jsonl_paths(),
                            value=_first_or_none(list_raw_jsonl_paths()),
                        )
                        prepare_device = gr.Dropdown(
                            label="Device",
                            choices=["auto", "cuda:0", "mps", "cpu"],
                            value="auto",
                        )
                        tokenizer_model_path = gr.Textbox(
                            label="Tokenizer model path",
                            value="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        )
                        output_filename = gr.Textbox(
                            label="Output filename",
                            value="train_with_codes.jsonl",
                        )
                        prepare_batch_infer_num = gr.Slider(
                            label="Batch infer size (prepare_data.py)",
                            minimum=1,
                            maximum=256,
                            value=32,
                            step=1,
                        )
                        with gr.Row():
                            run_prepare_button = gr.Button("Run prepare_data.py", variant="primary")
                            stop_prepare_button = gr.Button("Stop Prepare", variant="stop")
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Logs & Output</div>')
                        prepare_status = gr.Markdown()
                        prepare_logs = gr.Textbox(
                            label="Prepare Logs",
                            lines=20,
                            max_lines=30,
                            autoscroll=True,
                            interactive=False,
                        )
                        prepared_jsonl_dropdown = gr.Dropdown(
                            label="Prepared JSONL",
                            choices=list_coded_jsonl_paths(),
                            value=_first_or_none(list_coded_jsonl_paths()),
                        )

            with gr.Tab("4) Train"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Training Setup</div>')
                        init_model_path = gr.Textbox(
                            label="Init model path",
                            value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        )
                        gr.Markdown(
                            "### Notes\n"
                            "- Run `2) Quality & Normalize` → Preflight Go/No-Go Check before training.\n"
                            "- `attn_implementation=flash_attention_2` requires `flash_attn`.\n"
                            "- Use `max_steps` for quick smoke tests."
                        )
                        with gr.Accordion("Safety Gate (Recommended)", open=True):
                            require_preflight_train = gr.Checkbox(
                                label="Require preflight gate (NO-GO/BLOCKED prevents training)",
                                value=True,
                            )
                            train_preflight_report_file = gr.File(
                                label="Preflight Report JSON (from tab 2)",
                                file_types=[".json"],
                            )
                        train_jsonl_dropdown = gr.Dropdown(
                            label="Train JSONL (with audio_codes)",
                            choices=list_coded_jsonl_paths(),
                            value=_first_or_none(list_coded_jsonl_paths()),
                        )
                        with gr.Row():
                            run_name = gr.Textbox(label="Run Name", value=default_run_name())
                            new_run_name_button = gr.Button("New Name")
                        speaker_name = gr.Textbox(
                            label="Speaker Name",
                            value=default_speaker_name,
                            placeholder="e.g. my_speaker",
                        )
                        batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=64,
                            value=2,
                            step=1,
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=2e-5,
                        )
                        num_epochs = gr.Slider(
                            label="Num Epochs",
                            minimum=1,
                            maximum=50,
                            value=3,
                            step=1,
                        )
                        with gr.Accordion("Advanced Training Options", open=False):
                            speaker_id_train = gr.Number(
                                label="Speaker ID (advanced)",
                                value=3000,
                                precision=0,
                            )
                            gradient_accumulation_steps_train = gr.Slider(
                                label="Gradient Accumulation Steps",
                                minimum=1,
                                maximum=64,
                                value=4,
                                step=1,
                            )
                            mixed_precision_train = gr.Dropdown(
                                label="Mixed precision",
                                choices=["auto", "bf16", "fp16", "no"],
                                value="auto",
                            )
                            torch_dtype_train = gr.Dropdown(
                                label="Model torch_dtype",
                                choices=["auto", "bfloat16", "float16", "float32"],
                                value="auto",
                            )
                            attn_implementation_train = gr.Dropdown(
                                label="attn_implementation",
                                choices=["auto", "flash_attention_2", "sdpa", "eager"],
                                value="auto",
                            )
                            weight_decay_train = gr.Number(label="Weight decay", value=0.01)
                            max_grad_norm_train = gr.Number(label="Max grad norm", value=1.0)
                            subtalker_loss_weight_train = gr.Number(
                                label="Subtalker loss weight",
                                value=0.3,
                            )
                            log_every_n_steps_train = gr.Slider(
                                label="Log every N steps",
                                minimum=1,
                                maximum=200,
                                value=10,
                                step=1,
                            )
                            save_every_n_epochs_train = gr.Slider(
                                label="Save every N epochs",
                                minimum=1,
                                maximum=20,
                                value=1,
                                step=1,
                            )
                            max_steps_train = gr.Number(
                                label="Max steps (0 = unlimited)",
                                value=0,
                                precision=0,
                            )
                            seed_train = gr.Number(label="Random seed", value=42, precision=0)
                        with gr.Row():
                            run_train_button = gr.Button("Run sft_12hz.py", variant="primary")
                            stop_train_button = gr.Button("Stop Training", variant="stop")
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Logs & Checkpoints</div>')
                        train_status = gr.Markdown()
                        train_progress = gr.Markdown()
                        train_logs = gr.Textbox(
                            label="Training Logs",
                            lines=20,
                            max_lines=30,
                            autoscroll=True,
                            interactive=False,
                        )
                        run_dropdown = gr.Dropdown(
                            label="Runs",
                            choices=list_run_paths(),
                            value=_first_or_none(list_run_paths()),
                        )
                        checkpoint_dropdown_train = gr.Dropdown(
                            label="Checkpoints (train tab)",
                            choices=list_checkpoint_paths(),
                            value=_first_or_none(list_checkpoint_paths()),
                        )

            with gr.Tab("5) Full Pipeline"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Pipeline Config</div>')
                        pipeline_raw_jsonl = gr.Dropdown(
                            label="Raw JSONL",
                            choices=list_raw_jsonl_paths(),
                            value=_first_or_none(list_raw_jsonl_paths()),
                        )
                        pipeline_prepare_device = gr.Dropdown(
                            label="Prepare device",
                            choices=["auto", "cuda:0", "mps", "cpu"],
                            value="auto",
                        )
                        pipeline_tokenizer_model_path = gr.Textbox(
                            label="Tokenizer model path",
                            value="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                        )
                        pipeline_output_filename = gr.Textbox(
                            label="Prepared JSONL filename",
                            value="train_with_codes.jsonl",
                        )
                        pipeline_prepare_batch_infer_num = gr.Slider(
                            label="Batch infer size (prepare_data.py)",
                            minimum=1,
                            maximum=256,
                            value=32,
                            step=1,
                        )
                        pipeline_init_model_path = gr.Textbox(
                            label="Init model path",
                            value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        )
                        with gr.Row():
                            pipeline_run_name = gr.Textbox(
                                label="Run name",
                                value=default_run_name(),
                            )
                            pipeline_new_run_name = gr.Button("New Name")
                        pipeline_speaker_name = gr.Textbox(
                            label="Speaker name",
                            value=default_speaker_name,
                            placeholder="e.g. my_speaker",
                        )
                        pipeline_batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=64,
                            value=2,
                            step=1,
                        )
                        pipeline_learning_rate = gr.Number(
                            label="Learning Rate",
                            value=2e-5,
                        )
                        pipeline_num_epochs = gr.Slider(
                            label="Num Epochs",
                            minimum=1,
                            maximum=50,
                            value=3,
                            step=1,
                        )
                        with gr.Accordion("Safety Gate (Recommended)", open=True):
                            require_preflight_pipeline = gr.Checkbox(
                                label="Require preflight gate (NO-GO/BLOCKED prevents pipeline)",
                                value=True,
                            )
                            pipeline_preflight_report_file = gr.File(
                                label="Preflight Report JSON (from tab 2)",
                                file_types=[".json"],
                            )
                        with gr.Accordion("Advanced Training Options", open=False):
                            speaker_id_pipeline = gr.Number(
                                label="Speaker ID (advanced)",
                                value=3000,
                                precision=0,
                            )
                            gradient_accumulation_steps_pipeline = gr.Slider(
                                label="Gradient Accumulation Steps",
                                minimum=1,
                                maximum=64,
                                value=4,
                                step=1,
                            )
                            mixed_precision_pipeline = gr.Dropdown(
                                label="Mixed precision",
                                choices=["auto", "bf16", "fp16", "no"],
                                value="auto",
                            )
                            torch_dtype_pipeline = gr.Dropdown(
                                label="Model torch_dtype",
                                choices=["auto", "bfloat16", "float16", "float32"],
                                value="auto",
                            )
                            attn_implementation_pipeline = gr.Dropdown(
                                label="attn_implementation",
                                choices=["auto", "flash_attention_2", "sdpa", "eager"],
                                value="auto",
                            )
                            weight_decay_pipeline = gr.Number(label="Weight decay", value=0.01)
                            max_grad_norm_pipeline = gr.Number(label="Max grad norm", value=1.0)
                            subtalker_loss_weight_pipeline = gr.Number(
                                label="Subtalker loss weight",
                                value=0.3,
                            )
                            log_every_n_steps_pipeline = gr.Slider(
                                label="Log every N steps",
                                minimum=1,
                                maximum=200,
                                value=10,
                                step=1,
                            )
                            save_every_n_epochs_pipeline = gr.Slider(
                                label="Save every N epochs",
                                minimum=1,
                                maximum=20,
                                value=1,
                                step=1,
                            )
                            max_steps_pipeline = gr.Number(
                                label="Max steps (0 = unlimited)",
                                value=0,
                                precision=0,
                            )
                            seed_pipeline = gr.Number(label="Random seed", value=42, precision=0)
                        with gr.Row():
                            run_pipeline_button = gr.Button(
                                "Run Full Pipeline",
                                variant="primary",
                            )
                            stop_pipeline_button = gr.Button(
                                "Stop Pipeline",
                                variant="stop",
                            )
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Progress & Logs</div>')
                        pipeline_status = gr.Markdown()
                        pipeline_progress = gr.Markdown()
                        pipeline_logs = gr.Textbox(
                            label="Pipeline Logs",
                            lines=20,
                            max_lines=30,
                            autoscroll=True,
                            interactive=False,
                        )

            with gr.Tab("6) Inference"):
                infer_checkpoint_choices = list_checkpoint_paths()
                infer_checkpoint_value = (
                    infer_ckpt_default
                    if infer_ckpt_default in infer_checkpoint_choices
                    else _first_or_none(infer_checkpoint_choices)
                )
                infer_device_choices = ["auto", "cuda:0", "mps", "cpu"]
                infer_device_value = (
                    infer_device_default if infer_device_default in infer_device_choices else "auto"
                )
                infer_language_choices = [
                    "auto",
                    "korean",
                    "english",
                    "japanese",
                    "chinese",
                    "spanish",
                    "french",
                    "german",
                    "italian",
                    "portuguese",
                    "russian",
                ]
                infer_language_value = (
                    infer_language_default
                    if infer_language_default in infer_language_choices
                    else "auto"
                )
                infer_speaker_value = str(
                    (infer_cfg.get("speaker_name") if isinstance(infer_cfg, dict) else None)
                    or default_speaker_name
                    or ""
                ).strip()

                def _infer_param_value(name: str) -> Any:
                    value = infer_params.get(name, INFER_DEFAULT_PARAMS.get(name))
                    return INFER_DEFAULT_PARAMS.get(name) if value is None else value

                with gr.Row():
                    with gr.Column(scale=2, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Model & Voice</div>')
                        checkpoint_dropdown_infer = gr.Dropdown(
                            label="Checkpoint path",
                            choices=infer_checkpoint_choices,
                            value=infer_checkpoint_value,
                            allow_custom_value=True,
                        )
                        infer_device = gr.Dropdown(
                            label="Inference device",
                            choices=infer_device_choices,
                            value=infer_device_value,
                        )
                        infer_speaker_choice = gr.Dropdown(
                            label="Speaker Preset / Recent",
                            choices=list_speaker_choices(),
                            value=None,
                            info="Convenience picker. This sets Speaker Name below.",
                        )
                        infer_speaker_name = gr.Textbox(
                            label="Speaker Name",
                            value=infer_speaker_value,
                            placeholder="e.g. my_speaker",
                            info="Must match the speaker_name used during training for fine-tuned checkpoints.",
                        )
                        infer_language = gr.Dropdown(
                            label="Language",
                            choices=infer_language_choices,
                            value=infer_language_value,
                        )
                        infer_instruct = gr.Textbox(
                            label="Instruct (optional style/control prompt)",
                            value=infer_instruct_default,
                            placeholder="e.g. calm and warm narration style",
                        )

                    with gr.Column(scale=3, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Generate</div>')
                        infer_status = gr.Markdown()
                        single_text = gr.Textbox(
                            label="Single Generation Text",
                            lines=5,
                            value="안녕하세요. 파인튜닝된 Qwen3-TTS 모델로 생성한 음성입니다.",
                        )
                        single_generate_button = gr.Button(
                            "Generate Single",
                            variant="primary",
                            elem_classes=["generate-btn"],
                        )
                        single_audio_output = gr.Audio(label="Single Audio", type="filepath")

                        batch_text = gr.Textbox(
                            label="Batch Generation (one line = one utterance)",
                            lines=8,
                            value="첫 번째 문장입니다.\n두 번째 문장입니다.\n세 번째 문장입니다.",
                        )
                        batch_generate_button = gr.Button("Generate Batch")
                        batch_preview_audio = gr.Audio(
                            label="Batch Preview (first wav)", type="filepath"
                        )
                        batch_zip_output = gr.File(label="Batch ZIP")
                        unload_model_button = gr.Button("Unload Inference Model Cache")

                    with gr.Column(scale=1, elem_classes=["params-panel", "compact-params-panel"]):
                        gr.HTML('<div class="section-header">Parameters</div>')
                        infer_save_indicator = gr.HTML(
                            value="<span class='save-indicator'>Settings saved</span>"
                        )
                        with gr.Column(elem_classes=["preset-section"]):
                            gr.HTML(
                                '<div style="font-size:0.75rem;font-weight:600;color:var(--gray-700);margin-bottom:0.5rem;">Quick Presets</div>'
                            )
                            with gr.Row(elem_classes=["preset-btn-group"]):
                                preset_fast = gr.Button(
                                    "Fast", size="sm", elem_classes=["preset-btn-lg"]
                                )
                                preset_balanced = gr.Button(
                                    "Balanced", size="sm", elem_classes=["preset-btn-lg"]
                                )
                                preset_quality = gr.Button(
                                    "Quality", size="sm", elem_classes=["preset-btn-lg"]
                                )
                            preset_reset = gr.Button(
                                "Reset",
                                size="sm",
                                variant="secondary",
                            )

                        temperature = gr.Slider(
                            label="temperature",
                            minimum=0.1,
                            maximum=1.5,
                            value=min(1.5, max(0.1, float(_infer_param_value("temperature")))),
                            step=0.05,
                            info=INFER_PARAM_TOOLTIPS["temperature"],
                        )
                        top_k = gr.Slider(
                            label="top_k",
                            minimum=1,
                            maximum=200,
                            value=min(200, max(1, int(_infer_param_value("top_k")))),
                            step=1,
                            info=INFER_PARAM_TOOLTIPS["top_k"],
                        )
                        top_p = gr.Slider(
                            label="top_p",
                            minimum=0.1,
                            maximum=1.0,
                            value=min(1.0, max(0.1, float(_infer_param_value("top_p")))),
                            step=0.05,
                            info=INFER_PARAM_TOOLTIPS["top_p"],
                        )
                        repetition_penalty = gr.Slider(
                            label="repetition_penalty",
                            minimum=0.8,
                            maximum=1.5,
                            value=min(
                                1.5,
                                max(0.8, float(_infer_param_value("repetition_penalty"))),
                            ),
                            step=0.01,
                            info=INFER_PARAM_TOOLTIPS["repetition_penalty"],
                        )
                        max_new_tokens = gr.Slider(
                            label="max_new_tokens",
                            minimum=64,
                            maximum=4096,
                            value=min(
                                4096,
                                max(64, int(_infer_param_value("max_new_tokens"))),
                            ),
                            step=32,
                            info=INFER_PARAM_TOOLTIPS["max_new_tokens"],
                        )
                        subtalker_temperature = gr.Slider(
                            label="subtalker_temperature",
                            minimum=0.1,
                            maximum=1.5,
                            value=min(
                                1.5,
                                max(0.1, float(_infer_param_value("subtalker_temperature"))),
                            ),
                            step=0.05,
                            info=INFER_PARAM_TOOLTIPS["subtalker_temperature"],
                        )
                        subtalker_top_k = gr.Slider(
                            label="subtalker_top_k",
                            minimum=1,
                            maximum=200,
                            value=min(
                                200,
                                max(1, int(_infer_param_value("subtalker_top_k"))),
                            ),
                            step=1,
                            info=INFER_PARAM_TOOLTIPS["subtalker_top_k"],
                        )
                        subtalker_top_p = gr.Slider(
                            label="subtalker_top_p",
                            minimum=0.1,
                            maximum=1.0,
                            value=min(
                                1.0,
                                max(0.1, float(_infer_param_value("subtalker_top_p"))),
                            ),
                            step=0.05,
                            info=INFER_PARAM_TOOLTIPS["subtalker_top_p"],
                        )

            with gr.Tab("7) Runs & Export"):
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Run Registry</div>')
                        run_table = gr.Dataframe(
                            label="Run Registry",
                            headers=RUN_TABLE_HEADERS,
                            datatype=["str"] * len(RUN_TABLE_HEADERS),
                            value=ui_refresh_run_table(),
                            interactive=False,
                            wrap=True,
                        )
                        refresh_run_table_button = gr.Button("Refresh Run Registry")
                    with gr.Column(scale=1, elem_classes=["params-panel"]):
                        gr.HTML('<div class="section-header">Run Summary</div>')
                        run_dropdown_manage = gr.Dropdown(
                            label="Run path",
                            choices=list_run_paths(),
                            value=_first_or_none(list_run_paths()),
                        )
                        show_run_summary_button = gr.Button("Show Run Summary")
                        run_summary_view = gr.Markdown()
                        gr.HTML('<div class="section-header" style="margin-top:1rem;">Export</div>')
                        export_ckpt_dropdown = gr.Dropdown(
                            label="Checkpoint path",
                            choices=list_checkpoint_paths(),
                            value=_first_or_none(list_checkpoint_paths()),
                        )
                        include_optimizer_files = gr.Checkbox(
                            label="Include optimizer/rng files",
                            value=True,
                        )
                        export_checkpoint_button = gr.Button("Export Checkpoint ZIP")
                        export_status = gr.Markdown()
                        export_zip_file = gr.File(label="Exported ZIP")

            with gr.Tab("8) Workspace"):
                with gr.Column(elem_classes=["params-panel"]):
                    gr.HTML('<div class="section-header">Workspace</div>')
                    workspace_overview = gr.Markdown(value=ui_workspace_overview())
                    with gr.Row():
                        refresh_workspace_button = gr.Button("Refresh Workspace Overview")
                        env_check_button = gr.Button("Run Environment Check")
                    env_check_view = gr.Markdown()

        build_dataset_button.click(
            fn=ui_build_dataset,
            inputs=[dataset_name, uploaded_audios, transcript_file, reference_audio_file],
            outputs=[
                dataset_status,
                dataset_stats_view,
                dataset_preview,
                inspect_raw_path,
                quality_raw_jsonl,
                normalize_raw_jsonl,
                prepare_raw_jsonl,
                pipeline_raw_jsonl,
            ],
        )

        import_raw_button.click(
            fn=ui_import_raw_jsonl,
            inputs=[dataset_name, import_raw_jsonl_file],
            outputs=[
                dataset_status,
                dataset_stats_view,
                dataset_preview,
                inspect_raw_path,
                quality_raw_jsonl,
                normalize_raw_jsonl,
                prepare_raw_jsonl,
                pipeline_raw_jsonl,
            ],
        )

        inspect_button.click(
            fn=ui_inspect_raw,
            inputs=[inspect_raw_path],
            outputs=[dataset_stats_view, dataset_preview],
        )

        validate_button.click(
            fn=ui_validate_dataset_for_plan,
            inputs=[quality_raw_jsonl],
            outputs=[
                quality_report_view,
                quality_report_file,
                batch_size,
                learning_rate,
                num_epochs,
                pipeline_batch_size,
                pipeline_learning_rate,
                pipeline_num_epochs,
                preflight_batch_size,
                preflight_num_epochs,
            ],
        )

        preflight_button.click(
            fn=ui_run_preflight_review,
            inputs=[
                quality_raw_jsonl,
                preflight_init_model_path,
                preflight_device,
                preflight_batch_size,
                preflight_num_epochs,
            ],
            outputs=[
                preflight_report_view,
                preflight_report_file,
                train_preflight_report_file,
                pipeline_preflight_report_file,
            ],
        )

        normalize_button.click(
            fn=ui_normalize_dataset,
            inputs=[normalize_raw_jsonl, normalized_dataset_name, normalize_target_sr, normalize_peak],
            outputs=[
                normalize_status,
                inspect_raw_path,
                quality_raw_jsonl,
                normalize_raw_jsonl,
                prepare_raw_jsonl,
                pipeline_raw_jsonl,
            ],
        )

        run_prepare_button.click(
            fn=ui_run_prepare,
            inputs=[
                prepare_raw_jsonl,
                prepare_device,
                tokenizer_model_path,
                output_filename,
                prepare_batch_infer_num,
            ],
            outputs=[prepare_status, prepare_logs, prepared_jsonl_dropdown, train_jsonl_dropdown],
        )

        stop_prepare_button.click(
            fn=ui_stop_prepare,
            inputs=[],
            outputs=[prepare_status],
            queue=False,
        )

        new_run_name_button.click(
            fn=ui_make_new_run_name,
            inputs=[],
            outputs=[run_name],
            queue=False,
        )

        pipeline_new_run_name.click(
            fn=ui_make_new_run_name,
            inputs=[],
            outputs=[pipeline_run_name],
            queue=False,
        )

        run_train_button.click(
            fn=ui_run_training,
            inputs=[
                init_model_path,
                train_jsonl_dropdown,
                run_name,
                batch_size,
                learning_rate,
                num_epochs,
                speaker_name,
                require_preflight_train,
                train_preflight_report_file,
                speaker_id_train,
                gradient_accumulation_steps_train,
                mixed_precision_train,
                weight_decay_train,
                max_grad_norm_train,
                subtalker_loss_weight_train,
                attn_implementation_train,
                torch_dtype_train,
                log_every_n_steps_train,
                save_every_n_epochs_train,
                max_steps_train,
                seed_train,
            ],
            outputs=[
                train_status,
                train_progress,
                train_logs,
                run_dropdown,
                run_dropdown_manage,
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
        ).then(
            fn=ui_refresh_run_table,
            inputs=[],
            outputs=[run_table],
            queue=False,
        )

        stop_train_button.click(
            fn=ui_stop_training,
            inputs=[],
            outputs=[train_status],
            queue=False,
        )

        run_dropdown.change(
            fn=ui_load_run_checkpoints,
            inputs=[run_dropdown],
            outputs=[
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
            queue=False,
        )

        run_pipeline_button.click(
            fn=ui_run_pipeline,
            inputs=[
                pipeline_raw_jsonl,
                pipeline_prepare_device,
                pipeline_tokenizer_model_path,
                pipeline_output_filename,
                pipeline_prepare_batch_infer_num,
                pipeline_init_model_path,
                pipeline_run_name,
                pipeline_batch_size,
                pipeline_learning_rate,
                pipeline_num_epochs,
                pipeline_speaker_name,
                require_preflight_pipeline,
                pipeline_preflight_report_file,
                speaker_id_pipeline,
                gradient_accumulation_steps_pipeline,
                mixed_precision_pipeline,
                weight_decay_pipeline,
                max_grad_norm_pipeline,
                subtalker_loss_weight_pipeline,
                attn_implementation_pipeline,
                torch_dtype_pipeline,
                log_every_n_steps_pipeline,
                save_every_n_epochs_pipeline,
                max_steps_pipeline,
                seed_pipeline,
            ],
            outputs=[
                pipeline_status,
                pipeline_progress,
                pipeline_logs,
                prepared_jsonl_dropdown,
                train_jsonl_dropdown,
                run_dropdown,
                run_dropdown_manage,
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
        ).then(
            fn=ui_refresh_run_table,
            inputs=[],
            outputs=[run_table],
            queue=False,
        )

        stop_pipeline_button.click(
            fn=ui_stop_pipeline,
            inputs=[],
            outputs=[pipeline_status],
            queue=False,
        )

        checkpoint_dropdown_infer.change(
            fn=ui_infer_on_checkpoint_change,
            inputs=[checkpoint_dropdown_infer, infer_speaker_name],
            outputs=[infer_speaker_name, infer_save_indicator],
            queue=False,
        )

        infer_speaker_choice.change(
            fn=ui_infer_select_speaker_choice,
            inputs=[infer_speaker_choice],
            outputs=[infer_speaker_name, infer_save_indicator],
            queue=False,
        )

        preset_fast.click(
            fn=lambda: ui_infer_apply_preset("fast"),
            inputs=[],
            outputs=[
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                infer_save_indicator,
            ],
            queue=False,
        )
        preset_balanced.click(
            fn=lambda: ui_infer_apply_preset("balanced"),
            inputs=[],
            outputs=[
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                infer_save_indicator,
            ],
            queue=False,
        )
        preset_quality.click(
            fn=lambda: ui_infer_apply_preset("quality"),
            inputs=[],
            outputs=[
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                infer_save_indicator,
            ],
            queue=False,
        )
        preset_reset.click(
            fn=lambda: ui_infer_apply_preset("balanced"),
            inputs=[],
            outputs=[
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
                infer_save_indicator,
            ],
            queue=False,
        )

        for slider in [
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_new_tokens,
            subtalker_temperature,
            subtalker_top_k,
            subtalker_top_p,
        ]:
            slider.change(
                fn=ui_infer_on_params_change,
                inputs=[
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    max_new_tokens,
                    subtalker_temperature,
                    subtalker_top_k,
                    subtalker_top_p,
                ],
                outputs=[infer_save_indicator],
                show_progress="hidden",
                queue=False,
            )

        single_generate_button.click(
            fn=ui_generate_single,
            inputs=[
                checkpoint_dropdown_infer,
                infer_device,
                infer_speaker_name,
                infer_language,
                infer_instruct,
                single_text,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
            ],
            outputs=[infer_status, single_audio_output],
        )

        batch_generate_button.click(
            fn=ui_generate_batch,
            inputs=[
                checkpoint_dropdown_infer,
                infer_device,
                infer_speaker_name,
                infer_language,
                infer_instruct,
                batch_text,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                max_new_tokens,
                subtalker_temperature,
                subtalker_top_k,
                subtalker_top_p,
            ],
            outputs=[infer_status, batch_preview_audio, batch_zip_output],
        )

        unload_model_button.click(
            fn=ui_unload_infer_model,
            inputs=[],
            outputs=[infer_status],
            queue=False,
        )

        refresh_run_table_button.click(
            fn=ui_refresh_run_table,
            inputs=[],
            outputs=[run_table],
            queue=False,
        )

        show_run_summary_button.click(
            fn=ui_show_run_summary,
            inputs=[run_dropdown_manage],
            outputs=[run_summary_view],
            queue=False,
        )

        run_dropdown_manage.change(
            fn=ui_show_run_summary,
            inputs=[run_dropdown_manage],
            outputs=[run_summary_view],
            queue=False,
        )

        run_dropdown_manage.change(
            fn=ui_load_run_checkpoints,
            inputs=[run_dropdown_manage],
            outputs=[
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
            queue=False,
        )

        export_checkpoint_button.click(
            fn=ui_export_checkpoint,
            inputs=[export_ckpt_dropdown, include_optimizer_files],
            outputs=[export_status, export_zip_file],
        )

        refresh_workspace_button.click(
            fn=ui_workspace_overview,
            inputs=[],
            outputs=[workspace_overview],
            queue=False,
        )

        env_check_button.click(
            fn=ui_environment_check,
            inputs=[],
            outputs=[env_check_view],
            queue=False,
        )

        refresh_all_button.click(
            fn=ui_refresh_all_dropdowns,
            inputs=[
                inspect_raw_path,
                quality_raw_jsonl,
                normalize_raw_jsonl,
                prepare_raw_jsonl,
                prepared_jsonl_dropdown,
                train_jsonl_dropdown,
                pipeline_raw_jsonl,
                run_dropdown,
                run_dropdown_manage,
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
            outputs=[
                inspect_raw_path,
                quality_raw_jsonl,
                normalize_raw_jsonl,
                prepare_raw_jsonl,
                prepared_jsonl_dropdown,
                train_jsonl_dropdown,
                pipeline_raw_jsonl,
                run_dropdown,
                run_dropdown_manage,
                checkpoint_dropdown_train,
                checkpoint_dropdown_infer,
                export_ckpt_dropdown,
            ],
            queue=False,
        ).then(
            fn=ui_refresh_run_table,
            inputs=[],
            outputs=[run_table],
            queue=False,
        ).then(
            fn=ui_workspace_overview,
            inputs=[],
            outputs=[workspace_overview],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    server_name = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    app.queue(default_concurrency_limit=8).launch(
        server_name=server_name,
        server_port=server_port,
    )
