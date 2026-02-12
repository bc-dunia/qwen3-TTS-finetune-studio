#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Ensure repo root is on sys.path when running `python scripts/e2e_smoke.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune_studio.dataset_ops import build_dataset_from_uploads
from finetune_studio.inference_ops import synthesize_single, unload_model
from finetune_studio.paths import EXPORTS_DIR, WORKSPACE_ROOT
from finetune_studio.quality import run_preflight_review, validate_dataset
from finetune_studio.training_ops import run_prepare_data, run_training


def _make_sine_wav(path: Path, *, sr: int, seconds: float, freq_hz: float) -> None:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    wav = 0.15 * np.sin(2.0 * np.pi * freq_hz * t)
    sf.write(path, wav, sr)


def _resolve_auto_device(*, allow_mps: bool = True) -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _is_local_style(spec: str) -> bool:
    raw = (spec or "").strip()
    return raw.startswith(("/", "./", "../", "~"))


def main() -> int:
    keep = os.environ.get("KEEP_E2E_ARTIFACTS", "").strip() == "1"
    allow_hf_download = os.environ.get("E2E_ALLOW_HF_DOWNLOAD", "").strip() == "1"
    exit_code = 0

    # Prefer local models if present (matches typical qwen3-tts-studio setup), but avoid
    # hardcoding any absolute paths.
    base_models_root = None
    env_root = os.environ.get("QWEN3_TTS_MODELS_ROOT", "").strip()
    if env_root:
        base_models_root = Path(env_root).expanduser().resolve()
    else:
        sibling_root = (PROJECT_ROOT.parent / "qwen3-tts-studio" / "qwen3-TTS-studio").resolve()
        if sibling_root.exists():
            base_models_root = sibling_root

    init_model_spec = os.environ.get("E2E_INIT_MODEL_PATH", "").strip()
    tokenizer_spec = os.environ.get("E2E_TOKENIZER_MODEL_PATH", "").strip()

    if not init_model_spec:
        if base_models_root is not None:
            init_model_spec = str((base_models_root / "Qwen3-TTS-12Hz-0.6B-Base").resolve())
        else:
            init_model_spec = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if not tokenizer_spec:
        if base_models_root is not None:
            tokenizer_spec = str((base_models_root / "Qwen3-TTS-Tokenizer-12Hz").resolve())
        else:
            tokenizer_spec = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

    if _is_local_style(init_model_spec):
        init_model_path = Path(init_model_spec).expanduser().resolve()
        if not init_model_path.exists():
            print(f"[E2E] Missing init model dir: {init_model_path}")
            return 2
        init_model_spec = str(init_model_path)
    elif not allow_hf_download and base_models_root is None:
        print(
            "[E2E] No local init model found. "
            "Set QWEN3_TTS_MODELS_ROOT or E2E_INIT_MODEL_PATH, "
            "or set E2E_ALLOW_HF_DOWNLOAD=1 to allow downloading from HuggingFace."
        )
        return 2

    if _is_local_style(tokenizer_spec):
        tokenizer_model_path = Path(tokenizer_spec).expanduser().resolve()
        if not tokenizer_model_path.exists():
            print(f"[E2E] Missing tokenizer dir: {tokenizer_model_path}")
            return 2
        tokenizer_spec = str(tokenizer_model_path)
    elif not allow_hf_download and base_models_root is None:
        print(
            "[E2E] No local tokenizer found. "
            "Set QWEN3_TTS_MODELS_ROOT or E2E_TOKENIZER_MODEL_PATH, "
            "or set E2E_ALLOW_HF_DOWNLOAD=1 to allow downloading from HuggingFace."
        )
        return 2

    ts = time.strftime("%Y%m%d_%H%M%S")
    tmp_src = (WORKSPACE_ROOT / "_e2e_tmp" / ts).resolve()
    tmp_src.mkdir(parents=True, exist_ok=True)
    print(f"[E2E] tmp dir: {tmp_src}")

    # Create tiny synthetic dataset inputs.
    sr = 24000
    audio1 = tmp_src / "utt_0001.wav"
    audio2 = tmp_src / "utt_0002.wav"
    ref = tmp_src / "ref.wav"
    _make_sine_wav(audio1, sr=sr, seconds=2.2, freq_hz=220.0)
    _make_sine_wav(audio2, sr=sr, seconds=2.1, freq_hz=330.0)
    _make_sine_wav(ref, sr=sr, seconds=3.0, freq_hz=110.0)

    transcript = tmp_src / "transcript.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"audio": audio1.name, "text": "안녕하세요."}, ensure_ascii=False),
                json.dumps(
                    {"audio": audio2.name, "text": "이 문장은 E2E 스모크 테스트입니다."},
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset_name = f"e2e_{ts}"
    run_name = f"e2e_run_{ts}"
    speaker_name = f"e2e_speaker_{ts}"
    # Use lr=0 by default to avoid NaNs on tiny synthetic audio; override if needed.
    learning_rate = float(os.environ.get("E2E_LR", "0.0"))

    prepare_device = os.environ.get("E2E_PREPARE_DEVICE", "auto").strip() or "auto"
    if prepare_device.lower() == "auto":
        prepare_device = _resolve_auto_device(allow_mps=True)

    # Many Mac setups (MPS + fp16) can hit NaNs in sampling for fine-tuned checkpoints.
    # Default to CPU unless CUDA is available, but allow explicit override via env.
    infer_device = os.environ.get("E2E_INFER_DEVICE", "").strip()
    if not infer_device or infer_device.lower() == "auto":
        auto_infer = _resolve_auto_device(allow_mps=False)
        infer_device = auto_infer

    dataset_dir: Path | None = None
    raw_jsonl: Path | None = None
    run_dir: Path | None = None
    try:
        dataset_dir, raw_jsonl = build_dataset_from_uploads(
            dataset_name=dataset_name,
            uploaded_audios=[str(audio1), str(audio2), str(ref)],
            transcript_file=str(transcript),
            reference_audio_file=str(ref),
        )
        print(f"[E2E] dataset_dir: {dataset_dir}")
        print(f"[E2E] raw_jsonl: {raw_jsonl}")

        report = validate_dataset(str(raw_jsonl))
        errors = int(report.get("errors_count", 0))
        print(f"[E2E] validate errors: {errors}")
        if errors != 0:
            print("[E2E] Dataset validation failed. Aborting.")
            exit_code = 3
            return exit_code

        preflight = run_preflight_review(
            raw_jsonl_path=str(raw_jsonl),
            init_model_path=str(init_model_spec),
            prepare_device=prepare_device,
            batch_size=1,
            num_epochs=1,
        )
        print(f"[E2E] preflight decision: {preflight.get('decision')}")
        if str(preflight.get("decision", "")).strip().lower() == "no-go":
            print("[E2E] Preflight returned NO-GO. Aborting.")
            exit_code = 4
            return exit_code

        coded_jsonl = None
        for event in run_prepare_data(
            device=prepare_device,
            tokenizer_model_path=str(tokenizer_spec),
            input_jsonl=str(raw_jsonl),
            output_filename="train_with_codes.jsonl",
            batch_infer_num=2,
        ):
            if event.get("done"):
                if not event.get("success"):
                    print("[E2E] prepare_data failed.")
                    exit_code = 5
                    return exit_code
                coded_jsonl = event.get("output_jsonl")
                break
        if not coded_jsonl:
            print("[E2E] prepare_data produced no output.")
            exit_code = 5
            return exit_code
        coded_path = Path(str(coded_jsonl)).resolve()
        print(f"[E2E] coded jsonl: {coded_path}")
        if not coded_path.exists():
            print("[E2E] coded jsonl missing on disk.")
            exit_code = 5
            return exit_code

        last_checkpoint = ""
        for event in run_training(
            init_model_path=str(init_model_spec),
            train_jsonl=str(coded_path),
            run_name=run_name,
            batch_size=1,
            learning_rate=learning_rate,
            num_epochs=1,
            speaker_name=speaker_name,
            speaker_id=3000,
            gradient_accumulation_steps=1,
            mixed_precision="auto",
            weight_decay=0.01,
            max_grad_norm=1.0,
            subtalker_loss_weight=0.3,
            attn_implementation="auto",
            torch_dtype="auto",
            log_every_n_steps=1,
            save_every_n_epochs=1,
            max_steps=1,
            random_seed=42,
        ):
            run_dir_raw = event.get("run_dir") or ""
            if run_dir_raw:
                run_dir = Path(str(run_dir_raw)).resolve()
            if event.get("done"):
                if not event.get("success"):
                    print("[E2E] training failed.")
                    exit_code = 6
                    return exit_code
                last_checkpoint = str(event.get("last_checkpoint") or "").strip()
                break

        if not last_checkpoint:
            print("[E2E] training produced no checkpoint.")
            exit_code = 6
            return exit_code

        ckpt_path = Path(last_checkpoint).resolve()
        print(f"[E2E] last checkpoint: {ckpt_path}")
        if not ckpt_path.exists():
            print("[E2E] checkpoint path missing on disk.")
            exit_code = 6
            return exit_code

        wav_path, status = synthesize_single(
            checkpoint_path=str(ckpt_path),
            device=infer_device,
            speaker_name=speaker_name,
            text="안녕하세요. E2E 스모크 테스트 음성입니다.",
            language="korean",
            instruct="calm and warm narration style",
            params={
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 1.0,
                "repetition_penalty": 1.05,
                "max_new_tokens": 512,
                "subtalker_temperature": 0.9,
                "subtalker_top_k": 50,
                "subtalker_top_p": 1.0,
            },
        )
        print(f"[E2E] inference status: {status}")
        if not wav_path or not Path(wav_path).exists():
            print("[E2E] inference produced no wav.")
            exit_code = 7
            return exit_code

        print(f"[E2E] OK: generated wav: {wav_path}")
        print(f"[E2E] exports dir: {EXPORTS_DIR}")
        exit_code = 0
        return exit_code
    finally:
        try:
            unload_model()
        except Exception:
            pass

        if keep:
            print("[E2E] KEEP_E2E_ARTIFACTS=1: skipping cleanup.")
        else:
            for p in [tmp_src]:
                try:
                    if p and p.exists():
                        shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass

            # Delete dataset and run outputs to avoid multi-GB checkpoint copies lingering.
            for p in [dataset_dir, run_dir]:
                try:
                    if p and Path(p).exists():
                        shutil.rmtree(Path(p), ignore_errors=True)
                except Exception:
                    pass


if __name__ == "__main__":
    sys.exit(main())
