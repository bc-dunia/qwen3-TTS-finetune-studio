from __future__ import annotations

import os
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .paths import EXPORTS_DIR, ensure_workspace_dirs


@dataclass
class LoadedModel:
    model_path: str
    device: str
    model: Any


_MODEL_CACHE: LoadedModel | None = None
MIN_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN_TTS_MIN_NEW_TOKENS", "60"))


def _resolve_model_path(model_path: str) -> str:
    raw = (model_path or "").strip()
    if not raw:
        raise ValueError("model_path is required.")
    p = Path(raw).expanduser()
    is_local_style = raw.startswith(("/", "./", "../", "~"))
    if p.exists():
        return str(p.resolve())
    if is_local_style:
        raise FileNotFoundError(f"Model path not found: {p}")

    # HuggingFace repo id (or other hub identifier).
    try:
        from huggingface_hub import snapshot_download

        try:
            return str(Path(snapshot_download(repo_id=raw)).resolve())
        except Exception:
            # Public repos can fail if a bad/expired token is configured in env.
            return str(Path(snapshot_download(repo_id=raw, token=False)).resolve())
    except Exception:
        # Fall back to letting from_pretrained handle it.
        return raw


def _patch_generate_min_tokens(model: Any, min_new_tokens: int = MIN_NEW_TOKENS_DEFAULT) -> Any:
    try:
        generate_fn = model.model.generate
    except Exception:
        return model

    def _patched_generate(*args, **kwargs):
        try:
            if "min_new_tokens" not in kwargs or int(kwargs.get("min_new_tokens", 0)) < int(
                min_new_tokens
            ):
                kwargs["min_new_tokens"] = int(min_new_tokens)
        except Exception:
            kwargs["min_new_tokens"] = int(min_new_tokens)
        return generate_fn(*args, **kwargs)

    try:
        model.model.generate = _patched_generate
    except Exception:
        return model
    return model


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> Any:
    import torch

    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def _load_model(model_path: str, device: str) -> tuple[Any, str]:
    global _MODEL_CACHE

    resolved_path = _resolve_model_path(model_path)
    resolved_device = _resolve_device(device)

    if (
        _MODEL_CACHE is not None
        and _MODEL_CACHE.model_path == resolved_path
        and _MODEL_CACHE.device == resolved_device
    ):
        return _MODEL_CACHE.model, resolved_device

    from qwen_tts import Qwen3TTSModel

    dtype = _resolve_dtype(resolved_device)
    attn = "flash_attention_2" if resolved_device.startswith("cuda") else None

    candidates = [
        {
            "device_map": resolved_device,
            "dtype": dtype,
            "attn_implementation": attn,
        },
        {
            "device_map": resolved_device,
            "torch_dtype": dtype,
            "attn_implementation": attn,
        },
        {"device_map": resolved_device, "dtype": dtype},
        {"device_map": resolved_device},
    ]

    last_error: Exception | None = None
    model = None
    for kwargs in candidates:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            model = Qwen3TTSModel.from_pretrained(resolved_path, **kwargs)
            break
        except Exception as e:
            last_error = e
            continue

    if model is None:
        if last_error:
            raise RuntimeError(f"Failed to load model: {last_error}") from last_error
        raise RuntimeError("Failed to load model.")

    try:
        model.model.eval()
    except Exception:
        pass
    _patch_generate_min_tokens(model)

    _MODEL_CACHE = LoadedModel(
        model_path=resolved_path,
        device=resolved_device,
        model=model,
    )
    return model, resolved_device


def unload_model() -> str:
    global _MODEL_CACHE
    _MODEL_CACHE = None
    try:
        import gc
        import torch

        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return "Model cache cleared."


def _generate_audio(
    model: Any,
    text: str,
    speaker: str,
    params: dict[str, Any],
    *,
    language: str | None = None,
    instruct: str | None = None,
) -> tuple[Any, int]:
    kwargs = {k: v for k, v in params.items() if v is not None}

    lang = (language or "").strip()
    lang = None if not lang or lang.lower() == "auto" else lang
    inst = (instruct or "").strip() or None

    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=lang,
            instruct=inst,
            **kwargs,
        )
        return wavs[0], sr
    except RuntimeError as e:
        msg = str(e)
        if "probability tensor contains either" in msg and ("nan" in msg.lower() or "inf" in msg.lower()):
            raise RuntimeError(
                f"{msg}\n"
                "Hint: this often happens due to unstable sampling on some devices/dtypes. "
                "Try `device=cpu`, reduce `temperature`, or use a smaller `max_new_tokens`."
            ) from e
        raise
    except TypeError:
        pass

    fallback_keys = ["temperature", "top_k", "top_p", "repetition_penalty", "max_new_tokens"]
    fallback_kwargs = {k: kwargs[k] for k in fallback_keys if k in kwargs}
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=lang,
            instruct=inst,
            **fallback_kwargs,
        )
        return wavs[0], sr
    except RuntimeError as e:
        msg = str(e)
        if "probability tensor contains either" in msg and ("nan" in msg.lower() or "inf" in msg.lower()):
            raise RuntimeError(
                f"{msg}\n"
                "Hint: this often happens due to unstable sampling on some devices/dtypes. "
                "Try `device=cpu`, reduce `temperature`, or use a smaller `max_new_tokens`."
            ) from e
        raise
    except TypeError:
        wavs, sr = model.generate_custom_voice(text=text, speaker=speaker, language=lang, instruct=inst)
        return wavs[0], sr


def synthesize_single(
    *,
    checkpoint_path: str,
    device: str,
    speaker_name: str,
    text: str,
    params: dict[str, Any],
    language: str | None = None,
    instruct: str | None = None,
) -> tuple[str | None, str]:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required.")
    if not text.strip():
        raise ValueError("Input text is required.")
    final_speaker = speaker_name.strip()
    if not final_speaker:
        raise ValueError("Speaker name is required.")

    ensure_workspace_dirs()
    model, resolved_device = _load_model(checkpoint_path, device)

    audio, sr = _generate_audio(
        model=model,
        text=text.strip(),
        speaker=final_speaker,
        params=params,
        language=language,
        instruct=instruct,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (EXPORTS_DIR / "single").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"single_{ts}.wav"
    sf.write(output_path, audio, sr)
    message = f"Generated with device={resolved_device}: {output_path}"
    return str(output_path), message


def synthesize_batch(
    *,
    checkpoint_path: str,
    device: str,
    speaker_name: str,
    multiline_text: str,
    params: dict[str, Any],
    language: str | None = None,
    instruct: str | None = None,
) -> tuple[str | None, str | None, str]:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required.")
    final_speaker = speaker_name.strip()
    if not final_speaker:
        raise ValueError("Speaker name is required.")

    lines = [line.strip() for line in multiline_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Batch text is empty. Add one sentence per line.")

    ensure_workspace_dirs()
    model, resolved_device = _load_model(checkpoint_path, device)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (EXPORTS_DIR / f"batch_{ts}").resolve()
    wav_dir = batch_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_path = batch_dir / "metadata.jsonl"

    first_audio: str | None = None
    with meta_path.open("w", encoding="utf-8") as meta_file:
        for i, line in enumerate(lines, start=1):
            wav, sr = _generate_audio(
                model=model,
                text=line,
                speaker=final_speaker,
                params=params,
                language=language,
                instruct=instruct,
            )
            out_path = wav_dir / f"{i:03d}.wav"
            sf.write(out_path, wav, sr)
            if first_audio is None:
                first_audio = str(out_path)
            payload = {
                "index": i,
                "text": line,
                "audio": str(out_path),
                "sample_rate": sr,
                "checkpoint": str(Path(checkpoint_path).resolve()),
                "speaker": final_speaker,
                "language": language,
                "instruct": instruct,
            }
            meta_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    archive_base = batch_dir.parent / batch_dir.name
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=batch_dir)
    message = (
        f"Generated {len(lines)} files with device={resolved_device}. "
        f"Archive: {zip_path}"
    )
    return first_audio, zip_path, message
