from __future__ import annotations

import os
import threading
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .paths import EXPORTS_DIR, ensure_workspace_dirs, is_loadable_checkpoint_dir


@dataclass
class LoadedModel:
    model_path: str
    device: str
    model: Any


_MODEL_CACHE: LoadedModel | None = None
_BASE_MODEL_CACHE: LoadedModel | None = None
_CACHE_LOCK = threading.Lock()
MIN_NEW_TOKENS_DEFAULT = int(os.environ.get("QWEN_TTS_MIN_NEW_TOKENS", "0"))
_ADAPTIVE_MAX_TOKENS_DISABLED = (
    os.environ.get("QWEN_TTS_DISABLE_ADAPTIVE_MAX_TOKENS", "0") == "1"
)
_ADAPTIVE_MAX_TOKENS_FLOOR = int(
    os.environ.get("QWEN_TTS_ADAPTIVE_MAX_TOKENS_FLOOR", "192")
)
_ADAPTIVE_MAX_TOKENS_CEIL = int(
    os.environ.get("QWEN_TTS_ADAPTIVE_MAX_TOKENS_CEIL", "1024")
)


def _set_model_cache(value: LoadedModel | None) -> None:
    globals()["_MODEL_CACHE"] = value


def _set_base_model_cache(value: LoadedModel | None) -> None:
    globals()["_BASE_MODEL_CACHE"] = value


def _recommended_max_new_tokens(text: str) -> int:
    chars = max(1, len((text or "").strip()))
    # Keep short prompts from over-generating noisy tails, while allowing longer text to scale.
    scaled = int(chars * 4)
    floor = max(64, int(_ADAPTIVE_MAX_TOKENS_FLOOR))
    ceil = max(floor, int(_ADAPTIVE_MAX_TOKENS_CEIL))
    return max(floor, min(ceil, scaled))


def _normalize_max_new_tokens(text: str, raw_value: Any) -> int | None:
    if raw_value is None:
        return _recommended_max_new_tokens(text)
    try:
        requested = int(raw_value)
    except Exception:
        return _recommended_max_new_tokens(text)
    if requested <= 0:
        return None
    if _ADAPTIVE_MAX_TOKENS_DISABLED:
        return requested
    return min(requested, _recommended_max_new_tokens(text))


def _normalize_max_new_tokens_with_ceiling(
    text: str, raw_value: Any, adaptive_ceil: int
) -> int | None:
    normalized = _normalize_max_new_tokens(text, raw_value)
    if normalized is None:
        return None

    ceil_value = max(64, int(adaptive_ceil))
    if _ADAPTIVE_MAX_TOKENS_DISABLED:
        return min(int(normalized), ceil_value)

    try:
        requested = int(raw_value) if raw_value is not None else None
    except Exception:
        requested = None
    if requested is not None and requested > 0:
        return min(requested, ceil_value)

    chars = max(1, len((text or "").strip()))
    scaled = int(chars * 4)
    floor = max(64, int(_ADAPTIVE_MAX_TOKENS_FLOOR))
    ceil = max(floor, ceil_value)
    return max(floor, min(ceil, scaled))


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


def _patch_generate_min_tokens(
    model: Any, min_new_tokens: int = MIN_NEW_TOKENS_DEFAULT
) -> Any:
    if int(min_new_tokens) <= 0:
        return model
    try:
        generate_fn = model.model.generate
    except Exception:
        return model

    def _patched_generate(*args, **kwargs):
        try:
            if "min_new_tokens" not in kwargs or int(
                kwargs.get("min_new_tokens", 0)
            ) < int(min_new_tokens):
                kwargs["min_new_tokens"] = int(min_new_tokens)
        except Exception:
            kwargs["min_new_tokens"] = int(min_new_tokens)
        return generate_fn(*args, **kwargs)

    try:
        model.model.generate = _patched_generate
    except Exception:
        return model
    return model


def _model_supports_instruct(model: Any) -> bool:
    try:
        raw_size = str(getattr(model.model, "tts_model_size", "") or "")
    except Exception:
        return True

    normalized = "".join(ch for ch in raw_size.lower() if ch.isalnum())
    # qwen-tts 0.6B models do not support instruct in custom_voice mode.
    return normalized not in {"06b", "0b6"}


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


def _try_load_pretrained(resolved_path: str, resolved_device: str, err_prefix: str) -> Any:
    """Load Qwen3TTSModel with 4-candidate fallback. Returns loaded model."""
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
            raise RuntimeError(f"{err_prefix}: {last_error}") from last_error
        raise RuntimeError(f"{err_prefix}.")

    try:
        model.model.eval()
    except Exception:
        pass
    return model


def _load_model(model_path: str, device: str) -> tuple[Any, str]:
    global _MODEL_CACHE

    resolved_path = _resolve_model_path(model_path)
    resolved_device = _resolve_device(device)

    with _CACHE_LOCK:
        if (
            _MODEL_CACHE is not None
            and _MODEL_CACHE.model_path == resolved_path
            and _MODEL_CACHE.device == resolved_device
        ):
            return _MODEL_CACHE.model, resolved_device

        model = _try_load_pretrained(resolved_path, resolved_device, "Failed to load model")
        _patch_generate_min_tokens(model)

        _set_model_cache(
            LoadedModel(
                model_path=resolved_path,
                device=resolved_device,
                model=model,
            )
        )
        return model, resolved_device


def _load_base_model(device: str) -> tuple[Any, str]:
    global _BASE_MODEL_CACHE

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        hub_cache = Path(HF_HUB_CACHE)
    except Exception:
        hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    snapshots_dir = (
        hub_cache
        / "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base"
        / "snapshots"
    )
    base_source = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if snapshots_dir.exists() and snapshots_dir.is_dir():
        candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if candidates:
            latest_snapshot = sorted(
                candidates, key=lambda p: p.stat().st_mtime, reverse=True
            )[0]
            base_source = str(latest_snapshot)

    resolved_path = _resolve_model_path(base_source)
    resolved_device = _resolve_device(device)

    with _CACHE_LOCK:
        if (
            _BASE_MODEL_CACHE is not None
            and _BASE_MODEL_CACHE.model_path == resolved_path
            and _BASE_MODEL_CACHE.device == resolved_device
        ):
            return _BASE_MODEL_CACHE.model, resolved_device

        model = _try_load_pretrained(resolved_path, resolved_device, "Failed to load base model")

        _set_base_model_cache(
            LoadedModel(
                model_path=resolved_path,
                device=resolved_device,
                model=model,
            )
        )
        return model, resolved_device


def _seed_everything(seed: int | None) -> int | None:
    if seed is None:
        return None
    try:
        value = int(seed)
    except Exception:
        return None

    random.seed(value)
    np.random.seed(value)
    try:
        import torch

        torch.manual_seed(value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(value)
    except Exception:
        pass
    return value


def unload_model() -> str:
    with _CACHE_LOCK:
        _set_model_cache(None)
        _set_base_model_cache(None)
    try:
        import gc
        import torch

        gc.collect()
        if (
            torch.backends.mps.is_available()
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "empty_cache")
        ):
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return "Model cache cleared."


def _reraise_nan_inf(e: RuntimeError) -> None:
    """Re-raise with a user-friendly hint if the error is a NaN/Inf sampling failure."""
    msg = str(e)
    if "probability tensor contains either" in msg and (
        "nan" in msg.lower() or "inf" in msg.lower()
    ):
        raise RuntimeError(
            f"{msg}\n"
            "Hint: this often happens due to unstable sampling on some devices/dtypes. "
            "Try `device=cpu`, reduce `temperature`, or use a smaller `max_new_tokens`."
        ) from e



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
    normalized_max_new_tokens = _normalize_max_new_tokens(
        text, kwargs.get("max_new_tokens")
    )
    if normalized_max_new_tokens is None:
        kwargs.pop("max_new_tokens", None)
    else:
        kwargs["max_new_tokens"] = int(normalized_max_new_tokens)

    lang = (language or "").strip()
    lang = None if not lang or lang.lower() == "auto" else lang
    inst = (instruct or "").strip() or None
    if inst and not _model_supports_instruct(model):
        inst = None

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
        _reraise_nan_inf(e)
        raise
    except TypeError:
        pass

    fallback_keys = [
        "temperature",
        "top_k",
        "top_p",
        "repetition_penalty",
        "max_new_tokens",
    ]
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
        _reraise_nan_inf(e)
        raise
    except TypeError:
        wavs, sr = model.generate_custom_voice(
            text=text, speaker=speaker, language=lang, instruct=inst
        )
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
    seed: int | None = None,
) -> tuple[str | None, str]:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required.")
    if not text.strip():
        raise ValueError("Input text is required.")
    final_speaker = speaker_name.strip()
    if not final_speaker:
        raise ValueError("Speaker name is required.")
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.exists() and ckpt.is_dir() and not is_loadable_checkpoint_dir(ckpt):
        raise FileNotFoundError(
            "Checkpoint directory does not contain model weights "
            "(`model.safetensors` / `pytorch_model*.bin`)."
        )

    ensure_workspace_dirs()
    model, resolved_device = _load_model(checkpoint_path, device)
    final_seed = _seed_everything(seed)

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
    seed_note = f", seed={final_seed}" if final_seed is not None else ""
    message = f"Generated with device={resolved_device}{seed_note}: {output_path}"
    return str(output_path), message


def synthesize_voice_clone(
    *,
    device: str,
    text: str,
    ref_audio_path: str,
    ref_text: str,
    use_icl: bool = True,
    params: dict[str, Any],
    language: str | None = None,
    seed: int | None = None,
) -> tuple[str | None, str]:
    final_text = (text or "").strip()
    if not final_text:
        raise ValueError("Input text is required.")

    ref_audio = Path((ref_audio_path or "").strip()).expanduser()
    if not ref_audio.exists() or not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    final_ref_text = (ref_text or "").strip()
    final_use_icl = bool(use_icl)
    if final_use_icl and not final_ref_text:
        raise ValueError("Reference transcript is required when ICL mode is enabled.")

    ensure_workspace_dirs()
    model, resolved_device = _load_base_model(device)
    final_seed = _seed_everything(seed)
    if str(getattr(model.model, "tts_model_type", "") or "").strip().lower() != "base":
        raise RuntimeError(
            "Voice cloning requires the base model (`tts_model_type == 'base'`)."
        )

    gen_kwargs = {k: v for k, v in (params or {}).items() if v is not None}
    normalized_max_new_tokens = _normalize_max_new_tokens_with_ceiling(
        final_text, gen_kwargs.get("max_new_tokens"), 2048
    )
    if normalized_max_new_tokens is None:
        gen_kwargs.pop("max_new_tokens", None)
    else:
        gen_kwargs["max_new_tokens"] = int(min(2048, normalized_max_new_tokens))

    lang = (language or "").strip()
    lang = None if not lang or lang.lower() == "auto" else lang
    gen_kwargs["do_sample"] = True
    gen_kwargs["subtalker_dosample"] = True

    try:
        wavs, sr = model.generate_voice_clone(
            text=final_text,
            language=lang,
            ref_audio=str(ref_audio.resolve()),
            ref_text=final_ref_text if final_use_icl else None,
            x_vector_only_mode=not final_use_icl,
            **gen_kwargs,
        )
        audio = wavs[0]
    except RuntimeError as e:
        _reraise_nan_inf(e)
        raise

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (EXPORTS_DIR / "voice_clone").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"clone_{ts}.wav"
    sf.write(output_path, audio, sr)

    seed_note = f", seed={final_seed}" if final_seed is not None else ""
    mode_note = "icl" if final_use_icl else "x-vector"
    message = f"Generated ({mode_note}) with device={resolved_device}{seed_note}: {output_path}"
    return str(output_path), message


def synthesize_voice_clone_batch(
    *,
    device: str,
    multiline_text: str,
    ref_audio_path: str,
    ref_text: str,
    use_icl: bool = True,
    params: dict[str, Any],
    language: str | None = None,
    seed: int | None = None,
) -> tuple[str | None, str | None, str]:
    """Batch voice-clone generation: one wav per line, packaged as ZIP."""
    ref_audio = Path((ref_audio_path or "").strip()).expanduser()
    if not ref_audio.exists() or not ref_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    final_ref_text = (ref_text or "").strip()
    final_use_icl = bool(use_icl)
    if final_use_icl and not final_ref_text:
        raise ValueError("Reference transcript is required when ICL mode is enabled.")

    lines = [line.strip() for line in (multiline_text or "").splitlines() if line.strip()]
    if not lines:
        raise ValueError("Batch text is empty. Add one sentence per line.")

    ensure_workspace_dirs()
    model, resolved_device = _load_base_model(device)
    final_seed = _seed_everything(seed)
    if str(getattr(model.model, "tts_model_type", "") or "").strip().lower() != "base":
        raise RuntimeError(
            "Voice cloning requires the base model (`tts_model_type == 'base')."
        )

    lang = (language or "").strip()
    lang = None if not lang or lang.lower() == "auto" else lang

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = (EXPORTS_DIR / f"batch_clone_{ts}").resolve()
    wav_dir = batch_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta_path = batch_dir / "metadata.jsonl"

    first_audio: str | None = None
    with meta_path.open("w", encoding="utf-8") as meta_file:
        for i, line_text in enumerate(lines, start=1):
            gen_kwargs = {k: v for k, v in (params or {}).items() if v is not None}
            normalized_max_new_tokens = _normalize_max_new_tokens_with_ceiling(
                line_text, gen_kwargs.get("max_new_tokens"), 2048
            )
            if normalized_max_new_tokens is None:
                gen_kwargs.pop("max_new_tokens", None)
            else:
                gen_kwargs["max_new_tokens"] = int(min(2048, normalized_max_new_tokens))
            gen_kwargs["do_sample"] = True
            gen_kwargs["subtalker_dosample"] = True

            try:
                wavs, sr = model.generate_voice_clone(
                    text=line_text,
                    language=lang,
                    ref_audio=str(ref_audio.resolve()),
                    ref_text=final_ref_text if final_use_icl else None,
                    x_vector_only_mode=not final_use_icl,
                    **gen_kwargs,
                )
                audio = wavs[0]
            except RuntimeError as e:
                _reraise_nan_inf(e)
                raise

            out_path = wav_dir / f"{i:03d}.wav"
            sf.write(out_path, audio, sr)
            if first_audio is None:
                first_audio = str(out_path)
            mode_note = "icl" if final_use_icl else "x-vector"
            payload = {
                "index": i,
                "text": line_text,
                "audio": str(out_path),
                "sample_rate": sr,
                "mode": mode_note,
                "ref_audio": str(ref_audio.resolve()),
                "language": language,
            }
            meta_file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    archive_base = batch_dir.parent / batch_dir.name
    zip_path = shutil.make_archive(str(archive_base), "zip", root_dir=batch_dir)
    seed_note = f", seed={final_seed}" if final_seed is not None else ""
    mode_label = "ICL" if final_use_icl else "x-vector"
    message = (
        f"Generated {len(lines)} files ({mode_label}) with device={resolved_device}{seed_note}. "
        f"Archive: {zip_path}"
    )
    return first_audio, zip_path, message

def synthesize_batch(
    *,
    checkpoint_path: str,
    device: str,
    speaker_name: str,
    multiline_text: str,
    params: dict[str, Any],
    language: str | None = None,
    instruct: str | None = None,
    seed: int | None = None,
) -> tuple[str | None, str | None, str]:
    if not checkpoint_path:
        raise ValueError("Checkpoint path is required.")
    final_speaker = speaker_name.strip()
    if not final_speaker:
        raise ValueError("Speaker name is required.")
    ckpt = Path(checkpoint_path).expanduser()
    if ckpt.exists() and ckpt.is_dir() and not is_loadable_checkpoint_dir(ckpt):
        raise FileNotFoundError(
            "Checkpoint directory does not contain model weights "
            "(`model.safetensors` / `pytorch_model*.bin`)."
        )

    lines = [line.strip() for line in multiline_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Batch text is empty. Add one sentence per line.")

    ensure_workspace_dirs()
    model, resolved_device = _load_model(checkpoint_path, device)
    final_seed = _seed_everything(seed)

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
    seed_note = f", seed={final_seed}" if final_seed is not None else ""
    message = (
        f"Generated {len(lines)} files with device={resolved_device}{seed_note}. "
        f"Archive: {zip_path}"
    )
    return first_audio, zip_path, message
