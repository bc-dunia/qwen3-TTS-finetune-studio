from __future__ import annotations
import base64
import gc
import hashlib
import importlib
import importlib.util
import io
import os
import re
import shutil
import threading
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any
import numpy as np

BASE_MODEL_PATHS = {
    "qwen3-tts-1.7b": "/models/Qwen3-TTS-12Hz-1.7B-Base",
    "qwen3-tts-0.6b": "/models/Qwen3-TTS-12Hz-0.6B-Base",
}
DEFAULT_MODEL_ID = "qwen3-tts-1.7b"
CHECKPOINT_ROOT = Path("/tmp/checkpoints")
MAX_CACHED_MODELS = 2
R2_TIMEOUT_SEC = 600  # 10 min for large full checkpoints (~4GB)
MODEL_CACHE: OrderedDict[str, tuple[Any, Any | None, Path]] = OrderedDict()
CACHE_IDENTITY: dict[str, str] = {}
_MODEL_LOCK = threading.Lock()
_MERGE_LOCKS: dict[str, threading.Lock] = {}
_MERGE_LOCKS_LOCK = threading.Lock()
_ASR_MODELS: dict[str, Any] = {}
_ASR_LOCK = threading.Lock()
_EVAL_MODEL: Any | None = None
_EVAL_MODEL_LOCK = threading.Lock()
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
_NORMALIZED_TEXT_RE = re.compile(r"[^0-9a-zA-Z가-힣]+")
REFERENCE_CACHE_ROOT = Path("/tmp/reference_audio_cache")
LOCAL_ASR_MODEL_DIR = Path("/models/faster-whisper-base")
_WHISPER_LANGUAGE_ALIASES = {
    "auto": None,
    "english": "en",
    "en": "en",
    "en-us": "en",
    "en-gb": "en",
    "korean": "ko",
    "ko": "ko",
    "ko-kr": "ko",
    "japanese": "ja",
    "ja": "ja",
    "ja-jp": "ja",
    "chinese": "zh",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
    "mandarin": "zh",
    "spanish": "es",
    "es": "es",
    "es-es": "es",
    "es-mx": "es",
    "french": "fr",
    "fr": "fr",
    "fr-fr": "fr",
    "german": "de",
    "de": "de",
    "de-de": "de",
    "italian": "it",
    "it": "it",
    "it-it": "it",
    "portuguese": "pt",
    "pt": "pt",
    "pt-br": "pt",
    "pt-pt": "pt",
    "russian": "ru",
    "ru": "ru",
    "ru-ru": "ru",
}


def _log(msg: str) -> None:
    print(f"[runpod-tts] {msg}", flush=True)


def _torch() -> Any:
    return importlib.import_module("torch")


def _runpod() -> Any:
    return importlib.import_module("runpod")


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except Exception:
        return None


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _validate_id(value: str, name: str) -> None:
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"Invalid {name}: must be alphanumeric/dash/dot/underscore, 1-128 chars"
        )


def _r2_storage_cls() -> Any:
    try:
        return importlib.import_module("cloud.runpod.r2_storage").R2Storage
    except Exception:
        path = Path(__file__).with_name("r2_storage.py")
        spec = importlib.util.spec_from_file_location("runpod_r2_storage", path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Failed to load r2_storage.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.R2Storage


def _resolve_device() -> str:
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(device: str) -> Any:
    torch = _torch()
    if device.startswith("cuda"):
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Multi-language script detection for duration estimation
# ---------------------------------------------------------------------------
_HANGUL = re.compile(r"[\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F]")
_HAN = re.compile(r"[\u4E00-\u9FFF]")
_HIRA = re.compile(r"[\u3040-\u309F]")
_KATA = re.compile(r"[\u30A0-\u30FF\uFF66-\uFF9D]")
_LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")
_CJK_PUNCT = set(".,!?;:\u2026\u00b7\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\u30fb")


def _expected_seconds(text: str) -> float:
    """Estimate speech duration in seconds using per-script character rates.

    Rates are additive so mixed-language text (e.g. Korean with English loanwords)
    sums correctly instead of picking a single language.
    """
    t = (text or "").strip()
    if not t:
        return 0.0

    hangul = len(_HANGUL.findall(t))
    han = len(_HAN.findall(t))
    kana = len(_HIRA.findall(t)) + len(_KATA.findall(t))
    latin = len(_LATIN_DIGIT.findall(t))
    punct = sum(1 for ch in t if ch in _CJK_PUNCT)

    # Count characters not matched by any known script (Thai, Vietnamese, Arabic, etc.)
    known_count = hangul + han + kana + latin + punct
    other = sum(1 for ch in t if not ch.isspace()) - known_count
    other = max(0, other)

    sec = 0.0
    sec += hangul * 0.25  # Korean: ~4 chars/sec
    sec += han * 0.30  # Chinese / kanji: ~3.3 chars/sec
    sec += kana * 0.18  # Japanese kana: ~5.5 chars/sec
    sec += latin * 0.10  # English / digits: ~10 chars/sec
    sec += punct * 0.12  # punctuation pauses
    sec += other * 0.20  # unknown scripts: conservative ~5 chars/sec
    return max(0.2, sec)


def _adaptive_max_tokens(text: str) -> int:
    """Estimate max_new_tokens from text length using per-script duration."""
    est = _expected_seconds(text)
    if est <= 0.0:
        return 48  # ~4 second fallback for empty input
    # Convert to tokens: 12Hz * estimated_duration * 1.5x safety buffer
    tokens = int(est * 12 * 1.5)
    floor = 48  # minimum ~4 seconds (prevents truncation on short texts)
    ceil = 1024  # hard upper limit
    return max(floor, min(ceil, tokens))


def _normalize_text(text: str) -> str:
    return _NORMALIZED_TEXT_RE.sub("", (text or "").strip().lower())


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
    return 1.0 - (prev[n] / max(len(a), len(b), 1))


def _asr_similarity(target: str, pred: str) -> float:
    return float(_levenshtein_ratio(_normalize_text(target), _normalize_text(pred)))


def _load_asr_model() -> Any:
    return _load_asr_model_for_device(prefer_cpu=False)


def _load_asr_model_for_device(*, prefer_cpu: bool) -> Any:
    with _ASR_LOCK:
        cache_key = "cpu" if prefer_cpu else "auto"
        if cache_key in _ASR_MODELS:
            return _ASR_MODELS[cache_key]
        WhisperModel = importlib.import_module("faster_whisper").WhisperModel
        model_ref = str(LOCAL_ASR_MODEL_DIR) if LOCAL_ASR_MODEL_DIR.exists() else "base"
        if prefer_cpu:
            device = "cpu"
            compute_type = "int8"
        else:
            device = "cuda" if _torch().cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        try:
            model = WhisperModel(model_ref, device=device, compute_type=compute_type)
        except Exception as exc:
            if device == "cuda" and not prefer_cpu:
                _log(f"asr_model_load_failed device=cuda error={exc}; retrying on cpu")
                model = WhisperModel(model_ref, device="cpu", compute_type="int8")
                cache_key = "cpu"
            else:
                raise
        _log(
            f"asr_model_ready device={device if cache_key == 'auto' else 'cpu'} "
            f"compute_type={compute_type if cache_key == 'auto' else 'int8'} "
            f"source={model_ref}"
        )
        _ASR_MODELS[cache_key] = model
        return model


def _load_eval_model() -> Any:
    global _EVAL_MODEL
    with _EVAL_MODEL_LOCK:
        if _EVAL_MODEL is not None:
            return _EVAL_MODEL
        Qwen3TTSModel = importlib.import_module("qwen_tts").Qwen3TTSModel
        # Keep reference embedding extraction off the inference GPU so review does not
        # compete with the live generation model for VRAM.
        opts = {"device_map": "cpu", "torch_dtype": _torch().float32}
        opts = {k: v for k, v in opts.items() if v is not None}
        model = Qwen3TTSModel.from_pretrained(BASE_MODEL_PATHS["qwen3-tts-0.6b"], **opts)
        try:
            model.model.eval()
        except Exception:
            pass
        _EVAL_MODEL = model
        _log("loaded_eval_model device=cpu model=qwen3-tts-0.6b")
        return _EVAL_MODEL


def _normalize_whisper_language(language: str | None) -> str | None:
    lang = (language or "").strip().lower().replace("_", "-")
    if not lang:
        return None
    if lang in _WHISPER_LANGUAGE_ALIASES:
        return _WHISPER_LANGUAGE_ALIASES[lang]
    return None if lang == "auto" else lang


def _transcribe_for_review(
    audio: np.ndarray,
    sr: int,
    language: str | None,
) -> tuple[str, float | None]:
    sf = importlib.import_module("soundfile")
    audio_1d = np.asarray(audio, dtype=np.float32).reshape(-1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(str(tmp_path), audio_1d, sr)
        lang = _normalize_whisper_language(language)

        attempts = [
            {"language": lang, "vad_filter": True, "beam_size": 3},
            {"language": lang, "vad_filter": False, "beam_size": 3},
        ]
        if lang is not None:
            attempts.append({"language": None, "vad_filter": False, "beam_size": 5})

        best_text = ""
        best_prob: float | None = None
        best_score = -1
        model_attempts = [("auto", False)]
        if _torch().cuda.is_available():
            model_attempts.append(("cpu", True))

        for model_name, prefer_cpu in model_attempts:
            try:
                model = _load_asr_model_for_device(prefer_cpu=prefer_cpu)
            except Exception as exc:
                _log(f"asr_model_init_failed device={model_name} error={exc}")
                continue

            for idx, attempt in enumerate(attempts, start=1):
                try:
                    segments, info = model.transcribe(
                        str(tmp_path),
                        language=attempt["language"],
                        beam_size=attempt["beam_size"],
                        vad_filter=attempt["vad_filter"],
                        condition_on_previous_text=False,
                    )
                except Exception as exc:
                    _log(
                        f"asr_review_attempt_failed device={model_name} "
                        f"attempt={idx} language={attempt['language'] or 'auto'} "
                        f"vad_filter={attempt['vad_filter']} error={exc}"
                    )
                    if prefer_cpu:
                        break
                    # If the GPU path fails, immediately retry the whole sequence on CPU.
                    break

                text = "".join(str(seg.text or "") for seg in segments).strip()
                normalized_len = len(_normalize_text(text))
                try:
                    prob = float(getattr(info, "language_probability", 0.0))
                except Exception:
                    prob = None
                if normalized_len > best_score:
                    best_text = text
                    best_prob = prob
                    best_score = normalized_len
                if text:
                    if idx > 1 or prefer_cpu:
                        _log(
                            f"asr_review_recovered device={model_name} attempt={idx} "
                            f"language={attempt['language'] or 'auto'} "
                            f"vad_filter={attempt['vad_filter']}"
                        )
                    return text, prob
        return best_text, best_prob
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return vec.copy()
    return vec / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def _resample_audio(audio: np.ndarray, sr: int, target_sr: int = 24000) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(sr) == int(target_sr):
        return arr
    librosa = importlib.import_module("librosa")
    return np.asarray(
        librosa.resample(arr, orig_sr=int(sr), target_sr=int(target_sr)),
        dtype=np.float32,
    )


def _speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    arr = _resample_audio(audio, sr, 24000)
    model = _load_eval_model()
    emb = model.model.extract_speaker_embedding(arr.astype(np.float32), 24000)
    if hasattr(emb, "detach"):
        emb = emb.detach()
    if hasattr(emb, "cpu"):
        emb = emb.cpu()
    if hasattr(emb, "numpy"):
        emb = emb.numpy()
    return _unit(np.asarray(emb, dtype=np.float32).reshape(-1))


def _pitch_stats(audio: np.ndarray, sr: int) -> tuple[float, float]:
    librosa = importlib.import_module("librosa")
    arr = _resample_audio(audio, sr, 24000)
    try:
        f0, _, _ = librosa.pyin(
            arr.astype(np.float32),
            fmin=float(librosa.note_to_hz("C2")),
            fmax=float(librosa.note_to_hz("C6")),
            sr=24000,
            frame_length=1024,
            hop_length=256,
        )
        voiced = f0[np.isfinite(f0)]
        if len(voiced) == 0:
            return 0.0, 0.0
        return float(np.median(voiced)), float(np.std(voiced))
    except Exception:
        return 0.0, 0.0


def _ratio_score(a: float, b: float, *, tolerance: float = 1.0) -> float:
    x = max(float(a), 1e-6)
    y = max(float(b), 1e-6)
    diff = abs(np.log(x / y))
    return float(np.exp(-diff * tolerance))


def _cached_reference_audio(storage: Any, r2_key: str) -> Path:
    key = hashlib.sha1(r2_key.encode("utf-8")).hexdigest()
    local = REFERENCE_CACHE_ROOT / f"{key}{Path(r2_key).suffix or '.wav'}"
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    storage.download_file(r2_key, local)
    return local


def _reference_similarity_metrics(
    *,
    storage: Any | None,
    audio: np.ndarray,
    sr: int,
    text: str,
    review_cfg: dict[str, Any],
) -> dict[str, float | str]:
    if storage is None:
        return {}

    reference_key = str(review_cfg.get("reference_audio_key", "") or "").strip()
    if not reference_key:
        return {}

    try:
        librosa = importlib.import_module("librosa")
        ref_path = _cached_reference_audio(storage, reference_key)
        ref_audio, ref_sr = librosa.load(str(ref_path), sr=None, mono=True)
        ref_audio_24 = _resample_audio(np.asarray(ref_audio, dtype=np.float32), int(ref_sr or 24000), 24000)
        gen_audio_24 = _resample_audio(audio, sr, 24000)
    except Exception as exc:
        _log(f"reference_review_failed key={reference_key} error={exc}")
        return {}

    result: dict[str, float | str] = {}

    if bool(review_cfg.get("enable_speaker", True)):
        try:
            ref_emb = _speaker_embedding(ref_audio_24, 24000)
            gen_emb = _speaker_embedding(gen_audio_24, 24000)
            speaker_cos = _cosine(gen_emb, ref_emb)
            result["speaker_cosine"] = float(speaker_cos)
            result["speaker_score"] = float(_clamp((speaker_cos - 0.85) / 0.15, 0.0, 1.0))
        except Exception as exc:
            _log(f"reference_speaker_review_failed key={reference_key} error={exc}")

    if bool(review_cfg.get("enable_style", True)):
        try:
            ref_pitch_median, ref_pitch_std = _pitch_stats(ref_audio_24, 24000)
            gen_pitch_median, gen_pitch_std = _pitch_stats(gen_audio_24, 24000)
            result["reference_pitch_median"] = float(ref_pitch_median)
            result["reference_pitch_std"] = float(ref_pitch_std)
            result["generated_pitch_median"] = float(gen_pitch_median)
            result["generated_pitch_std"] = float(gen_pitch_std)
            if ref_pitch_median > 0.0 and gen_pitch_median > 0.0:
                tone_score = (
                    (_ratio_score(gen_pitch_median, ref_pitch_median, tolerance=1.3) * 0.7)
                    + (_ratio_score(max(gen_pitch_std, 1.0), max(ref_pitch_std, 1.0), tolerance=1.1) * 0.3)
                )
                result["tone_score"] = float(_clamp(tone_score, 0.0, 1.0))
        except Exception as exc:
            _log(f"reference_style_review_failed key={reference_key} error={exc}")

    reference_text = str(review_cfg.get("reference_text", "") or "").strip()
    if bool(review_cfg.get("enable_speed", True)) and reference_text:
        try:
            ref_duration_sec = float(len(ref_audio_24) / 24000.0)
            gen_duration_sec = float(len(gen_audio_24) / 24000.0)
            ref_chars = len(_normalize_text(reference_text))
            gen_chars = len(_normalize_text(text))
            if ref_chars > 0 and gen_chars > 0 and ref_duration_sec > 0 and gen_duration_sec > 0:
                ref_cps = ref_chars / ref_duration_sec
                gen_cps = gen_chars / gen_duration_sec
                result["reference_chars_per_sec"] = float(ref_cps)
                result["generated_chars_per_sec"] = float(gen_cps)
                result["speed_score"] = float(
                    _clamp(_ratio_score(gen_cps, ref_cps, tolerance=1.6), 0.0, 1.0)
                )
        except Exception as exc:
            _log(f"reference_speed_review_failed key={reference_key} error={exc}")

    return result


def _model_supports_instruct(model: Any) -> bool:
    """0.6B models do not support instruct in custom_voice mode."""
    try:
        raw_size = str(getattr(model.model, "tts_model_size", "") or "")
    except Exception:
        return True
    normalized = "".join(ch for ch in raw_size.lower() if ch.isalnum())
    return normalized not in {"06b", "0b6"}


def _decode_params(job_input: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    vs = job_input.get("voice_settings") or {}
    stability = _clamp(_to_float(vs.get("stability")) or 0.85, 0.0, 1.0)
    similarity = _clamp(_to_float(vs.get("similarity_boost")) or 0.85, 0.0, 1.0)
    style = _clamp(_to_float(vs.get("style")) or 0.05, 0.0, 1.0)
    speed = _to_float(vs.get("speed"))
    variability = 1.0 - stability
    p: dict[str, Any] = {
        "temperature": _clamp(0.38 + (variability * 0.22) + (style * 0.08), 0.32, 0.75),
        "top_k": int(round(_clamp(18 + (variability * 18) + (style * 10), 12, 50))),
        "top_p": _clamp(0.86 + (style * 0.08) + (variability * 0.03), 0.82, 0.96),
        "repetition_penalty": _clamp(1.01 + (similarity * 0.06), 1.0, 1.10),
        "subtalker_temperature": _clamp(
            0.50 + (style * 0.18) + (variability * 0.10), 0.45, 0.85
        ),
        "subtalker_top_k": int(round(_clamp(18 + (style * 12) + (variability * 8), 12, 40))),
        "subtalker_top_p": _clamp(0.86 + (style * 0.08), 0.82, 0.96),
        "do_sample": True,
        "subtalker_dosample": True,
        "max_new_tokens": _adaptive_max_tokens(job_input.get("text", "")),
    }
    t = _to_float(job_input.get("temperature"))
    k = _to_int(job_input.get("top_k"))
    pp = _to_float(job_input.get("top_p"))
    rp = _to_float(job_input.get("repetition_penalty"))
    mx = _to_int(job_input.get("max_new_tokens"))
    st = _to_float(job_input.get("subtalker_temperature"))
    sk = _to_int(job_input.get("subtalker_top_k"))
    sp = _to_float(job_input.get("subtalker_top_p"))
    if t is not None:
        p["temperature"] = t
    if k is not None:
        p["top_k"] = k
    if pp is not None:
        p["top_p"] = pp
    if rp is not None:
        p["repetition_penalty"] = rp
    if mx is not None:
        p["max_new_tokens"] = mx
    if st is not None:
        p["subtalker_temperature"] = st
    if sk is not None:
        p["subtalker_top_k"] = sk
    if sp is not None:
        p["subtalker_top_p"] = sp
    return p, speed


def _seed(seed: int | None) -> None:
    if seed is None:
        return
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _download_checkpoint(
    storage: Any, voice_id: str, run_name: str, epoch: int, dst: Path
) -> None:
    start = time.time()
    prefix = f"checkpoints/{voice_id}/{run_name}/checkpoint-epoch-{epoch}/"
    objs = storage.list_prefix(prefix)
    if not objs:
        raise FileNotFoundError(
            f"Unknown voice_id/checkpoint: {voice_id}/{run_name}/checkpoint-epoch-{epoch}"
        )
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(storage.download_checkpoint, voice_id, run_name, epoch, dst)
    try:
        fut.result(timeout=R2_TIMEOUT_SEC)
    except TimeoutError as exc:
        fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"R2 download timed out after {R2_TIMEOUT_SEC}s") from exc
    finally:
        pool.shutdown(wait=False)
    _log(
        f"checkpoint_download_ms={int((time.time() - start) * 1000)} files={len(objs)}"
    )


def _download_full_checkpoint(storage: Any, r2_prefix: str, local_dir: Path) -> None:
    start = time.time()
    prefix = r2_prefix.strip().strip("/")
    if not prefix:
        raise ValueError("Missing checkpoint_info.r2_prefix for full checkpoint")
    prefix = f"{prefix}/"
    objs = storage.list_prefix(prefix)
    if not objs:
        raise FileNotFoundError(f"Unknown full checkpoint prefix: {r2_prefix}")

    local_dir.mkdir(parents=True, exist_ok=True)

    def _download_all() -> None:
        for obj in objs:
            key = obj["key"]
            relative = key[len(prefix) :]
            if not relative:
                continue
            storage.download_file(key, local_dir / relative)

    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(_download_all)
    try:
        fut.result(timeout=R2_TIMEOUT_SEC)
    except TimeoutError as exc:
        fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise TimeoutError(f"R2 download timed out after {R2_TIMEOUT_SEC}s") from exc
    finally:
        pool.shutdown(wait=False)

    _log(
        f"full_checkpoint_download_ms={int((time.time() - start) * 1000)} files={len(objs)}"
    )


def _merge_checkpoint(base_dir: Path, delta_dir: Path, merged_dir: Path) -> None:
    start = time.time()
    tmp = merged_dir.with_name(f"{merged_dir.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(base_dir, tmp)
    for src in delta_dir.rglob("*"):
        if src.is_file():
            out = tmp / src.relative_to(delta_dir)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    shutil.move(str(tmp), str(merged_dir))
    if delta_dir.exists():
        shutil.rmtree(delta_dir, ignore_errors=True)
    _log(f"checkpoint_merge_ms={int((time.time() - start) * 1000)} path={merged_dir}")


def _artifact_root_for_checkpoint_dir(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent


def _safe_remove_tree(path: Path | None) -> None:
    if path is None:
        return
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        checkpoint_root = CHECKPOINT_ROOT.resolve()
    except Exception:
        checkpoint_root = CHECKPOINT_ROOT
    if resolved == checkpoint_root or checkpoint_root not in resolved.parents:
        return
    shutil.rmtree(resolved, ignore_errors=True)


def _current_cache_roots(extra_keep: Path | None = None) -> set[Path]:
    roots: set[Path] = set()
    with _MODEL_LOCK:
        for _, _, checkpoint_dir in MODEL_CACHE.values():
            roots.add(_artifact_root_for_checkpoint_dir(checkpoint_dir))
    if extra_keep is not None:
        roots.add(extra_keep)
    return roots


def _prune_stale_checkpoint_dirs(extra_keep: Path | None = None) -> None:
    if not CHECKPOINT_ROOT.exists():
        return
    keep_roots = {root.resolve() for root in _current_cache_roots(extra_keep)}
    for candidate in CHECKPOINT_ROOT.rglob("checkpoint-epoch-*"):
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in keep_roots:
            continue
        shutil.rmtree(resolved, ignore_errors=True)


def _ensure_merged_checkpoint(
    storage: Any,
    voice_id: str,
    run_name: str | None,
    epoch: int | None,
    model_id: str,
    checkpoint_type: str = "delta",
    r2_prefix: str | None = None,
) -> Path:
    normalized_prefix: str | None = None
    delta_dir: Path | None = None
    if checkpoint_type == "full":
        if not r2_prefix:
            raise ValueError("Missing checkpoint_info.r2_prefix for full checkpoint")
        normalized_prefix = r2_prefix.strip().strip("/")
        if not normalized_prefix:
            raise ValueError("Missing checkpoint_info.r2_prefix for full checkpoint")
        if ".." in normalized_prefix.split("/"):
            raise ValueError("Invalid r2_prefix: path traversal not allowed")
        root = CHECKPOINT_ROOT / voice_id / "full" / normalized_prefix
        merged_dir = root / f"merged-{model_id}"
        merge_key = f"{voice_id}/full/{normalized_prefix}/{model_id}"
    else:
        if not run_name or epoch is None:
            raise ValueError(
                "Missing checkpoint_info.run_name or checkpoint_info.epoch"
            )
        root = CHECKPOINT_ROOT / voice_id / run_name / f"checkpoint-epoch-{epoch}"
        delta_dir = root / "delta"
        merged_dir = root / f"merged-{model_id}"
        merge_key = f"{voice_id}/{run_name}/{epoch}/{model_id}"

    with _MERGE_LOCKS_LOCK:
        if merge_key not in _MERGE_LOCKS:
            _MERGE_LOCKS[merge_key] = threading.Lock()
        merge_lock = _MERGE_LOCKS[merge_key]

    with merge_lock:
        _prune_stale_checkpoint_dirs(extra_keep=root)
        if (merged_dir / "model.safetensors").exists():
            return merged_dir

        if checkpoint_type == "full":
            if normalized_prefix is None:
                raise ValueError(
                    "Missing checkpoint_info.r2_prefix for full checkpoint"
                )
            if merged_dir.exists():
                shutil.rmtree(merged_dir)
            _download_full_checkpoint(storage, normalized_prefix, merged_dir)
        else:
            if run_name is None or epoch is None or delta_dir is None:
                raise ValueError(
                    "Missing checkpoint_info.run_name or checkpoint_info.epoch"
                )
            if not delta_dir.exists() or not any(delta_dir.iterdir()):
                delta_dir.mkdir(parents=True, exist_ok=True)
                _download_checkpoint(storage, voice_id, run_name, epoch, delta_dir)
            base = Path(BASE_MODEL_PATHS[model_id])
            if not base.exists():
                raise FileNotFoundError(f"Base model not found in image: {base}")
            _merge_checkpoint(base, delta_dir, merged_dir)
        return merged_dir


def _load_model(checkpoint_dir: Path) -> tuple[Any, Any | None]:
    Qwen3TTSModel = importlib.import_module("qwen_tts").Qwen3TTSModel
    device = _resolve_device()
    dtype = _resolve_dtype(device)
    attn = "sdpa" if device.startswith("cuda") else None
    start = time.time()
    opts = [
        {"device_map": device, "dtype": dtype, "attn_implementation": attn},
        {"device_map": device, "torch_dtype": dtype, "attn_implementation": attn},
        {"device_map": device, "dtype": dtype},
        {"device_map": device},
    ]
    model = None
    last_error: Exception | None = None
    for kw in opts:
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            model = Qwen3TTSModel.from_pretrained(str(checkpoint_dir), **kw)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        if last_error:
            raise RuntimeError(f"Model loading failure: {last_error}") from last_error
        raise RuntimeError("Model loading failure")
    try:
        model.model.eval()
    except Exception:
        pass
    _log(
        f"model_loading_ms={int((time.time() - start) * 1000)} device={device} path={checkpoint_dir}"
    )
    return model, getattr(model, "tokenizer", None)


def _evict_lru() -> None:
    torch = _torch()
    if not MODEL_CACHE:
        return
    voice_id, (model, _, checkpoint_dir) = MODEL_CACHE.popitem(last=False)
    CACHE_IDENTITY.pop(voice_id, None)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _safe_remove_tree(_artifact_root_for_checkpoint_dir(checkpoint_dir))
    _log(f"cache_evict voice_id={voice_id}")


def _get_model(
    storage: Any,
    voice_id: str,
    run_name: str | None,
    epoch: int | None,
    model_id: str,
    checkpoint_type: str = "delta",
    r2_prefix: str | None = None,
) -> Any:
    torch = _torch()
    if checkpoint_type == "full":
        normalized_prefix = (r2_prefix or "").strip().strip("/")
        identity = f"{model_id}:full:{normalized_prefix}"
    else:
        identity = f"{model_id}:delta:{run_name}:{epoch}"
    stale_model: Any | None = None
    stale_checkpoint_dir: Path | None = None
    with _MODEL_LOCK:
        if voice_id in MODEL_CACHE and CACHE_IDENTITY.get(voice_id) == identity:
            MODEL_CACHE.move_to_end(voice_id)
            return MODEL_CACHE[voice_id][0]
        if voice_id in MODEL_CACHE:
            stale_model, _, stale_checkpoint_dir = MODEL_CACHE.pop(voice_id)
            CACHE_IDENTITY.pop(voice_id, None)
    if stale_model is not None:
        del stale_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _safe_remove_tree(
        _artifact_root_for_checkpoint_dir(stale_checkpoint_dir)
        if stale_checkpoint_dir is not None
        else None
    )
    merged_dir = _ensure_merged_checkpoint(
        storage=storage,
        voice_id=voice_id,
        run_name=run_name,
        epoch=epoch,
        model_id=model_id,
        checkpoint_type=checkpoint_type,
        r2_prefix=r2_prefix,
    )
    model, tokenizer = _load_model(merged_dir)
    with _MODEL_LOCK:
        while len(MODEL_CACHE) >= MAX_CACHED_MODELS:
            _evict_lru()
        MODEL_CACHE[voice_id] = (model, tokenizer, merged_dir)
        CACHE_IDENTITY[voice_id] = identity
        MODEL_CACHE.move_to_end(voice_id)
    return model


def _generate_audio(
    model: Any,
    text: str,
    speaker_name: str,
    language: str | None,
    instruct: str | None,
    params: dict[str, Any],
) -> tuple[np.ndarray, int, int]:
    torch = _torch()
    lang = (language or "").strip()
    lang = None if not lang or lang.lower() == "auto" else lang
    inst = (instruct or "").strip() or None
    # 0.6B models do not support instruct in custom_voice mode
    if inst and not _model_supports_instruct(model):
        inst = None
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    start = time.time()
    gen_kwargs = {
        "temperature": params["temperature"],
        "top_k": params["top_k"],
        "top_p": params["top_p"],
        "repetition_penalty": params["repetition_penalty"],
        "subtalker_temperature": params.get("subtalker_temperature"),
        "subtalker_top_k": params.get("subtalker_top_k"),
        "subtalker_top_p": params.get("subtalker_top_p"),
        "do_sample": params.get("do_sample", True),
        "subtalker_dosample": params.get("subtalker_dosample", True),
        "max_new_tokens": params["max_new_tokens"],
    }
    try:
        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker_name,
            language=lang,
            instruct=inst,
            **gen_kwargs,
        )
    except TypeError:
        # Fallback: model API may not accept all kwargs
        fallback_keys = [
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "subtalker_temperature",
            "subtalker_top_k",
            "subtalker_top_p",
            "do_sample",
            "subtalker_dosample",
            "max_new_tokens",
        ]
        fallback = {k: gen_kwargs[k] for k in fallback_keys if k in gen_kwargs}
        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                speaker=speaker_name,
                language=lang,
                instruct=inst,
                **fallback,
            )
        except TypeError:
            wavs, sr = model.generate_custom_voice(
                text=text, speaker=speaker_name, language=lang, instruct=inst
            )
    first_wav = wavs[0]
    if hasattr(first_wav, "detach"):
        first_wav = first_wav.detach()
    if hasattr(first_wav, "cpu"):
        first_wav = first_wav.cpu()
    if hasattr(first_wav, "numpy"):
        first_wav = first_wav.numpy()

    audio = np.asarray(first_wav, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    return audio, int(sr), int((time.time() - start) * 1000)


def _verify_quality(
    audio: np.ndarray,
    sr: int,
    text: str,
    params: dict[str, Any],
    *,
    language: str | None = None,
    review: dict[str, Any] | None = None,
    storage: Any | None = None,
) -> dict[str, Any]:
    del params
    review_cfg = review or {}
    audio_flat = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite_mask = np.isfinite(audio_flat)
    finite_audio = audio_flat[finite_mask]
    safe_sr = max(int(sr), 1)
    duration_sec = float(len(audio_flat) / safe_sr) if len(audio_flat) > 0 else 0.0

    expected_duration_sec = max(0.3, _expected_seconds(text or ""))
    ratio = duration_sec / max(expected_duration_sec, 1e-6)
    # Wider acceptable band: TTS naturally varies in pacing
    if 0.5 <= ratio <= 2.0:
        duration_score = 1.0
    elif 0.3 <= ratio < 0.5:
        duration_score = (ratio - 0.3) / 0.2
    elif 2.0 < ratio <= 3.5:
        duration_score = (3.5 - ratio) / 1.5
    else:
        duration_score = 0.0

    if len(audio_flat) > 0 and len(finite_audio) > 0:
        rms = float(np.sqrt(np.mean(np.square(finite_audio))))
        peak = float(np.max(np.abs(finite_audio)))
        silence_ratio = float(np.mean(np.abs(finite_audio) < 0.001))
        if len(finite_audio) > 1:
            zcr = float(
                np.mean(
                    np.not_equal(
                        np.signbit(finite_audio[:-1]), np.signbit(finite_audio[1:])
                    )
                )
            )
        else:
            zcr = 1.0
    else:
        rms = 0.0
        peak = 0.0
        silence_ratio = 1.0
        zcr = 1.0

    if 0.03 <= rms <= 0.15:
        rms_score = 1.0
    elif rms < 0.03:
        rms_score = _clamp(rms / 0.03, 0.0, 1.0)
    else:
        rms_score = _clamp((0.3 - rms) / 0.15, 0.0, 1.0)

    if peak < 0.95:
        peak_score = 1.0
    else:
        peak_score = _clamp((1.0 - peak) / 0.05, 0.0, 1.0)

    if silence_ratio <= 0.5:
        silence_score = 1.0
    else:
        silence_score = _clamp((0.95 - silence_ratio) / 0.45, 0.0, 1.0)

    if 0.02 <= zcr <= 0.22:
        zcr_score = 1.0
    elif zcr < 0.02:
        zcr_score = _clamp(zcr / 0.02, 0.0, 1.0)
    elif zcr <= 0.40:
        zcr_score = _clamp((0.40 - zcr) / 0.18, 0.0, 1.0)
    else:
        zcr_score = 0.0

    health_score = (
        (rms_score * 0.3)
        + (peak_score * 0.2)
        + (silence_score * 0.2)
        + (zcr_score * 0.3)
    )

    finite_score = 1.0 if len(audio_flat) > 0 and finite_mask.all() else 0.0
    non_empty_score = 1.0 if len(audio_flat) > 0 else 0.0
    min_duration_score = _clamp(duration_sec / 0.3, 0.0, 1.0)
    stability_score = (
        finite_score * 0.5 + non_empty_score * 0.2 + min_duration_score * 0.3
    )

    asr_text = ""
    asr_similarity = -1.0
    asr_score = 0.0
    if bool(review_cfg.get("enable_asr")) and text.strip():
        try:
            asr_text, _ = _transcribe_for_review(audio, sr, language)
            asr_similarity = _asr_similarity(text, asr_text)
            asr_score = float(_clamp(asr_similarity, 0.0, 1.0))
        except Exception as exc:
            _log(f"asr_review_failed error={exc}")

    reference_metrics = _reference_similarity_metrics(
        storage=storage,
        audio=audio,
        sr=sr,
        text=text,
        review_cfg=review_cfg,
    )

    if bool(review_cfg.get("enable_asr")) or reference_metrics:
        weighted_parts: list[tuple[float, float]] = [
            (_clamp(duration_score, 0.0, 1.0), 0.22),
            (_clamp(health_score, 0.0, 1.0), 0.33),
            (_clamp(stability_score, 0.0, 1.0), 0.15),
        ]
        if bool(review_cfg.get("enable_asr")):
            weighted_parts.append((_clamp(asr_score, 0.0, 1.0), 0.15))
        speaker_score = reference_metrics.get("speaker_score")
        tone_score = reference_metrics.get("tone_score")
        speed_score = reference_metrics.get("speed_score")
        if isinstance(speaker_score, (int, float)):
            weighted_parts.append((_clamp(float(speaker_score), 0.0, 1.0), 0.10))
        if isinstance(tone_score, (int, float)):
            weighted_parts.append((_clamp(float(tone_score), 0.0, 1.0), 0.03))
        if isinstance(speed_score, (int, float)):
            weighted_parts.append((_clamp(float(speed_score), 0.0, 1.0), 0.02))
        total_weight = sum(weight for _, weight in weighted_parts) or 1.0
        overall_score = sum(score * weight for score, weight in weighted_parts) / total_weight
    else:
        overall_score = (
            _clamp(duration_score, 0.0, 1.0) * 0.3
            + _clamp(health_score, 0.0, 1.0) * 0.5
            + _clamp(stability_score, 0.0, 1.0) * 0.2
        )

    result = {
        "overall_score": float(_clamp(overall_score, 0.0, 1.0)),
        "duration_score": float(_clamp(duration_score, 0.0, 1.0)),
        "health_score": float(_clamp(health_score, 0.0, 1.0)),
        "stability_score": float(_clamp(stability_score, 0.0, 1.0)),
    }
    if bool(review_cfg.get("enable_asr")):
        result["asr_score"] = float(_clamp(asr_score, 0.0, 1.0))
        result["asr_similarity"] = float(max(asr_similarity, 0.0))
        result["asr_text"] = asr_text
    for key, value in reference_metrics.items():
        result[key] = value
    return result


def _encode_wav(audio: np.ndarray, sr: int) -> str:
    sf = importlib.import_module("soundfile")
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    request_start = time.time()
    try:
        inp = job["input"]
        text = str(inp.get("text", "")).strip()
        voice_id = str(inp.get("voice_id", "")).strip()
        speaker_name = str(inp.get("speaker_name", "")).strip()
        if not text:
            return {"error": "Missing required field: text"}
        if not voice_id:
            return {"error": "Missing required field: voice_id"}
        if not speaker_name:
            return {"error": "Missing required field: speaker_name"}
        try:
            _validate_id(voice_id, "voice_id")
        except ValueError as exc:
            return {"error": str(exc)}

        ck = inp.get("checkpoint_info") or {}
        checkpoint_type = str(ck.get("type") or "delta").strip().lower()
        if checkpoint_type not in {"delta", "full"}:
            return {"error": "checkpoint_info.type must be 'delta' or 'full'"}

        run_name = str(ck.get("run_name", "")).strip()
        epoch = _to_int(ck.get("epoch"))
        r2_prefix = str(ck.get("r2_prefix", "")).strip()
        if r2_prefix and (".." in r2_prefix.split("/") or r2_prefix.startswith("/")):
            return {
                "error": "Invalid checkpoint_info.r2_prefix: path traversal not allowed"
            }

        if checkpoint_type == "full":
            if not r2_prefix:
                return {
                    "error": "Missing checkpoint_info.r2_prefix for full checkpoint"
                }
        else:
            if not run_name or epoch is None:
                return {
                    "error": "Missing checkpoint_info.run_name or checkpoint_info.epoch"
                }
            try:
                _validate_id(run_name, "run_name")
            except ValueError as exc:
                return {"error": str(exc)}
        ck_voice = str(ck.get("voice_id", "")).strip()
        if ck_voice and ck_voice != voice_id:
            return {"error": "checkpoint_info.voice_id does not match voice_id"}

        model_id = str(inp.get("model_id") or DEFAULT_MODEL_ID).strip().lower()
        if model_id not in BASE_MODEL_PATHS:
            return {"error": f"Unsupported model_id: {model_id}"}

        params, speed_hint = _decode_params(inp)
        if model_id == "qwen3-tts-0.6b":
            if inp.get("temperature") is None:
                params["temperature"] = min(float(params.get("temperature", 0.7)), 0.45)
            if inp.get("top_p") is None:
                params["top_p"] = min(float(params.get("top_p", 0.88)), 0.88)
            if inp.get("top_k") is None:
                params["top_k"] = min(int(params.get("top_k", 24)), 30)
            if inp.get("repetition_penalty") is None:
                params["repetition_penalty"] = max(
                    float(params.get("repetition_penalty", 1.05)), 1.04
                )
            if inp.get("subtalker_temperature") is None:
                params["subtalker_temperature"] = min(
                    float(params.get("subtalker_temperature", 0.55)), 0.7
                )
        base_seed = _to_int(inp.get("seed"))
        seed_anchor = base_seed if base_seed is not None else int(time.time() * 1000)
        num_candidates = _to_int(inp.get("num_candidates"))
        if num_candidates is None:
            num_candidates = 3 if model_id == "qwen3-tts-0.6b" else 2
        num_candidates = max(1, min(5, num_candidates))
        review_cfg = inp.get("quality_review") or {}
        allow_below_threshold = bool(
            review_cfg.get("allow_below_threshold", False)
            if isinstance(review_cfg, dict)
            else False
        )

        storage = _r2_storage_cls()()
        try:
            model = _get_model(
                storage=storage,
                voice_id=voice_id,
                run_name=run_name,
                epoch=epoch,
                model_id=model_id,
                checkpoint_type=checkpoint_type,
                r2_prefix=r2_prefix,
            )
        except FileNotFoundError as exc:
            return {"error": f"Unknown voice_id/checkpoint: {exc}"}
        except TimeoutError as exc:
            return {"error": f"R2 download failure: {exc}"}
        except Exception as exc:
            return {"error": f"Model loading failure: {exc}"}

        retries = 0
        max_rounds = 3
        best_candidate: dict[str, Any] | None = None
        candidate_scores: list[float] = []
        last_error: Exception | None = None
        for round_idx in range(max_rounds):
            round_params = dict(params)
            if round_idx > 0:
                round_params["temperature"] = max(
                    0.1, float(round_params.get("temperature", 1.0)) * 0.8
                )
                round_params["top_p"] = max(
                    0.2, float(round_params.get("top_p", 1.0)) * 0.94
                )
                if "subtalker_temperature" in round_params:
                    round_params["subtalker_temperature"] = max(
                        0.35,
                        float(round_params.get("subtalker_temperature", 0.6)) * 0.9,
                    )
                retries = round_idx
                _log(
                    f"quality_retry round={round_idx + 1} "
                    f"temperature={round_params['temperature']:.3f} "
                    f"top_p={round_params['top_p']:.3f}"
                )

            round_best: dict[str, Any] | None = None
            for candidate_idx in range(num_candidates):
                candidate_seed = seed_anchor + (round_idx * 1000) + candidate_idx
                _seed(candidate_seed)
                try:
                    audio, sr, gen_ms = _generate_audio(
                        model=model,
                        text=text,
                        speaker_name=speaker_name,
                        language=inp.get("language", "auto"),
                        instruct=inp.get("instruct"),
                        params=round_params,
                    )
                except Exception as exc:
                    last_error = exc
                    _log(
                        f"candidate_failed round={round_idx + 1} "
                        f"idx={candidate_idx + 1} error={exc}"
                    )
                    continue

                quality = _verify_quality(
                    audio,
                    sr,
                    text,
                    round_params,
                    language=inp.get("language", "auto"),
                    review=review_cfg if isinstance(review_cfg, dict) else None,
                    storage=storage,
                )
                candidate_score = float(quality["overall_score"])
                candidate_scores.append(candidate_score)
                candidate = {
                    "audio": audio,
                    "sr": sr,
                    "gen_ms": gen_ms,
                    "quality": quality,
                    "round": round_idx + 1,
                    "candidate": candidate_idx + 1,
                }
                if (
                    round_best is None
                    or candidate_score > round_best["quality"]["overall_score"]
                ):
                    round_best = candidate

            if round_best is not None and (
                best_candidate is None
                or round_best["quality"]["overall_score"]
                > best_candidate["quality"]["overall_score"]
            ):
                best_candidate = round_best

            best_score = (
                float(best_candidate["quality"]["overall_score"])
                if best_candidate is not None
                else 0.0
            )
            if best_score >= 0.9:
                break

        if best_candidate is None:
            return {
                "error": f"Generation failure: {last_error or 'all candidates failed'}"
            }

        audio = best_candidate["audio"]
        sr = best_candidate["sr"]
        gen_ms = best_candidate["gen_ms"]
        quality = best_candidate["quality"]
        min_accept_score = 0.9 if model_id == "qwen3-tts-0.6b" else 0.82
        allow_below_threshold = bool(review_cfg.get("allow_below_threshold", False))
        below_threshold = float(quality["overall_score"]) < min_accept_score
        accepted = not below_threshold
        if below_threshold and not allow_below_threshold:
            return {
                "error": (
                    f"Generated audio below quality threshold: "
                    f"overall_score={float(quality['overall_score']):.3f} "
                    f"required={min_accept_score:.3f}"
                )
            }

        torch = _torch()
        duration_ms = int((len(audio) / max(sr, 1)) * 1000)
        gpu_peak = (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        )
        _log(
            f"generation_ms={gen_ms} audio_duration_ms={duration_ms} gpu_max_memory_allocated={gpu_peak}"
        )

        out: dict[str, Any] = {
            "audio": _encode_wav(audio, sr),
            "sample_rate": sr if sr > 0 else 24000,
            "duration_ms": duration_ms,
            "format": "wav",
            "voice_id": voice_id,
            "latency_ms": int((time.time() - request_start) * 1000),
            "quality": {
                "overall_score": float(quality["overall_score"]),
                "duration_score": float(quality["duration_score"]),
                "health_score": float(quality["health_score"]),
                "stability_score": float(quality["stability_score"]),
                "accepted": accepted,
                "required_score": float(min_accept_score),
                "candidates_generated": len(candidate_scores),
                "candidate_scores": [float(s) for s in candidate_scores],
                "retries": retries,
            },
        }
        if below_threshold:
            out["warning"] = (
                "Generated audio below quality threshold: "
                f"overall_score={float(quality['overall_score']):.3f} "
                f"required={min_accept_score:.3f}"
            )
        if "asr_score" in quality:
            out["quality"]["asr_score"] = float(quality["asr_score"])
            out["quality"]["asr_similarity"] = float(quality.get("asr_similarity", 0.0))
            out["quality"]["asr_text"] = str(quality.get("asr_text", ""))
        for key in [
            "speaker_cosine",
            "speaker_score",
            "tone_score",
            "speed_score",
            "reference_pitch_median",
            "reference_pitch_std",
            "generated_pitch_median",
            "generated_pitch_std",
            "reference_chars_per_sec",
            "generated_chars_per_sec",
        ]:
            if key in quality:
                raw_value = quality[key]
                out["quality"][key] = float(raw_value) if isinstance(raw_value, (int, float)) else raw_value
        if speed_hint is not None:
            out["speed_hint"] = speed_hint
            out["speed_hint_note"] = (
                "speed is accepted as a hint but has no direct Qwen3-TTS parameter"
            )
        if not accepted:
            out["warning"] = (
                f"Generated audio below quality threshold: "
                f"overall_score={float(quality['overall_score']):.3f} "
                f"required={min_accept_score:.3f}"
            )
        return out
    except Exception as exc:
        return {"error": f"Unhandled error: {exc}"}


if __name__ == "__main__":
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    _log(f"handler_start cwd={os.getcwd()}")
    _runpod().serverless.start({"handler": handler})
